from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate_hat_gray import _forward_gray_once, build_hat, load_hat_weights
from evaluate_sst_gray import build_sst, forward_sst_gray, normalize_state
from make_hat_submission import (
    DATA,
    ROOT,
    parse_tta_modes,
    zip_submission,
)
from runtime import INTERP, _aug, _deaug, apply_postprocess, image_to_tensor, interp_tensor, tensor_to_image


def validate_weights(weight_cur: float, weight_prev: float, weight_sst: float) -> None:
    total = weight_cur + weight_prev + weight_sst
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ensemble weights must sum to 1.0, got {total:.8f}")
    if min(weight_cur, weight_prev, weight_sst) < 0.0:
        raise ValueError(
            f"Ensemble weights must be non-negative, got "
            f"cur={weight_cur} prev={weight_prev} sst={weight_sst}"
        )


def forward_hat_tta_batched(
    model: torch.nn.Module,
    lr: torch.Tensor,
    scale: int,
    gray_mode: str,
    native_io: bool,
    use_tta: bool,
    modes: list[int] | None,
    batch_size: int,
) -> torch.Tensor:
    if not use_tta:
        return _forward_gray_once(model, lr, scale, gray_mode, native_io=native_io)
    selected = modes if modes is not None else list(range(8))
    if selected == [0]:
        return _forward_gray_once(model, lr, scale, gray_mode, native_io=native_io)
    groups: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = {}
    for mode in selected:
        aug = _aug(lr, mode)
        groups.setdefault(tuple(aug.shape[-2:]), []).append((mode, aug))
    preds: list[torch.Tensor] = []
    for items in groups.values():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch = torch.cat([aug for _, aug in chunk], dim=0)
            out = _forward_gray_once(model, batch, scale, gray_mode, native_io=native_io)
            for idx, (mode, _) in enumerate(chunk):
                preds.append(_deaug(out[idx : idx + 1], mode))
    return torch.stack(preds).mean(dim=0)


def forward_sst_tta_batched(
    model: torch.nn.Module,
    lr: torch.Tensor,
    gray_mode: str,
    use_tta: bool,
    batch_size: int,
) -> torch.Tensor:
    if not use_tta:
        return forward_sst_gray(model, lr, gray_mode, tta=False)
    groups: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = {}
    for mode in range(8):
        aug = _aug(lr, mode)
        groups.setdefault(tuple(aug.shape[-2:]), []).append((mode, aug))
    preds: list[torch.Tensor] = []
    for items in groups.values():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch = torch.cat([aug for _, aug in chunk], dim=0)
            out = forward_sst_gray(model, batch, gray_mode, tta=False)
            for idx, (mode, _) in enumerate(chunk):
                preds.append(_deaug(out[idx : idx + 1], mode))
    return torch.stack(preds).mean(dim=0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DATA / "初赛测试集" / "input_320"))
    parser.add_argument("--team-name", default="fqy_hat_hat_ssttta_275225500_raw")
    parser.add_argument("--zip-path")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--hat-cur-weights", required=True)
    parser.add_argument("--hat-cur-variant", choices=["l", "m"], default="l")
    parser.add_argument("--hat-cur-param-key", default="state_dict")
    parser.add_argument("--hat-cur-native-io", action="store_true")
    parser.add_argument("--hat-cur-tta", action="store_true")
    parser.add_argument("--hat-cur-tta-modes", default="")
    parser.add_argument("--hat-prev-weights", required=True)
    parser.add_argument("--hat-prev-variant", choices=["l", "m"], default="l")
    parser.add_argument("--hat-prev-param-key", default="state_dict")
    parser.add_argument("--hat-prev-native-io", action="store_true")
    parser.add_argument("--hat-prev-tta", action="store_true")
    parser.add_argument("--hat-prev-tta-modes", default="")
    parser.add_argument("--sst-weights", required=True)
    parser.add_argument(
        "--sst-variant",
        choices=["light", "light_plus", "base", "base_plus", "large", "large_plus", "xl_plus"],
        default="xl_plus",
    )
    parser.add_argument("--sst-model-scale", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--sst-param-key", default="params_ema")
    parser.add_argument("--sst-tta", action="store_true")
    parser.add_argument("--weight-cur", type=float, default=0.275)
    parser.add_argument("--weight-prev", type=float, default=0.225)
    parser.add_argument("--weight-sst", type=float, default=0.500)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--sst-gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--interp", choices=["raw", *sorted(INTERP)], default="raw")
    parser.add_argument("--blend-interp", type=float, default=0.0)
    parser.add_argument("--sharpen-amount", type=float, default=0.0)
    parser.add_argument("--sharpen-radius", type=float, default=1.0)
    parser.add_argument("--clip-mode", choices=["hard", "match-base", "none"], default="hard")
    parser.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    parser.add_argument("--tta-batch-size", type=int, default=4)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    validate_weights(args.weight_cur, args.weight_prev, args.weight_sst)
    if args.tta_batch_size < 1:
        raise ValueError(f"--tta-batch-size must be positive, got {args.tta_batch_size}")
    tta_cur_modes = parse_tta_modes(args.hat_cur_tta_modes, args.hat_cur_tta)
    tta_prev_modes = parse_tta_modes(args.hat_prev_tta_modes, args.hat_prev_tta)

    input_dir = Path(args.input_dir)
    names = sorted(path.name for path in input_dir.glob("*.png"))
    if not names:
        raise ValueError(f"No PNG inputs in {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    hat_cur = build_hat(
        args.hat_cur_variant,
        args.scale,
        args.use_checkpoint,
        native_io=args.hat_cur_native_io,
    ).to(device).eval()
    load_hat_weights(
        hat_cur,
        args.hat_cur_weights,
        args.hat_cur_param_key,
        native_io=args.hat_cur_native_io,
        gray_mode=args.gray_mode,
    )

    hat_prev = build_hat(
        args.hat_prev_variant,
        args.scale,
        args.use_checkpoint,
        native_io=args.hat_prev_native_io,
    ).to(device).eval()
    load_hat_weights(
        hat_prev,
        args.hat_prev_weights,
        args.hat_prev_param_key,
        native_io=args.hat_prev_native_io,
        gray_mode=args.gray_mode,
    )

    sst = build_sst(args.sst_variant, args.sst_model_scale).to(device).eval()
    sst_ckpt = torch.load(args.sst_weights, map_location="cpu")
    sst.load_state_dict(normalize_state(sst_ckpt, args.sst_param_key), strict=True)

    package_dir = ROOT / "submission" / args.team_name
    prelim = package_dir / "preliminary"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    prelim.mkdir(parents=True, exist_ok=True)

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    amp_enabled = device.type == "cuda" and args.amp != "off"
    interp = None if args.interp == "raw" else args.interp
    print(
        f"device={device} images={len(names)} "
        f"weights=cur:{args.weight_cur:.6f},prev:{args.weight_prev:.6f},sst:{args.weight_sst:.6f} "
        f"hat_cur_tta={args.hat_cur_tta} hat_prev_tta={args.hat_prev_tta} sst_tta={args.sst_tta} "
        f"interp={args.interp} blend={args.blend_interp} tta_batch_size={args.tta_batch_size}"
    )
    for name in tqdm(names):
        img = cv2.imread(str(input_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(input_dir / name)
        lr = image_to_tensor(img, device)
        target_size = (lr.shape[-2] * args.scale, lr.shape[-1] * args.scale)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            pred_cur = forward_hat_tta_batched(
                hat_cur,
                lr,
                args.scale,
                args.gray_mode,
                args.hat_cur_native_io,
                args.hat_cur_tta,
                tta_cur_modes,
                args.tta_batch_size,
            ).float()
            pred_prev = forward_hat_tta_batched(
                hat_prev,
                lr,
                args.scale,
                args.gray_mode,
                args.hat_prev_native_io,
                args.hat_prev_tta,
                tta_prev_modes,
                args.tta_batch_size,
            ).float()
            pred_sst = forward_sst_tta_batched(
                sst,
                lr,
                args.sst_gray_mode,
                args.sst_tta,
                args.tta_batch_size,
            ).float()
        if pred_sst.shape[-2:] != target_size:
            pred_sst = F.interpolate(pred_sst, size=target_size, mode="area")
        pred = (
            pred_cur * args.weight_cur
            + pred_prev * args.weight_prev
            + pred_sst * args.weight_sst
        )
        base = interp_tensor(lr, args.scale, interp) if interp is not None else None
        pred = apply_postprocess(
            pred.float(),
            base,
            lr=lr,
            blend_interp=args.blend_interp,
            sharpen_amount=args.sharpen_amount,
            sharpen_radius=args.sharpen_radius,
            clip_mode=args.clip_mode,
        )
        cv2.imwrite(str(prelim / name), tensor_to_image(pred))

    (package_dir / "README.md").write_text(
        "# RCWN first-stage x2 submission\n\n"
        "Generated by legal pure model inference with a fixed image-space ensemble: "
        "current native-io HAT-L 8-TTA, previous HAT-L 8-TTA, and public "
        "SSTXLarge_Plus_DFLIP_X2 8-TTA. No training HR image is copied, substituted, "
        "retrieved, or provided as inference input.\n",
        encoding="utf-8",
    )
    zip_path = Path(args.zip_path) if args.zip_path else ROOT / "submission" / f"{args.team_name}.zip"
    zip_submission(package_dir, zip_path)
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
