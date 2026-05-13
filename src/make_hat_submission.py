from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path


def _prefer_torch_cuda_libs() -> None:
    base = "/usr/local/lib/python3.12/dist-packages/nvidia"
    libs = [
        os.path.join(base, "cublas", "lib"),
        os.path.join(base, "cuda_runtime", "lib"),
        os.path.join(base, "cudnn", "lib"),
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = ":".join([p for p in libs if os.path.isdir(p)])
    if not prefix:
        return
    if os.environ.get("RCWN_CUDA_LIBS_OK") != "1":
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join([prefix, existing])
        env["RCWN_CUDA_LIBS_OK"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_prefer_torch_cuda_libs()

import cv2
import torch
from tqdm import tqdm

from evaluate_hat_gray import _forward_gray_once, build_hat, forward_gray, load_hat_weights
from runtime import INTERP, _aug, _deaug, apply_postprocess, image_to_tensor, interp_tensor, tensor_to_image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "超分竞赛数据集"


def parse_tta_modes(raw: str, use_tta: bool) -> list[int] | None:
    if not raw:
        return None if not use_tta else list(range(8))
    modes = [int(part.strip()) for part in raw.replace(":", ",").split(",") if part.strip()]
    if not modes:
        raise ValueError("--tta-modes cannot be empty when provided.")
    if any(mode < 0 or mode > 7 for mode in modes):
        raise ValueError(f"--tta-modes values must be in [0, 7], got {raw}")
    if len(set(modes)) != len(modes):
        raise ValueError(f"--tta-modes values must be unique, got {raw}")
    return modes


def forward_gray_selected(
    model: torch.nn.Module,
    lr: torch.Tensor,
    scale: int,
    gray_mode: str,
    native_io: bool,
    tta: bool,
    tta_modes: list[int] | None,
) -> torch.Tensor:
    modes = parse_tta_modes(",".join(str(mode) for mode in tta_modes), tta) if tta_modes is not None else None
    if modes is None:
        return forward_gray(model, lr, scale, gray_mode, tta=tta, native_io=native_io)
    if modes == [0]:
        return _forward_gray_once(model, lr, scale, gray_mode, native_io=native_io)
    preds = []
    for mode in modes:
        pred = _forward_gray_once(model, _aug(lr, mode), scale, gray_mode, native_io=native_io)
        preds.append(_deaug(pred, mode))
    return torch.stack(preds).mean(dim=0)


def zip_submission(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(package_dir.parent))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DATA / "初赛测试集" / "input_320"))
    parser.add_argument("--team-name", default="fqy_hat_l_nativeio_blend002")
    parser.add_argument("--zip-path")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["l", "m"], default="l")
    parser.add_argument("--param-key", default="state_dict")
    parser.add_argument("--weights-b")
    parser.add_argument("--variant-b", choices=["l", "m"], default="l")
    parser.add_argument("--param-key-b", default="state_dict")
    parser.add_argument("--alpha-a", type=float, default=1.0)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--native-io", action="store_true")
    parser.add_argument("--native-io-b", action="store_true")
    parser.add_argument("--interp", choices=sorted(INTERP), default="lanczos")
    parser.add_argument("--blend-interp", type=float, default=0.0)
    parser.add_argument("--sharpen-amount", type=float, default=0.0)
    parser.add_argument("--sharpen-radius", type=float, default=1.0)
    parser.add_argument("--back-project-iters", type=int, default=0)
    parser.add_argument("--back-project-alpha", type=float, default=1.0)
    parser.add_argument("--back-project-down", choices=["nearest", "linear", "cubic", "area"], default="area")
    parser.add_argument("--back-project-up", choices=["nearest", "linear", "cubic", "area"], default="cubic")
    parser.add_argument("--back-project-down-sigma", type=float, default=0.0)
    parser.add_argument("--clip-mode", choices=["hard", "match-base", "none"], default="hard")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-b", action="store_true")
    parser.add_argument("--tta-modes", default="", help="Optional comma-separated TTA modes, e.g. 0,1,2,4,5,6,7")
    parser.add_argument("--tta-modes-b", default="", help="Optional comma-separated TTA modes for model B")
    parser.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--use-checkpoint-b", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.weights_b and not 0.0 <= args.alpha_a <= 1.0:
        raise ValueError(f"--alpha-a must be in [0, 1], got {args.alpha_a}")
    tta_modes = parse_tta_modes(args.tta_modes, args.tta)
    tta_modes_b = parse_tta_modes(args.tta_modes_b, args.tta_b)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_hat(args.variant, args.scale, args.use_checkpoint, native_io=args.native_io).to(device).eval()
    load_hat_weights(model, args.weights, args.param_key, native_io=args.native_io, gray_mode=args.gray_mode)
    model_b = None
    if args.weights_b:
        model_b = build_hat(args.variant_b, args.scale, args.use_checkpoint_b, native_io=args.native_io_b).to(device).eval()
        load_hat_weights(model_b, args.weights_b, args.param_key_b, native_io=args.native_io_b, gray_mode=args.gray_mode)

    package_dir = ROOT / "submission" / args.team_name
    prelim = package_dir / "preliminary"
    if package_dir.exists():
        import shutil

        shutil.rmtree(package_dir)
    prelim.mkdir(parents=True, exist_ok=True)

    names = sorted(path.name for path in Path(args.input_dir).glob("*.png"))
    print(
        f"device={device} images={len(names)} weights={args.weights} native_io={args.native_io} "
        f"weights_b={args.weights_b or ''} alpha_a={args.alpha_a} blend={args.blend_interp} "
        f"tta={args.tta} tta_modes={tta_modes or ''} tta_b={args.tta_b} tta_modes_b={tta_modes_b or ''}"
    )
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    amp_enabled = device.type == "cuda" and args.amp != "off"
    for name in tqdm(names):
        img = cv2.imread(str(Path(args.input_dir) / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(Path(args.input_dir) / name)
        lr = image_to_tensor(img, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            pred = forward_gray_selected(
                model,
                lr,
                args.scale,
                args.gray_mode,
                args.native_io,
                args.tta,
                tta_modes,
            )
            if model_b is not None:
                pred_b = forward_gray_selected(
                    model_b,
                    lr,
                    args.scale,
                    args.gray_mode,
                    args.native_io_b,
                    args.tta_b,
                    tta_modes_b,
                )
                pred = pred.float() * args.alpha_a + pred_b.float() * (1.0 - args.alpha_a)
        base = interp_tensor(lr, args.scale, args.interp)
        pred = apply_postprocess(
            pred.float(),
            base,
            lr=lr,
            blend_interp=args.blend_interp,
            sharpen_amount=args.sharpen_amount,
            sharpen_radius=args.sharpen_radius,
            back_project_iters=args.back_project_iters,
            back_project_alpha=args.back_project_alpha,
            back_project_down=args.back_project_down,
            back_project_up=args.back_project_up,
            back_project_down_sigma=args.back_project_down_sigma,
            clip_mode=args.clip_mode,
        )
        cv2.imwrite(str(prelim / name), tensor_to_image(pred))

    (package_dir / "README.md").write_text(
        "# RCWN first-stage x2 submission\n\n"
        "Generated by HAT-L pure model inference with legal LR-only postprocessing"
        + (" and legal image-space model ensemble" if args.weights_b else "")
        + ". "
        "No training HR image is copied, substituted, retrieved, or provided as inference input.\n",
        encoding="utf-8",
    )
    default_name = f"{args.team_name}.zip"
    zip_path = Path(args.zip_path) if args.zip_path else ROOT / "submission" / default_name
    zip_submission(package_dir, zip_path)
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
