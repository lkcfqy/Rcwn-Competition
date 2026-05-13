from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import load_pair, make_split
from evaluate_hat_postprocess_sweep import safe_name
from evaluate_sst_gray import autocast_context, build_sst, forward_sst_gray, normalize_state


def read_names(args: Any) -> list[str]:
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    else:
        names = split.val
    return names[: args.limit] if args.limit else names


def cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{safe_name(Path(name).stem)}.pt"


@torch.no_grad()
def run(args: Any) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    names = read_names(args)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} val={len(names)} scale=x{args.scale} cache_dir={cache_dir}")

    model = build_sst(args.variant, args.model_scale).to(device).eval()
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(normalize_state(ckpt, args.param_key), strict=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.2f}M preset=sst_{args.variant}")
    print(f"loaded={args.weights} param_key={args.param_key}")

    written = 0
    skipped = 0
    for name in tqdm(names, desc="cache", leave=False):
        out_path = cache_path(cache_dir, name)
        if out_path.exists() and not args.refresh_cache:
            skipped += 1
            continue
        lr, hr = load_pair(args.data, name, args.scale, device)
        with autocast_context(device, args.amp):
            pred = forward_sst_gray(model, lr, args.gray_mode, tta=args.tta)
        if pred.shape[-2:] != hr.shape[-2:]:
            mode = "area" if args.resize_output == "area" else "bicubic"
            kwargs = {} if mode == "area" else {"align_corners": False}
            pred = F.interpolate(pred, size=hr.shape[-2:], mode=mode, **kwargs)
        torch.save(
            {
                "name": name,
                "scale": args.scale,
                "gray_mode": args.gray_mode,
                "model_scale": args.model_scale,
                "ext_pred": pred.detach().cpu(),
            },
            out_path,
        )
        written += 1
    print(f"cache_done written={written} skipped={skipped}")


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--model-scale", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["light", "light_plus", "base", "base_plus", "large", "large_plus", "xl_plus"], default="xl_plus")
    parser.add_argument("--param-key", default="params_ema")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--resize-output", choices=["area", "bicubic"], default="area")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
