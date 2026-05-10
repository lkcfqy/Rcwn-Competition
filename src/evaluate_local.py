from __future__ import annotations

import argparse
import os
import sys


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

import torch
from tqdm import tqdm

from checkpoint import load_model_from_checkpoint
from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy
from models import MODEL_PRESETS
from runtime import INTERP, apply_postprocess, forward_ensemble, interp_tensor, parse_coeffs


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    names = split.val[: args.limit] if args.limit else split.val
    print(f"device={device} scale=x{args.scale} images={len(names)}")

    models = []
    if args.weights:
        for weights in args.weights:
            model, _ = load_model_from_checkpoint(weights, device, scale=args.scale, preset=args.preset)
            model.eval()
            models.append(model)
    coeffs = parse_coeffs(args.ensemble_coeffs, len(models)) if models else None

    lpips_fn = None
    if args.lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    edge_metric = EdgeMetric().to(device)
    sums = {}

    for name in tqdm(names):
        lr, hr = load_pair(args.data, name, args.scale, device)
        base = interp_tensor(lr, args.scale, args.interp)
        if not models:
            pred = interp_tensor(lr, args.scale, args.interp)
        else:
            pred = forward_ensemble(models, lr, args.amp, args.tta, coeffs)
        pred = apply_postprocess(
            pred,
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
        metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    out = {key: value / len(names) for key, value in sums.items()}
    out["proxy"] = metric_proxy(out)
    meta = {
        "tta": args.tta,
        "interp": args.interp,
        "blend": args.blend_interp,
        "sharpen": args.sharpen_amount,
        "radius": args.sharpen_radius,
        "bp_iters": args.back_project_iters,
        "bp_alpha": args.back_project_alpha,
        "bp_down": args.back_project_down,
        "bp_up": args.back_project_up,
        "bp_down_sigma": args.back_project_down_sigma,
        "clip": args.clip_mode,
        "weights": ",".join(args.weights or []),
        "coeffs": args.ensemble_coeffs or "",
    }
    print(" ".join(f"{key}={value:.6f}" for key, value in out.items()))
    print(" ".join(f"{key}={value}" for key, value in meta.items()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=2)
    parser.add_argument("--weights", nargs="+")
    parser.add_argument("--ensemble-coeffs")
    parser.add_argument("--preset", choices=["auto", *sorted(MODEL_PRESETS)], default="base")
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
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
