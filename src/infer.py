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

import cv2
import torch
from tqdm import tqdm

from checkpoint import load_model_from_checkpoint
from models import MODEL_PRESETS
from runtime import (
    INTERP,
    apply_postprocess,
    forward_ensemble,
    image_to_tensor,
    interpolate_np,
    parse_coeffs,
    tensor_to_image,
)


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    models = []
    if args.weights:
        for weights in args.weights:
            model, cfg = load_model_from_checkpoint(weights, device, scale=args.scale, preset=args.preset)
            model.eval()
            models.append(model)
            args.scale = int(cfg.get("scale", args.scale))
    coeffs = parse_coeffs(args.ensemble_coeffs, len(models)) if models else None

    os.makedirs(args.output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(args.input_dir) if n.lower().endswith(".png"))
    out_size = (640, 512)
    print(
        f"device={device} images={len(names)} scale=x{args.scale} tta={args.tta} "
        f"weights={args.weights or 'interp'} blend={args.blend_interp} "
        f"sharpen={args.sharpen_amount}/{args.sharpen_radius} clip={args.clip_mode}"
    )

    for name in tqdm(names):
        img = cv2.imread(os.path.join(args.input_dir, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        base_np = interpolate_np(img, out_size, args.interp)
        if not models:
            out_np = base_np
        else:
            lr = image_to_tensor(img, device)
            pred = forward_ensemble(models, lr, args.amp, args.tta, coeffs)
            if pred.shape[-2:] != (512, 640):
                pred = torch.nn.functional.interpolate(pred, size=(512, 640), mode="bicubic", align_corners=False)
            base = image_to_tensor(base_np, device)
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
            out_np = tensor_to_image(pred)
        cv2.imwrite(os.path.join(args.output_dir, name), out_np)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", nargs="+")
    parser.add_argument("--ensemble-coeffs")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=2)
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
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())
