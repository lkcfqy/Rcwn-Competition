from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from evaluate_hat_gray import build_hat, load_hat_weights
from evaluate_mambairv2_gray import build_mambairv2, forward_gray as forward_mambair_gray, normalize_state
from make_hat_submission import (
    DATA,
    ROOT,
    forward_gray_selected,
    parse_tta_modes,
    zip_submission,
)
from runtime import INTERP, apply_postprocess, image_to_tensor, interp_tensor, tensor_to_image


@dataclass(frozen=True)
class RouteSpec:
    name: str
    weight_hat_a: float
    weight_hat_b: float
    weight_mamba: float
    interp: str | None
    blend_interp: float


def parse_route_spec(raw: str) -> RouteSpec:
    # name:weight_hat_a:weight_hat_b:weight_mamba:interp:blend_interp
    parts = raw.split(":")
    if len(parts) != 6:
        raise ValueError(f"Invalid --route-spec, expected name:wa:wb:wm:interp:blend, got {raw}")
    name, wa_raw, wb_raw, wm_raw, interp_raw, blend_raw = parts
    if not name:
        raise ValueError(f"Route name is empty in {raw}")
    wa = float(wa_raw)
    wb = float(wb_raw)
    wm = float(wm_raw)
    total = wa + wb + wm
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Route weights must sum to 1.0, got {total:.8f} for {name}")
    interp = None if interp_raw in {"", "none", "raw"} else interp_raw
    if interp is not None and interp not in INTERP:
        raise ValueError(f"Unsupported interp {interp} for route {name}")
    blend = float(blend_raw)
    if blend < 0.0:
        raise ValueError(f"Negative blend for route {name}: {blend}")
    return RouteSpec(name=name, weight_hat_a=wa, weight_hat_b=wb, weight_mamba=wm, interp=interp, blend_interp=blend)


def read_assignments(path: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name", "")
            route = row.get("selected_route", "")
            if not name or not route:
                raise ValueError(f"Assignment row must contain name and selected_route: {row}")
            assignments[name] = route
    return assignments


def default_route_specs() -> list[RouteSpec]:
    return [
        RouteSpec("hhat_b0005", 0.65, 0.35, 0.0, "cubic", 0.005),
        RouteSpec("hhat_b0015", 0.65, 0.35, 0.0, "cubic", 0.015),
        RouteSpec("hmamba_a09", 0.90, 0.0, 0.10, None, 0.0),
        RouteSpec("hmamba_a095", 0.95, 0.0, 0.05, None, 0.0),
        RouteSpec("hhmamba_603010", 0.60, 0.30, 0.10, None, 0.0),
        RouteSpec("hhmamba_553510", 0.55, 0.35, 0.10, None, 0.0),
        RouteSpec("hhmamba_603505", 0.60, 0.35, 0.05, None, 0.0),
        RouteSpec("hhmamba_653005", 0.65, 0.30, 0.05, None, 0.0),
    ]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DATA / "初赛测试集" / "input_320"))
    parser.add_argument("--assignments")
    parser.add_argument("--default-route")
    parser.add_argument("--team-name", default="fqy_hat_hat_mambair_routed")
    parser.add_argument("--zip-path")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--hat-a-weights", required=True)
    parser.add_argument("--hat-a-variant", choices=["l", "m"], default="l")
    parser.add_argument("--hat-a-param-key", default="state_dict")
    parser.add_argument("--hat-a-native-io", action="store_true")
    parser.add_argument("--hat-a-tta", action="store_true")
    parser.add_argument("--hat-a-tta-modes", default="")
    parser.add_argument("--hat-b-weights", required=True)
    parser.add_argument("--hat-b-variant", choices=["l", "m"], default="l")
    parser.add_argument("--hat-b-param-key", default="state_dict")
    parser.add_argument("--hat-b-native-io", action="store_true")
    parser.add_argument("--hat-b-tta", action="store_true")
    parser.add_argument("--hat-b-tta-modes", default="")
    parser.add_argument("--mamba-weights", required=True)
    parser.add_argument("--mamba-variant", choices=["light", "base", "large"], default="large")
    parser.add_argument("--mamba-param-key", default="params")
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--route-spec", action="append", default=[])
    parser.add_argument("--clip-mode", choices=["hard", "match-base", "none"], default="hard")
    parser.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    route_specs = {spec.name: spec for spec in default_route_specs()}
    for raw in args.route_spec:
        spec = parse_route_spec(raw)
        route_specs[spec.name] = spec

    input_dir = Path(args.input_dir)
    names = sorted(path.name for path in input_dir.glob("*.png"))
    if not names:
        raise ValueError(f"No PNG inputs in {input_dir}")
    if args.assignments:
        assignments = read_assignments(args.assignments)
        unknown_routes = sorted({route for route in assignments.values() if route not in route_specs})
        if unknown_routes:
            raise ValueError(f"Assignments reference unknown routes: {unknown_routes}")
        missing = sorted(name for name in names if name not in assignments)
        extra = sorted(name for name in assignments if name not in set(names))
        if missing or extra:
            raise ValueError(f"Assignment/input mismatch: missing={missing[:5]} extra={extra[:5]}")
    else:
        if not args.default_route:
            raise ValueError("Either --assignments or --default-route is required.")
        if args.default_route not in route_specs:
            raise ValueError(f"Unknown --default-route {args.default_route}")
        assignments = {name: args.default_route for name in names}

    tta_a_modes = parse_tta_modes(args.hat_a_tta_modes, args.hat_a_tta)
    tta_b_modes = parse_tta_modes(args.hat_b_tta_modes, args.hat_b_tta)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    hat_a = build_hat(args.hat_a_variant, args.scale, args.use_checkpoint, native_io=args.hat_a_native_io).to(device).eval()
    load_hat_weights(
        hat_a,
        args.hat_a_weights,
        args.hat_a_param_key,
        native_io=args.hat_a_native_io,
        gray_mode=args.gray_mode,
    )
    hat_b = build_hat(args.hat_b_variant, args.scale, args.use_checkpoint, native_io=args.hat_b_native_io).to(device).eval()
    load_hat_weights(
        hat_b,
        args.hat_b_weights,
        args.hat_b_param_key,
        native_io=args.hat_b_native_io,
        gray_mode=args.gray_mode,
    )
    mamba = build_mambairv2(args.mamba_variant, args.scale, args.use_checkpoint).to(device).eval()
    mamba_ckpt = torch.load(args.mamba_weights, map_location="cpu")
    mamba.load_state_dict(normalize_state(mamba_ckpt, args.mamba_param_key), strict=True)

    package_dir = ROOT / "submission" / args.team_name
    prelim = package_dir / "preliminary"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    prelim.mkdir(parents=True, exist_ok=True)

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    amp_enabled = device.type == "cuda" and args.amp != "off"
    used_routes: dict[str, int] = {}
    print(f"device={device} images={len(names)} assignments={args.assignments or ''} default_route={args.default_route or ''}")
    for name in tqdm(names):
        route_name = assignments[name]
        route = route_specs[route_name]
        used_routes[route_name] = used_routes.get(route_name, 0) + 1

        img = cv2.imread(str(input_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(input_dir / name)
        lr = image_to_tensor(img, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            pred_a = forward_gray_selected(
                hat_a,
                lr,
                args.scale,
                args.gray_mode,
                args.hat_a_native_io,
                args.hat_a_tta,
                tta_a_modes,
            ).float()
            pred_b = forward_gray_selected(
                hat_b,
                lr,
                args.scale,
                args.gray_mode,
                args.hat_b_native_io,
                args.hat_b_tta,
                tta_b_modes,
            ).float()
            pred = pred_a * route.weight_hat_a + pred_b * route.weight_hat_b
            if route.weight_mamba > 0.0:
                pred_m = forward_mambair_gray(mamba, lr, args.gray_mode).float()
                pred = pred + pred_m * route.weight_mamba

        base = interp_tensor(lr, args.scale, route.interp) if route.interp is not None else None
        pred = apply_postprocess(
            pred.float(),
            base,
            lr=lr,
            blend_interp=route.blend_interp,
            clip_mode=args.clip_mode,
        )
        cv2.imwrite(str(prelim / name), tensor_to_image(pred))

    (package_dir / "README.md").write_text(
        "# RCWN first-stage x2 submission\n\n"
        "Generated by legal pure model inference with LR-feature cluster routing between fixed "
        "HAT/HAT and HAT/HAT/MambaIRv2 image-space ensembles. No training HR image is copied, "
        "substituted, retrieved, or provided as inference input.\n",
        encoding="utf-8",
    )
    default_name = f"{args.team_name}.zip"
    zip_path = Path(args.zip_path) if args.zip_path else ROOT / "submission" / default_name
    zip_submission(package_dir, zip_path)
    print(f"route_counts={used_routes}")
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
