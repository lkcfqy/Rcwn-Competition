from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x != ""]


def parse_str_list(raw: str) -> list[str]:
    return [x for x in raw.split(",") if x != ""]


def parse_metrics(output: str) -> dict[str, float]:
    metrics = {}
    for line in output.splitlines():
        if "psnr=" not in line or "proxy=" not in line:
            continue
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            try:
                metrics[key] = float(value)
            except ValueError:
                pass
    if not metrics:
        raise RuntimeError(f"Could not parse metrics from output:\n{output}")
    return metrics


def run_eval(args, interp: str, blend: float, sharpen: float, tta: bool, clip_mode: str) -> dict[str, str | float | bool]:
    cmd = [
        sys.executable,
        str(ROOT / "src" / "evaluate_local.py"),
        "--scale",
        "2",
        "--weights",
        *args.weights,
        "--interp",
        interp,
        "--blend-interp",
        str(blend),
        "--sharpen-amount",
        str(sharpen),
        "--sharpen-radius",
        str(args.sharpen_radius),
        "--clip-mode",
        clip_mode,
        "--val-count",
        str(args.val_count),
        "--lpips-net",
        args.lpips_net,
        "--amp",
        args.amp,
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.ensemble_coeffs:
        cmd.extend(["--ensemble-coeffs", args.ensemble_coeffs])
    if tta:
        cmd.append("--tta")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    metrics = parse_metrics(proc.stdout)
    row: dict[str, str | float | bool] = {
        "weights": "|".join(args.weights),
        "coeffs": args.ensemble_coeffs or "",
        "tta": tta,
        "interp": interp,
        "blend": blend,
        "sharpen": sharpen,
        "sharpen_radius": args.sharpen_radius,
        "clip_mode": clip_mode,
    }
    row.update(metrics)
    return row


def main(args):
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    interps = parse_str_list(args.interps)
    blends = parse_float_list(args.blends)
    sharpens = parse_float_list(args.sharpens)
    tta_values = [x == "on" for x in parse_str_list(args.tta_values)]
    clip_modes = parse_str_list(args.clip_modes)

    rows = []
    runs = 0
    for tta in tta_values:
        for interp in interps:
            for blend in blends:
                for sharpen in sharpens:
                    for clip_mode in clip_modes:
                        runs += 1
                        if args.max_runs and runs > args.max_runs:
                            break
                        print(
                            f"[{runs}] tta={tta} interp={interp} blend={blend} "
                            f"sharpen={sharpen} clip={clip_mode}",
                            flush=True,
                        )
                        row = run_eval(args, interp, blend, sharpen, tta, clip_mode)
                        rows.append(row)
                        print(
                            " -> "
                            + " ".join(
                                f"{k}={row[k]:.6f}" for k in ["psnr", "ssim", "edge", "lpips", "proxy"] if k in row
                            ),
                            flush=True,
                        )

    rows = sorted(rows, key=lambda r: float(r.get("proxy", -1)), reverse=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_path}")
        print("best")
        best = rows[0]
        print(" ".join(f"{k}={best[k]}" for k in fieldnames))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--ensemble-coeffs")
    parser.add_argument("--out", default="experiments/sweep_x2.csv")
    parser.add_argument("--interps", default="lanczos,cubic")
    parser.add_argument("--blends", default="0,0.03,0.05,0.08,0.10,0.15")
    parser.add_argument("--sharpens", default="0,0.03,0.05,0.08,0.10")
    parser.add_argument("--sharpen-radius", type=float, default=1.0)
    parser.add_argument("--clip-modes", default="hard")
    parser.add_argument("--tta-values", default="off,on")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
