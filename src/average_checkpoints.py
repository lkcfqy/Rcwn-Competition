from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def get_state_dict(ckpt: Any, key: str) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and key in ckpt:
        state = ckpt[key]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise TypeError("Checkpoint must be a dict.")
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint key {key!r} is not a state dict.")
    return state


def parse_weights(raw: str, n: int) -> list[float]:
    if not raw:
        return [1.0 / n] * n
    weights = [float(x) for x in raw.split(",") if x.strip()]
    if len(weights) != n:
        raise ValueError(f"Expected {n} averaging weights, got {len(weights)}.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Averaging weights must sum to a positive value.")
    return [w / total for w in weights]


def average_checkpoints(paths: list[Path], weights: list[float], key: str) -> dict[str, Any]:
    base = torch.load(paths[0], map_location="cpu", mmap=True)
    base_state = get_state_dict(base, key)
    out_state: dict[str, torch.Tensor] = {}

    checkpoints = [base]
    checkpoints.extend(torch.load(path, map_location="cpu", mmap=True) for path in paths[1:])
    states = [get_state_dict(ckpt, key) for ckpt in checkpoints]
    base_keys = set(base_state)
    for idx, state in enumerate(states[1:], start=1):
        if set(state) != base_keys:
            missing = sorted(base_keys - set(state))[:5]
            extra = sorted(set(state) - base_keys)[:5]
            raise ValueError(f"State keys differ for {paths[idx]} missing={missing} extra={extra}")

    for name, tensor in base_state.items():
        if not torch.is_tensor(tensor):
            out_state[name] = tensor
            continue
        if not tensor.is_floating_point():
            out_state[name] = tensor.clone()
            continue
        acc = tensor.float() * weights[0]
        for state, weight in zip(states[1:], weights[1:], strict=True):
            other = state[name]
            if other.shape != tensor.shape:
                raise ValueError(f"Shape mismatch for {name}: {tensor.shape} vs {other.shape}")
            acc.add_(other.float(), alpha=weight)
        out_state[name] = acc.to(dtype=tensor.dtype)

    if isinstance(base, dict) and key in base:
        out = dict(base)
        out[key] = out_state
        out["metrics"] = {
            "averaged_from": [str(path) for path in paths],
            "average_weights": weights,
        }
        return out
    return out_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--key", default="state_dict")
    parser.add_argument("--weights", default="")
    parser.add_argument("checkpoints", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.checkpoints]
    weights = parse_weights(args.weights, len(paths))
    out = average_checkpoints(paths, weights, args.key)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved={out_path}")
    print("weights=" + ",".join(f"{w:.6f}" for w in weights))


if __name__ == "__main__":
    main()
