from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def to_uint8(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    lo = float(np.percentile(x, 0.5))
    hi = float(np.percentile(x, 99.5))
    if hi <= lo:
        lo = float(x.min())
        hi = float(x.max())
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    x = (x - lo) / (hi - lo)
    return np.round(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def thermal_raw_to_uint16(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    # MultiPoint stores thermal_raw normalized by 16383 according to the dataset card.
    x = np.clip(x, 0.0, 1.0) * 16383.0
    return np.round(x).astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-key", choices=["thermal", "thermal_raw"], default="thermal_raw")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--prefix", default="mp")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5_file, "r", swmr=True) as h5_file:
        members = sorted(h5_file.keys())
        if args.start_index:
            members = members[args.start_index :]
        if args.stride > 1:
            members = members[:: args.stride]
        if args.limit > 0:
            members = members[: args.limit]

        print(
            f"hdf5={args.hdf5_file} total_members={len(h5_file.keys())} "
            f"selected={len(members)} dataset_key={args.dataset_key}"
        )

        written = 0
        for index, member in enumerate(members):
            sample = h5_file[member]
            if args.dataset_key not in sample:
                continue
            arr = np.asarray(sample[args.dataset_key][...])
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(f"Unexpected shape for {member}/{args.dataset_key}: {arr.shape}")

            if args.dataset_key == "thermal_raw":
                image = Image.fromarray(thermal_raw_to_uint16(arr), mode="I;16")
            else:
                image = Image.fromarray(to_uint8(arr), mode="L")

            out_name = f"{args.prefix}_{index:07d}_{member}.png"
            image.save(out_dir / out_name)
            written += 1
            if written % 500 == 0:
                print(f"written={written}/{len(members)}")

    print(f"done written={written}")


if __name__ == "__main__":
    main()
