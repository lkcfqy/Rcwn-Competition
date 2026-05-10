from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path


API_ROOT = "https://datasets-server.huggingface.co/rows"


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    retries: int = 5,
) -> dict:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{API_ROOT}?{params}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rcwn-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(8.0, 1.5 * attempt))
    assert last_error is not None
    raise last_error


def safe_name(path_text: str, row_idx: int) -> str:
    raw = Path(path_text)
    stem = raw.stem.replace("/", "_").replace("\\", "_")
    suffix = raw.suffix or ".jpg"
    return f"{row_idx:06d}_{stem}{suffix}"


def download_file(url: str, dst: Path, retries: int = 5) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rcwn-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dst)
            return
        except Exception as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                break
            time.sleep(min(8.0, 1.5 * attempt))
    assert last_error is not None
    raise last_error


def find_image_entry(values: dict) -> tuple[str, dict] | None:
    preferred_keys = ("image", "img", "jpg", "jpeg", "png")
    for key in preferred_keys:
        candidate = values.get(key)
        if isinstance(candidate, dict) and candidate.get("src"):
            return key, candidate
    for key, candidate in values.items():
        if isinstance(candidate, dict) and candidate.get("src"):
            return str(key), candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-width", type=int, default=0)
    parser.add_argument("--min-height", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = fetch_rows(args.dataset, args.config, args.split, offset=0, length=1)
    total_rows = int(probe["num_rows_total"])
    print(f"dataset={args.dataset} split={args.split} total_rows={total_rows}")

    rng = random.Random(args.seed)
    max_start = max(0, total_rows - args.block_size)
    starts = list(range(0, max_start + 1, args.block_size))
    rng.shuffle(starts)
    if not starts:
        starts = [0]

    selected: list[dict[str, int | str]] = []
    seen_paths: set[str] = set()

    for start in starts:
        payload = fetch_rows(args.dataset, args.config, args.split, offset=start, length=args.block_size)
        for row in payload.get("rows", []):
            row_idx = int(row["row_idx"])
            values = row["row"]
            image_entry = find_image_entry(values)
            if image_entry is None:
                continue
            image_key, image = image_entry
            path_text = str(values.get("path") or values.get("fname") or f"{row_idx}_{image_key}.jpg")
            src = image.get("src")
            width = int(values.get("w") or values.get("width") or image.get("width") or 0)
            height = int(values.get("h") or values.get("height") or image.get("height") or 0)
            if not src:
                continue
            if width < args.min_width or height < args.min_height:
                continue
            if path_text in seen_paths:
                continue
            seen_paths.add(path_text)
            selected.append(
                {
                    "row_idx": row_idx,
                    "path": path_text,
                    "src": str(src),
                    "width": width,
                    "height": height,
                }
            )
            if len(selected) >= args.count:
                break
        print(f"selected={len(selected)} start={start}")
        if len(selected) >= args.count:
            break

    if not selected:
        raise RuntimeError("No rows selected.")

    selected = selected[: args.count]
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_idx", "path", "width", "height", "src", "local_path"])
        writer.writeheader()
        for idx, item in enumerate(selected, start=1):
            dst = out_dir / safe_name(str(item["path"]), int(item["row_idx"]))
            if not dst.exists():
                download_file(str(item["src"]), dst)
            writer.writerow(
                {
                    "row_idx": item["row_idx"],
                    "path": item["path"],
                    "width": item["width"],
                    "height": item["height"],
                    "src": item["src"],
                    "local_path": str(dst.resolve()),
                }
            )
            if idx % 100 == 0 or idx == len(selected):
                print(f"downloaded={idx}/{len(selected)}")

    print(f"written_manifest={manifest_path} images={len(selected)}")


if __name__ == "__main__":
    main()
