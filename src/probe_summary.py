from __future__ import annotations

import csv
from pathlib import Path


FIELDNAMES = [
    "source",
    "recipe",
    "static_score",
    "model_family",
    "model_variant",
    "weights",
    "out_dir",
    "initial_val",
    "final_val",
    "best_val",
    "delta",
    "train_loss",
    "promoted_or_closed",
]


def probe_status(
    final_proxy: float | None,
    baseline_proxy: float | None,
    promote_threshold: float | None,
) -> str:
    if final_proxy is None or baseline_proxy is None:
        return ""
    if promote_threshold is not None:
        # Treat small thresholds as required improvement over baseline and
        # large thresholds as absolute proxy targets for backward compatibility.
        if promote_threshold < 1.0:
            if final_proxy - baseline_proxy >= promote_threshold:
                return "promoted"
        elif final_proxy >= promote_threshold:
            return "promoted"
    if final_proxy >= baseline_proxy:
        return "archived"
    return "closed"


def append_probe_summary(path: str | None, row: dict[str, object]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exists = out_path.is_file()
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
