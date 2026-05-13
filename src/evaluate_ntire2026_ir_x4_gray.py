from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any


def _prefer_torch_cuda_libs() -> None:
    base = "/usr/local/lib/python3.12/dist-packages"
    libs = [
        os.path.join(base, "torch", "lib"),
        os.path.join(base, "nvidia", "cublas", "lib"),
        os.path.join(base, "nvidia", "cuda_runtime", "lib"),
        os.path.join(base, "nvidia", "cudnn", "lib"),
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
import numpy as np
import torch
import torch.nn.functional as F

from dataset import load_pair, make_split
from metrics import EdgeMetric, measure_batch, metric_proxy


def _install_import_aliases(team_module: str) -> None:
    """Some NTIRE team folders were renamed after submission; keep imports local."""
    if "team02_SUATSR" in team_module:
        target = _load_module_from_file(
            "models.team02_SUATSR.model",
            Path("external_models/NTIRE2026_infraredSR/models/team02_SUATSR/model.py"),
        )
        package = types.ModuleType("models.team02_WIRSR_TEFA_Net")
        package.model = target
        sys.modules.setdefault("models.team02_WIRSR_TEFA_Net", package)
        sys.modules.setdefault("models.team02_WIRSR_TEFA_Net.model", target)
    if "team08_Earth4D" in team_module:
        target = _load_module_from_file(
            "models.team08_Earth4D.model",
            Path("external_models/NTIRE2026_infraredSR/models/team08_Earth4D/model.py"),
        )
        package = types.ModuleType("models.team08_IRSR")
        package.model = target
        sys.modules.setdefault("models.team08_IRSR", package)
        sys.modules.setdefault("models.team08_IRSR.model", target)


def _load_module_from_file(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_names(args: Any) -> list[str]:
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    split = make_split(args.data, args.scale, val_count=args.val_count, seed=args.seed)
    return split.val[: args.limit] if args.limit else split.val


def _prepare_input_dir(args: Any, names: list[str]) -> Path:
    input_dir = Path(args.work_dir) / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    lr_dir = Path(args.data) / f"input_{640 // args.scale}"
    for name in names:
        shutil.copy2(lr_dir / name, input_dir / name)
    return input_dir


def _read_prediction(path: Path, device: torch.device, gray_mode: str) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    image = image.astype(np.float32) / (65535.0 if image.dtype == np.uint16 else 255.0)
    if image.ndim == 2:
        gray = image
    elif gray_mode == "avg":
        gray = image[:, :, :3].mean(axis=2)
    elif gray_mode == "y":
        bgr = image[:, :, :3]
        gray = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
    elif gray_mode == "r":
        gray = image[:, :, 2]
    elif gray_mode == "g":
        gray = image[:, :, 1]
    elif gray_mode == "b":
        gray = image[:, :, 0]
    else:
        raise ValueError(f"Unsupported gray mode: {gray_mode}")
    return torch.from_numpy(np.ascontiguousarray(gray)).view(1, 1, gray.shape[0], gray.shape[1]).to(device)


def _downsample_to_target(pred: torch.Tensor, target_size: tuple[int, int], mode: str) -> torch.Tensor:
    if pred.shape[-2:] == target_size:
        return pred
    if mode == "area":
        return F.interpolate(pred, size=target_size, mode="area")
    return F.interpolate(pred, size=target_size, mode="bicubic", align_corners=False)


@torch.no_grad()
def validate(args: Any) -> None:
    ntire_root = Path(args.ntire_root).resolve()
    if str(ntire_root) not in sys.path:
        sys.path.insert(0, str(ntire_root))
    for module_name in list(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]
    models_package = types.ModuleType("models")
    models_package.__path__ = [str(ntire_root / "models")]
    sys.modules["models"] = models_package

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    names = _load_names(args)
    print(f"device={device} val={len(names)} module={args.team_module}")
    print(f"weights={args.weights}")

    input_dir = _prepare_input_dir(args, names)
    output_dir = Path(args.work_dir) / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _install_import_aliases(args.team_module)
    module = importlib.import_module(args.team_module)
    if not hasattr(module, "main"):
        raise AttributeError(f"{args.team_module} has no main(model_dir, input_path, output_path, device)")
    module.main(args.weights, str(input_dir), str(output_dir), device=device)

    edge_metric = EdgeMetric().to(device)
    lpips_fn = None
    if args.val_lpips_net != "none":
        import lpips

        lpips_fn = lpips.LPIPS(net=args.val_lpips_net).to(device).eval()

    rows: list[dict[str, float | str]] = []
    sums: dict[str, float] = {}
    for name in names:
        pred_path = output_dir / name
        if not pred_path.exists():
            stem_matches = sorted(output_dir.glob(Path(name).stem + ".*"))
            if not stem_matches:
                raise FileNotFoundError(f"Missing prediction for {name} in {output_dir}")
            pred_path = stem_matches[0]
        pred = _read_prediction(pred_path, device, args.gray_mode)
        _, hr = load_pair(args.data, name, args.scale, device)
        pred = _downsample_to_target(pred.float(), hr.shape[-2:], args.downsample)
        metrics = measure_batch(pred, hr, edge_metric, lpips_fn)
        row = {"name": name, **metrics, "proxy": metric_proxy(metrics)}
        rows.append(row)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value

    means = {key: value / len(names) for key, value in sums.items()}
    means["proxy"] = metric_proxy(means)
    print("val_result " + " ".join(f"{key}={value:.5f}" for key, value in means.items()))

    if args.metrics_csv:
        Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_csv, "w", newline="", encoding="utf-8") as handle:
            fieldnames = ["name", "psnr", "ssim", "edge", "lpips", "proxy"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        print(f"metrics_csv={args.metrics_csv}")


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/lkc/lkcproject/rcwn/超分竞赛数据集/训练数据集")
    parser.add_argument("--scale", type=int, choices=[2], default=2)
    parser.add_argument("--ntire-root", default="external_models/NTIRE2026_infraredSR")
    parser.add_argument("--team-module", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--gray-mode", choices=["avg", "y", "r", "g", "b"], default="avg")
    parser.add_argument("--downsample", choices=["area", "bicubic"], default="area")
    parser.add_argument("--val-count", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names-file", default="")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-lpips-net", choices=["none", "alex", "vgg"], default="alex")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
