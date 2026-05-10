import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODEL_PRESETS


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "超分竞赛数据集"


def run_infer(input_dir: Path, output_dir: Path, scale: int, weights: list[str] | None, args) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "src" / "infer.py"),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--scale",
        str(scale),
        "--preset",
        args.preset,
        "--interp",
        args.interp,
        "--amp",
        args.amp,
    ]
    if weights:
        cmd.append("--weights")
        cmd.extend(weights)
    if args.ensemble_coeffs:
        cmd.extend(["--ensemble-coeffs", args.ensemble_coeffs])
    if args.tta:
        cmd.append("--tta")
    if args.blend_interp > 0:
        cmd.extend(["--blend-interp", str(args.blend_interp)])
    if args.sharpen_amount > 0:
        cmd.extend(["--sharpen-amount", str(args.sharpen_amount)])
        cmd.extend(["--sharpen-radius", str(args.sharpen_radius)])
    if args.back_project_iters > 0:
        cmd.extend(["--back-project-iters", str(args.back_project_iters)])
        cmd.extend(["--back-project-alpha", str(args.back_project_alpha)])
        cmd.extend(["--back-project-down", args.back_project_down])
        cmd.extend(["--back-project-up", args.back_project_up])
        cmd.extend(["--back-project-down-sigma", str(args.back_project_down_sigma)])
    cmd.extend(["--clip-mode", args.clip_mode])
    subprocess.run(cmd, cwd=ROOT, check=True)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def zip_dir(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(folder.parent))


def main(args):
    package_dir = ROOT / "submission" / args.team_name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    readme_src = ROOT / "submission" / "README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, package_dir / "README.md")

    preliminary = package_dir / "preliminary"
    run_infer(
        DATA / "初赛测试集" / "input_320",
        preliminary,
        2,
        args.weights_x2,
        args,
    )

    if args.phase == 2:
        if not args.weights_x4:
            raise SystemExit("--weights-x4 is required for a phase-2 package.")
        final_validation = package_dir / "final_validation"
        final_test = package_dir / "final_test"
        run_infer(
            DATA / "决赛测试集" / "test_finalRound" / "input_160",
            final_validation,
            4,
            args.weights_x4,
            args,
        )
        run_infer(
            DATA / "决赛测试集" / "无监督",
            final_test,
            4,
            args.weights_x4,
            args,
        )
        weights_dir = package_dir / "weights"
        weights_dir.mkdir()
        for index, weight_path in enumerate(args.weights_x4):
            dst_name = "model.pth" if index == 0 else f"model_{index}.pth"
            shutil.copy2(weight_path, weights_dir / dst_name)
        copy_tree(ROOT / "src", package_dir / "src")
        req = ROOT / "requirements.txt"
        if req.exists():
            shutil.copy2(req, package_dir / "requirements.txt")

    zip_path = ROOT / "submission" / f"{args.team_name}.zip"
    zip_dir(package_dir, zip_path)
    print(f"wrote {zip_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--team-name", default="fqy_rcan")
    parser.add_argument("--weights-x2", nargs="+")
    parser.add_argument("--weights-x4", nargs="+")
    parser.add_argument("--ensemble-coeffs")
    parser.add_argument("--preset", choices=["auto", *sorted(MODEL_PRESETS)], default="base")
    parser.add_argument("--interp", default="lanczos")
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
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
