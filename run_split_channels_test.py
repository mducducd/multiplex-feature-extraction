#!/usr/bin/env python3
"""Run a small split-channel extraction test on a subset of slides.

This script:
1. Loads a base multiplex config YAML.
2. Selects N `*.qptiff` files from `multiplex_image_path`.
3. Creates a temporary test workspace with symlinks to those slides.
4. Writes a derived config that points to test input/output paths.
5. Executes `extract_features_split_channels.py --config <derived_config>`.

Example:
  python run_split_channels_test.py \
    --base-config multiplex_config.yaml \
    --num-slides 3 \
    --markers HER2 DAPI \
    --device cpu
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run split-channel extraction on a small slide subset")
    p.add_argument(
        "--base-config",
        default="multiplex_config.yaml",
        help="Path to base multiplex config YAML",
    )
    p.add_argument("--num-slides", type=int, default=2, help="Number of slides to test")
    p.add_argument(
        "--slide-pattern",
        default=None,
        help="Glob pattern for slide discovery. Defaults to auto-discovering *.qptiff, *.tiff, and *.tif.",
    )
    p.add_argument(
        "--markers",
        nargs="*",
        default=None,
        help="Optional marker names for split_channels.markers_to_extract",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Optional override for split_channels.patch_size",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for split_channels.batch_size",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for split_channels.num_workers",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Optional override for top-level device (e.g. cpu, cuda)",
    )
    p.add_argument(
        "--work-dir",
        default="temp/split_channels_test",
        help="Directory to place symlinked inputs, derived config, and outputs",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for deterministic slide selection order",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous work-dir contents before running",
    )
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> int:
    args = parse_args()

    base_config_path = Path(args.base_config).resolve()
    if not base_config_path.exists():
        print(f"ERROR: base config not found: {base_config_path}", file=sys.stderr)
        return 1

    cfg = load_yaml(base_config_path)

    src_root = Path(cfg["multiplex_image_path"])
    if not src_root.exists():
        print(f"ERROR: multiplex_image_path does not exist: {src_root}", file=sys.stderr)
        return 1

    if args.slide_pattern:
        all_slides = sorted(src_root.rglob(args.slide_pattern))
    else:
        all_slides = sorted({
            *src_root.rglob("*.qptiff"),
            *src_root.rglob("*.tiff"),
            *src_root.rglob("*.tif"),
        })
    if not all_slides:
        print(
            f"ERROR: no slides found under {src_root} (tried *.qptiff, *.tiff, *.tif)",
            file=sys.stderr,
        )
        return 1

    if args.num_slides < 1:
        print("ERROR: --num-slides must be >= 1", file=sys.stderr)
        return 1

    # Deterministic pseudo-random subset by modular stepping.
    # Keeps behavior stable without importing random for such a small utility.
    step = max(1, (args.seed % len(all_slides)) + 1)
    selected = []
    idx = args.seed % len(all_slides)
    seen = set()
    while len(selected) < min(args.num_slides, len(all_slides)):
        if idx not in seen:
            selected.append(all_slides[idx])
            seen.add(idx)
        idx = (idx + step) % len(all_slides)

    work_dir = Path(args.work_dir).resolve()
    input_dir = work_dir / "input"
    output_dir = work_dir / "h5"
    derived_config = work_dir / "multiplex_config.test.yaml"

    if args.overwrite and work_dir.exists():
        shutil.rmtree(work_dir)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for slide in selected:
        link_path = input_dir / slide.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(slide)

    cfg["multiplex_image_path"] = str(input_dir)
    cfg["split_channels"]["h5_path"] = str(output_dir)

    if args.markers is not None and len(args.markers) > 0:
        cfg["split_channels"]["markers_to_extract"] = args.markers
    if args.patch_size is not None:
        cfg["split_channels"]["patch_size"] = args.patch_size
    if args.batch_size is not None:
        cfg["split_channels"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["split_channels"]["num_workers"] = args.num_workers
    if args.device is not None:
        device = args.device.strip().lower()
        if device == "gpu":
            device = "cuda"
        cfg["device"] = device

    save_yaml(derived_config, cfg)

    script_path = base_config_path.parent / "extract_features_split_channels.py"
    if not script_path.exists():
        print(f"ERROR: cannot find extractor script next to config: {script_path}", file=sys.stderr)
        return 1

    print("Prepared split-channel test run")
    print(f"  base config:     {base_config_path}")
    print(f"  extractor:       {script_path}")
    print(f"  selected slides: {len(selected)}")
    for s in selected:
        print(f"    - {s}")
    print(f"  test input dir:  {input_dir}")
    print(f"  output dir:      {output_dir}")
    print(f"  derived config:  {derived_config}")

    script_dir = base_config_path.parent
    venv_python = script_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = script_dir.parent / ".venv" / "bin" / "python"
    python_bin = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [python_bin, str(script_path), "--config", str(derived_config)]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd)

    if completed.returncode != 0:
        print(f"Extractor failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    produced = sorted(output_dir.glob("*.h5"))
    print(f"Success. Produced {len(produced)} H5 files:")
    for h5 in produced:
        print(f"  - {h5}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
