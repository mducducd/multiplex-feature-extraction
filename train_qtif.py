#!/usr/bin/env python3
"""CLI replacement for `4- Training.ipynb`.

This script wraps STAMP commands used in the notebook:
- crossval
- statistics
- heatmaps

Examples:
  python train_qtif.py crossval
  python train_qtif.py statistics --output-dir /path/to/exp
  python train_qtif.py full
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


def default_crossval_config(output_dir: Path) -> dict[str, Any]:
    return {
        "crossval": {
            "output_dir": str(output_dir),
            "clini_table": "/mnt/bulk-neptune/laura/multiplex/data/Vorversuche_2_WSI/clinicalData_MeDDrive_Paula_4.xlsx",
            "feature_dir": "/mnt/bulk-neptune/laura/multiplex/features/Immunpanel_WSI_Her2_256px",
            "slide_table": "/mnt/bulk-neptune/laura/multiplex/data/Vorversuche_2_WSI/Immunpanel_WSI_metadata/slide.csv",
            "task": "classification",
            "ground_truth_label": "TTF1",
            "patient_label": "Image",
            "filename_label": "FILENAME",
            "n_splits": 5,
        },
        "advanced_config": {
            "seed": 42,
            "max_epochs": 64,
            "patience": 8,
            "batch_size": 64,
            "bag_size": 512,
            "max_lr": 1e-4,
            "div_factor": 25.0,
            "model_name": "vit",
            "model_params": {
                "vit": {
                    "dim_model": 512,
                    "dim_feedforward": 512,
                    "n_heads": 8,
                    "n_layers": 2,
                    "dropout": 0.25,
                },
                "trans_mil": {
                    "dim_hidden": 512,
                },
                "mlp": {
                    "dim_hidden": 512,
                    "num_layers": 2,
                    "dropout": 0.25,
                },
            },
        },
    }


def default_statistics_config(output_dir: Path) -> dict[str, Any]:
    pred_csvs = [str(p) for p in sorted(output_dir.rglob("patient-preds.csv"))]
    return {
        "statistics": {
            "output_dir": str(output_dir),
            "slide_table": "/mnt/bulk-neptune/laura/multiplex/data/Vorversuche_2_WSI/Immunpanel_WSI_metadata/slide.csv",
            "task": "classification",
            "ground_truth_label": "TTF1",
            "true_class": "positiv",
            "pred_csvs": pred_csvs,
        }
    }


def default_heatmap_config() -> dict[str, Any]:
    return {
        "heatmaps": {
            "output_dir": "/mnt/bulk-sirius/nguyenmin/multiplex/temp/heatmap",
            "feature_dir": "/mnt/bulk-sirius/nguyenmin/multiplex/features/Her2_64px",
            "wsi_dir": "/mnt/bulk-neptune/laura/multiplex/data/IMPALUX_CA/split_channels_MSIs_unmixed",
            "checkpoint_path": "/mnt/bulk-sirius/nguyenmin/multiplex/exp/her2_64px/split-0/model.ckpt",
            "slide_paths": [
                "/mnt/bulk-neptune/laura/multiplex/data/IMPALUX_CA/split_channels_MSIs_unmixed/E-19-1584_[4724,57210]_component_data_Her2.tif"
            ],
        }
    }


def choose_stamp_binary(script_dir: Path) -> str:
    local_stamp = script_dir / ".venv" / "bin" / "stamp"
    if local_stamp.exists():
        return str(local_stamp)
    parent_stamp = script_dir.parent / ".venv" / "bin" / "stamp"
    if parent_stamp.exists():
        return str(parent_stamp)
    return "stamp"


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(data))
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False)


def run_stamp(stamp_bin: str, config_path: Path, action: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [stamp_bin, "--config", str(config_path), action]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def cmd_crossval(args: argparse.Namespace, script_dir: Path) -> int:
    output_dir = Path(args.output_dir).resolve()
    cfg = default_crossval_config(output_dir)

    if args.n_splits is not None:
        cfg["crossval"]["n_splits"] = args.n_splits
    if args.ground_truth_label is not None:
        cfg["crossval"]["ground_truth_label"] = args.ground_truth_label
    if args.feature_dir is not None:
        cfg["crossval"]["feature_dir"] = args.feature_dir
    if args.slide_table is not None:
        cfg["crossval"]["slide_table"] = args.slide_table
    if args.clini_table is not None:
        cfg["crossval"]["clini_table"] = args.clini_table
    if args.max_epochs is not None:
        cfg["advanced_config"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg["advanced_config"]["batch_size"] = args.batch_size
    if args.model_name is not None:
        cfg["advanced_config"]["model_name"] = args.model_name

    config_path = output_dir / "crossval_config.yaml"
    write_yaml(config_path, cfg)
    print(f"Saved crossval config to: {config_path}")

    stamp_bin = choose_stamp_binary(script_dir)
    run_stamp(stamp_bin, config_path, "crossval")
    return 0


def cmd_statistics(args: argparse.Namespace, script_dir: Path) -> int:
    output_dir = Path(args.output_dir).resolve()
    cfg = default_statistics_config(output_dir)

    if args.ground_truth_label is not None:
        cfg["statistics"]["ground_truth_label"] = args.ground_truth_label
    if args.true_class is not None:
        cfg["statistics"]["true_class"] = args.true_class
    if args.slide_table is not None:
        cfg["statistics"]["slide_table"] = args.slide_table
    if args.pred_csv:
        cfg["statistics"]["pred_csvs"] = args.pred_csv

    pred_csvs = cfg["statistics"]["pred_csvs"]
    if not pred_csvs:
        raise FileNotFoundError(f"No patient-preds.csv found under {output_dir}")

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        config_path = Path(tmp.name)

    print(f"Saved statistics config to: {config_path}")
    stamp_bin = choose_stamp_binary(script_dir)
    run_stamp(stamp_bin, config_path, "statistics", check=True)
    return 0


def cmd_heatmaps(args: argparse.Namespace, script_dir: Path) -> int:
    cfg = default_heatmap_config()

    if args.output_dir is not None:
        cfg["heatmaps"]["output_dir"] = args.output_dir
    if args.feature_dir is not None:
        cfg["heatmaps"]["feature_dir"] = args.feature_dir
    if args.wsi_dir is not None:
        cfg["heatmaps"]["wsi_dir"] = args.wsi_dir
    if args.checkpoint_path is not None:
        cfg["heatmaps"]["checkpoint_path"] = args.checkpoint_path
    if args.slide_path:
        cfg["heatmaps"]["slide_paths"] = args.slide_path

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        config_path = Path(tmp.name)

    print(f"Saved heatmaps config to: {config_path}")
    stamp_bin = choose_stamp_binary(script_dir)
    run_stamp(stamp_bin, config_path, "heatmaps", check=True)
    return 0


def cmd_full(args: argparse.Namespace, script_dir: Path) -> int:
    crossval_args = argparse.Namespace(
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        ground_truth_label=args.ground_truth_label,
        feature_dir=args.feature_dir,
        slide_table=args.slide_table,
        clini_table=args.clini_table,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
    cmd_crossval(crossval_args, script_dir)

    stat_args = argparse.Namespace(
        output_dir=args.output_dir,
        ground_truth_label=args.ground_truth_label,
        true_class=args.true_class,
        slide_table=args.slide_table,
        pred_csv=None,
    )
    cmd_statistics(stat_args, script_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_output = "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Laura/exp/her2_256px_qtif_TTF1"

    p = argparse.ArgumentParser(description="Run training pipeline from 4- Training.ipynb")
    sub = p.add_subparsers(dest="command", required=True)

    p_cross = sub.add_parser("crossval", help="Run STAMP cross-validation")
    p_cross.add_argument("--output-dir", default=default_output)
    p_cross.add_argument("--n-splits", type=int, default=None)
    p_cross.add_argument("--ground-truth-label", default=None)
    p_cross.add_argument("--feature-dir", default=None)
    p_cross.add_argument("--slide-table", default=None)
    p_cross.add_argument("--clini-table", default=None)
    p_cross.add_argument("--max-epochs", type=int, default=None)
    p_cross.add_argument("--batch-size", type=int, default=None)
    p_cross.add_argument("--model-name", choices=["vit", "mlp", "trans_mil"], default=None)

    p_stats = sub.add_parser("statistics", help="Run STAMP statistics")
    p_stats.add_argument("--output-dir", default=default_output)
    p_stats.add_argument("--ground-truth-label", default="TTF1")
    p_stats.add_argument("--true-class", default="positiv")
    p_stats.add_argument("--slide-table", default=None)
    p_stats.add_argument("--pred-csv", action="append", help="Optional explicit prediction CSV. Repeatable.")

    p_heat = sub.add_parser("heatmaps", help="Run STAMP heatmaps")
    p_heat.add_argument("--output-dir", default=None)
    p_heat.add_argument("--feature-dir", default=None)
    p_heat.add_argument("--wsi-dir", default=None)
    p_heat.add_argument("--checkpoint-path", default=None)
    p_heat.add_argument("--slide-path", action="append", help="Slide path for heatmap generation. Repeatable.")

    p_full = sub.add_parser("full", help="Run crossval then statistics")
    p_full.add_argument("--output-dir", default=default_output)
    p_full.add_argument("--n-splits", type=int, default=None)
    p_full.add_argument("--ground-truth-label", default="TTF1")
    p_full.add_argument("--feature-dir", default=None)
    p_full.add_argument("--slide-table", default=None)
    p_full.add_argument("--clini-table", default=None)
    p_full.add_argument("--max-epochs", type=int, default=None)
    p_full.add_argument("--batch-size", type=int, default=None)
    p_full.add_argument("--model-name", choices=["vit", "mlp", "trans_mil"], default=None)
    p_full.add_argument("--true-class", default="positiv")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.command == "crossval":
        return cmd_crossval(args, script_dir)
    if args.command == "statistics":
        return cmd_statistics(args, script_dir)
    if args.command == "heatmaps":
        return cmd_heatmaps(args, script_dir)
    if args.command == "full":
        return cmd_full(args, script_dir)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
