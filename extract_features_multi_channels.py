#!/usr/bin/env python3
"""
Multi-channel feature extraction using KRONOS embeddings.

Processes slide files (*.qptiff, *.tiff, *.tif) and writes one H5 per patient:
  {h5_path}/{patient_stem}.h5

Each H5 contains:
  feats              (N, embed_dim)                              float32
  marker_embeddings  (N, num_markers, embed_dim)                 float32
  token_embeddings   (N, num_markers, tokens, tokens, embed_dim) float32
  coord_x            (N,)                                        int32
  coord_y            (N,)                                        int32

Usage:
  python extract_features_multi_channels.py --config multiplex_config.yaml
"""

import argparse
import os
import random
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import h5py
import tifffile

# Disable external decoding libraries
tifffile.imagecodecs = None
tifffile._imagecodecs = None

import torch
import yaml
from tifffile import TiffFile
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from kronos import create_model_from_pretrained


def _stem(path: Path) -> str:
    """Return patient stem, stripping .qptiff/.tiff/.tif suffixes."""
    name = path.name
    name_lower = name.lower()
    for ext in (".qptiff", ".tiff", ".tif"):
        if name_lower.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _discover_slide_files(root: Path) -> list[Path]:
    """
    Auto-detect slide files and deduplicate by patient stem.

    If multiple extensions exist for the same stem, keep the highest-priority
    file: .qptiff > .tiff > .tif.
    """
    ext_priority = {".qptiff": 0, ".tiff": 1, ".tif": 2}
    chosen: dict[str, tuple[int, Path]] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        suffix = path.suffix.lower()
        if suffix not in ext_priority:
            continue

        stem = _stem(path)
        prio = ext_priority[suffix]
        prev = chosen.get(stem)
        if prev is None or prio < prev[0]:
            chosen[stem] = (prio, path)

    return sorted(item[1] for item in chosen.values())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-channel KRONOS feature extraction")
    p.add_argument("--config", default="multiplex_config.yaml", help="Path to multiplex_config.yaml")
    p.add_argument(
        "--device",
        default=None,
        help="GPU index like 0/1 or full torch device string like cuda:0/cpu",
    )
    p.add_argument("--parallel", action="store_true", help="Opt-in: split slides across all visible GPUs")
    p.add_argument("--slide-index", type=int, default=0, help="Start slide index (default: 0)")
    p.add_argument("--num-slides", type=int, default=1, help="How many slides to process from slide-index (default: 1)")
    return p.parse_args()


def _normalize_device_arg(device_arg: str | None) -> str | None:
    """Map simple GPU index args like '0'/'1' to torch devices."""
    if device_arg is None:
        return None
    d = str(device_arg).strip()
    if d.isdigit():
        return f"cuda:{d}"
    return d


def _metadata_csv_candidates(cfg_path: str, raw_cfg: dict | None = None) -> list[Path]:
    """Return candidate marker metadata CSV paths, in lookup priority order."""
    cfg_dir = Path(cfg_path).resolve().parent
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    configured_path = (raw_cfg or {}).get("marker_metadata_csv")
    if configured_path:
        csv_path = Path(str(configured_path)).expanduser()
        if not csv_path.is_absolute():
            csv_path = (cfg_dir / csv_path).resolve()
        candidates.append(csv_path)

    candidates.extend(
        [
            cfg_dir / "marker_metadata.csv",
            script_dir / "marker_metadata.csv",
            Path.cwd() / "marker_metadata.csv",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_metadata_stats(cfg_path: str, raw_cfg: dict | None = None) -> dict[str, tuple[float, float]]:
    """Load marker mean/std from marker_metadata.csv. Keys are lowercase marker names."""
    for csv_path in _metadata_csv_candidates(cfg_path, raw_cfg):
        if not csv_path.exists():
            continue

        stats: dict[str, tuple[float, float]] = {}
        with open(csv_path, newline="") as f:
            reader = __import__("csv").DictReader(f)
            for row in reader:
                try:
                    name = row["marker_name"].strip().lower()
                    stats[name] = (float(row["marker_mean"]), float(row["marker_std"]))
                except (ValueError, KeyError, AttributeError):
                    continue

        if stats:
            print(f"Loaded marker stats for {len(stats)} markers from {csv_path}")
        else:
            print(f"Found marker metadata file but parsed 0 valid rows: {csv_path}")
        return stats

    looked_in = ", ".join(str(p) for p in _metadata_csv_candidates(cfg_path, raw_cfg))
    print(f"No marker_metadata.csv found. Looked in: {looked_in}")
    return {}


def _resolve_marker_stats(
    markers: list[dict], cfg_path: str, raw_cfg: dict | None = None
) -> tuple[list[float | None], list[float | None]]:
    """Fill missing mean/std from marker_metadata.csv (case-insensitive). Leave None if not found."""
    metadata = _load_metadata_stats(cfg_path, raw_cfg)
    means, stds = [], []
    for m in markers:
        mean, std = m.get("mean"), m.get("std")
        if mean is None or std is None:
            key = m["name"].strip().lower()
            if key in metadata:
                csv_mean, csv_std = metadata[key]
                mean = mean if mean is not None else csv_mean
                std = std if std is not None else csv_std
                print(f"  {m['name']}: loaded stats from marker_metadata.csv (mean={mean:.4f}, std={std:.4f})")
            else:
                print(f"  {m['name']}: no stats found — skipping normalization")
        means.append(mean)
        stds.append(std)
    return means, stds


def build_config(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    mode = raw["multi_channels"]
    all_markers = raw["markers"]

    markers_to_extract = mode.get("markers_to_extract")
    if markers_to_extract:
        markers_to_extract_lower = [m.lower() for m in markers_to_extract]
        markers = [m for m in all_markers if m["name"].lower() in markers_to_extract_lower]
        if not markers:
            raise ValueError(f"markers_to_extract {markers_to_extract} matched none of {[m['name'] for m in all_markers]}")
    else:
        markers = all_markers

    means, stds = _resolve_marker_stats(markers, cfg_path, raw)

    return {
        "multiplex_image_path": raw["multiplex_image_path"],
        "h5_path": mode["h5_path"],
        "device": raw.get("device", "cuda"),
        "checkpoint_path": raw["model"]["checkpoint_path"],
        "hf_auth_token": raw["model"].get("hf_auth_token"),
        "cache_dir": raw["model"].get("cache_dir", "./model_assets"),
        "model_type": raw["model"].get("model_type", "vits16"),
        "token_overlap": raw["model"].get("token_overlap", False),
        "marker_order": [m["name"] for m in markers],
        "marker_means": means,
        "marker_stds": stds,
        "patch_size": mode.get("patch_size", 32),
        "batch_size": mode.get("batch_size", 1),
        "num_workers": mode.get("num_workers", 0),  # 0 = main process; >0 risks OOM on large slides
    }


class MultiChannelDataset(IterableDataset):
    """
    Yields (patch, coord_x, coord_y, fname) for each multi-channel patch.

    Reads the full (C, H, W) qptiff and yields normalized (C, ps, ps) patches.
    Skips patients whose H5 already exists.
    """

    def __init__(self, config: dict, shuffle: bool = False, slide_start: int = None, slide_end: int = None):
        self.h5_path = Path(config["h5_path"])
        self.patch_size = config["patch_size"]
        self.marker_order = config["marker_order"]
        # Shape (C, 1, 1) for broadcast normalization
        self.mean = torch.tensor([m if m is not None else 0.0 for m in config["marker_means"]], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([s if s is not None else 1.0 for s in config["marker_stds"]], dtype=torch.float32)[:, None, None]
        self.shuffle = shuffle

        root = Path(config["multiplex_image_path"])
        all_slides = _discover_slide_files(root)

        self.file_paths = []
        for slide in all_slides:
            if not (self.h5_path / f"{_stem(slide)}.h5").exists():
                self.file_paths.append(slide)

        if slide_start is not None or slide_end is not None:
            self.file_paths = self.file_paths[slide_start:slide_end]

        print(f"Found {len(all_slides)} slide files → {len(self.file_paths)} to process")

    def __iter__(self):
        paths = self.file_paths.copy()
        if self.shuffle:
            random.shuffle(paths)

        for path in paths:
            patient_stem = _stem(path)

            with TiffFile(path) as tif:
                img = tif.series[0].asarray()

            if img.ndim != 3:
                tqdm.write(f"Skipping {path.name}: expected (C, H, W) but got {img.shape}")
                continue

            n_markers = len(self.marker_order)
            if img.shape[0] < n_markers:
                tqdm.write(f"Skipping {path.name}: has {img.shape[0]} channels but config expects {n_markers}")
                continue
            img = img[:n_markers]

            _, H, W = img.shape
            ps = self.patch_size

            for y in range(0, H - ps + 1, ps):
                for x in range(0, W - ps + 1, ps):
                    patch = torch.tensor(img[:, y : y + ps, x : x + ps], dtype=torch.float32)
                    patch = (patch - self.mean) / self.std
                    yield patch, x, y, patient_stem


def run(config: dict, slide_start: int = None, slide_end: int = None) -> None:
    assert config["patch_size"] % 16 == 0, "patch_size must be divisible by 16"

    out_dir = Path(config["h5_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = MultiChannelDataset(config, slide_start=slide_start, slide_end=slide_end)
    if not dataset.file_paths:
        print("Nothing to process — all H5 files already exist.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    model, _, embed_dim = create_model_from_pretrained(
        checkpoint_path=config["checkpoint_path"],
        cfg_path=None,
        hf_auth_token=config["hf_auth_token"],
        cache_dir=config["cache_dir"],
        cfg={"model_type": config["model_type"], "token_overlap": config["token_overlap"]},
    )
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        gpu_info = ", ".join(f"cuda:{i} {torch.cuda.get_device_name(i)}" for i in range(n_gpus))
    else:
        gpu_info = "CPU"
    print(f"Embedding dim: {embed_dim} | Device: {device} | GPUs available: [{gpu_info}]")
    model.to(device).eval()

    num_markers = len(config["marker_order"])
    tokens_per_side = config["patch_size"] // 16
    total = len(dataset.file_paths)
    h5_files: dict = {}
    finalized_count = 0
    active_fname = None
    patches_done = 0
    t_start = time.time()

    pbar = tqdm(total=total, desc="Progress", unit="file",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} files "
                            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    skip_fnames: set = set()

    def _finalize_one(fname: str) -> None:
        nonlocal finalized_count
        entry = h5_files.pop(fname, None)
        if entry is None:
            return
        entry["file"].flush()
        entry["file"].close()
        os.replace(entry["tmp_path"], entry["final_path"])
        finalized_count += 1

    try:
        for batch, coord_x, coord_y, fname in dataloader:
            with torch.no_grad():
                patch_emb, marker_emb, token_emb = model(batch.to(device))

            patch_np = patch_emb.cpu().numpy()
            marker_np = marker_emb.cpu().numpy()
            token_np = token_emb.cpu().numpy()
            cx_np = coord_x.numpy()
            cy_np = coord_y.numpy()

            for i, sample_fname in enumerate(fname):
                if sample_fname in skip_fnames:
                    continue

                if not torch.isfinite(patch_emb[i]).all():
                    tqdm.write(f"NaN — skipping {sample_fname} at ({cx_np[i]}, {cy_np[i]})")
                    continue

                if active_fname is not None and sample_fname != active_fname:
                    _finalize_one(active_fname)
                if sample_fname not in h5_files:
                    pbar.update(1)
                    active_fname = sample_fname
                    final_path = out_dir / f"{sample_fname}.h5"
                    tmp_path = out_dir / f".{sample_fname}.h5.partial"
                    if tmp_path.exists():
                        tmp_path.unlink()
                    try:
                        hf = h5py.File(tmp_path, "w")
                    except BlockingIOError:
                        tqdm.write(f"Skipping {sample_fname} — H5 locked by another process")
                        skip_fnames.add(sample_fname)
                        continue
                    hf.attrs["marker_names"] = config["marker_order"]
                    patch_ds = hf.create_dataset(
                        "feats",
                        shape=(0, embed_dim),
                        maxshape=(None, embed_dim),
                        dtype="f",
                    )
                    # Backward-compatible alias expected by some STAMP loading paths.
                    hf["patch_embeddings"] = patch_ds
                    h5_files[sample_fname] = {
                        "tmp_path": tmp_path,
                        "final_path": final_path,
                        "file": hf,
                        "patch_ds": patch_ds,
                        "marker_ds": hf.create_dataset(
                            "marker_embeddings",
                            shape=(0, num_markers, embed_dim),
                            maxshape=(None, num_markers, embed_dim),
                            dtype="f",
                        ),
                        "token_ds": hf.create_dataset(
                            "token_embeddings",
                            shape=(0, num_markers, tokens_per_side, tokens_per_side, embed_dim),
                            maxshape=(None, num_markers, tokens_per_side, tokens_per_side, embed_dim),
                            dtype="f",
                        ),
                        "coord_x_ds": hf.create_dataset("coord_x", shape=(0,), maxshape=(None,), dtype="i"),
                        "coord_y_ds": hf.create_dataset("coord_y", shape=(0,), maxshape=(None,), dtype="i"),
                    }

                entry = h5_files[sample_fname]
                n = entry["patch_ds"].shape[0]
                end = n + 1

                for key in ("patch_ds", "marker_ds", "token_ds", "coord_x_ds", "coord_y_ds"):
                    entry[key].resize(end, axis=0)

                entry["patch_ds"][n] = patch_np[i]
                entry["marker_ds"][n] = marker_np[i]
                entry["token_ds"][n] = token_np[i]
                entry["coord_x_ds"][n] = cx_np[i]
                entry["coord_y_ds"][n] = cy_np[i]

                patches_done += 1
            elapsed = time.time() - t_start
            patch_rate = patches_done / elapsed if elapsed > 0 else 0
            eta_total = str(timedelta(seconds=int(elapsed / pbar.n * total))) if pbar.n > 0 else "?"
            pbar.set_postfix({"patches/s": f"{patch_rate:.0f}", "ETA total": eta_total, "current": active_fname}, refresh=False)

        if active_fname is not None:
            _finalize_one(active_fname)

    finally:
        pbar.close()
        for entry in h5_files.values():
            entry["file"].close()

    elapsed_total = timedelta(seconds=int(time.time() - t_start))
    print(f"\nDone. {finalized_count} H5 files written to {out_dir} in {elapsed_total}")


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args.config)

    normalized_device = _normalize_device_arg(args.device)
    if normalized_device:
        config["device"] = normalized_device

    slide_start = args.slide_index
    slide_end = args.slide_index + args.num_slides

    # Parallel only when explicitly requested.
    n_gpus = torch.cuda.device_count()
    if args.parallel and n_gpus > 1:
        # Discover total slides to process
        probe = MultiChannelDataset(config)
        total_slides = len(probe.file_paths)
        if total_slides == 0:
            print("Nothing to process — all H5 files already exist.")
        else:
            chunk = (total_slides + n_gpus - 1) // n_gpus
            print(f"Launching {n_gpus} parallel workers across GPUs ({total_slides} slides):")
            procs = []
            for gpu_idx in range(n_gpus):
                start = gpu_idx * chunk
                end = min(start + chunk, total_slides)
                if start >= total_slides:
                    break
                print(f"  cuda:{gpu_idx} ({torch.cuda.get_device_name(gpu_idx)}) → slides {start}–{end - 1}")
                cmd = [
                    sys.executable, __file__,
                    "--config", args.config,
                    "--slide-index", str(start),
                    "--num-slides", str(end - start),
                    "--device", "cuda:0",
                ]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_idx)}
                procs.append(subprocess.Popen(cmd, env=env))
            for p in procs:
                p.wait()
    else:
        run(config, slide_start=slide_start, slide_end=slide_end)
