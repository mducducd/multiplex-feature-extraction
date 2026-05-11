#!/usr/bin/env python3
"""
Split-channel feature extraction using KRONOS embeddings.

Processes *.qptiff files and writes one H5 per (patient, marker):
  {h5_path}/{patient_stem}_{marker_name}.h5

Each H5 contains:
  feats  (N, embed_dim)  float32
  coord_x           (N,)            int32
  coord_y           (N,)            int32
  attrs: marker_names = marker_name

Usage:
  python extract_features_split_channels.py --config multiplex_config.yaml
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
import torch
import yaml
from tifffile import TiffFile
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from torchvision.transforms import Normalize

from kronos import create_model_from_pretrained


def _stem(path: Path) -> str:
    """Return patient stem, stripping .unmixed.qptiff, .tiff, or .tif."""
    name = path.name
    for ext in (".qptiff", ".tiff", ".tif"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split-channel KRONOS feature extraction")
    p.add_argument("--config", default="multiplex_config.yaml", help="Path to multiplex_config.yaml")
    p.add_argument("--device", default=None, help="Override config device (e.g. cuda:0)")
    p.add_argument("--markers", nargs="*", default=None, help="Override markers_to_extract")
    return p.parse_args()


def build_config(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    mode = raw["split_channels"]
    all_markers = raw["markers"]

    markers_to_extract = mode.get("markers_to_extract")
    if markers_to_extract:
        markers = [m for m in all_markers if m["name"] in markers_to_extract]
        if not markers:
            raise ValueError(f"markers_to_extract {markers_to_extract} matched none of {[m['name'] for m in all_markers]}")
    else:
        markers = all_markers

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
        "marker_means": [m["mean"] for m in markers],
        "marker_stds": [m["std"] for m in markers],
        "patch_size": mode.get("patch_size", 64),
        "batch_size": mode.get("batch_size", 1),
        "num_workers": mode.get("num_workers", 0),
    }


class SplitChannelDataset(IterableDataset):
    """
    Yields (patch, coord_x, coord_y, fname) for each single-channel patch.

    Iterates over all (qptiff, marker) pairs whose H5 does not yet exist,
    reading one channel at a time from the qptiff.
    """

    def __init__(self, config: dict, shuffle: bool = False):
        self.h5_path = Path(config["h5_path"])
        self.patch_size = config["patch_size"]
        self.marker_order = config["marker_order"]
        self.mean = torch.tensor([m if m is not None else 0.0 for m in config["marker_means"]], dtype=torch.float32)
        self.std = torch.tensor([s if s is not None else 1.0 for s in config["marker_stds"]], dtype=torch.float32)
        self.shuffle = shuffle

        root = Path(config["multiplex_image_path"])
        all_slides = sorted(
            f for f in {
                *root.rglob("*.qptiff"),
                *root.rglob("*.tiff"),
                *root.rglob("*.tif"),
            }
            if not any(part.startswith(".") for part in f.parts)
        )

        self.file_paths = []
        for slide in all_slides:
            patient_stem = _stem(slide)
            for ch_idx, marker_name in enumerate(self.marker_order):
                if not (self.h5_path / f"{patient_stem}_{marker_name}.h5").exists():
                    self.file_paths.append((slide, marker_name, ch_idx))

        print(f"Found {len(all_slides)} slide files → {len(self.file_paths)} (file, marker) pairs to process")

    def __iter__(self):
        items = self.file_paths.copy()
        if self.shuffle:
            random.shuffle(items)

        for path, marker_name, ch_idx in items:
            fname = f"{_stem(path)}_{marker_name}"

            with TiffFile(path) as tif:
                img = tif.series[0].asarray()

            channel_img = img[ch_idx] if img.ndim == 3 else img
            H, W = channel_img.shape
            ps = self.patch_size
            normalizer = Normalize(mean=[self.mean[ch_idx]], std=[self.std[ch_idx]])

            for y in range(0, H - ps + 1, ps):
                for x in range(0, W - ps + 1, ps):
                    patch = torch.tensor(channel_img[y : y + ps, x : x + ps], dtype=torch.float32).unsqueeze(0)
                    yield normalizer(patch), x, y, fname


def run(config: dict) -> None:
    assert config["patch_size"] % 16 == 0, "patch_size must be divisible by 16"

    out_dir = Path(config["h5_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SplitChannelDataset(config)
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

    total = len(dataset.file_paths)
    h5_files: dict = {}
    file_count = 0
    active_fname = None
    patches_done = 0
    t_start = time.time()

    pbar = tqdm(total=total, desc="Progress", unit="file",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} files "
                            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    try:
        for batch, coord_x_batch, coord_y_batch, fname_batch in dataloader:
            with torch.no_grad():
                patch_emb, _, _ = model(batch.to(device))

            patch_np = patch_emb.cpu().numpy()
            cx_np = coord_x_batch.numpy()
            cy_np = coord_y_batch.numpy()

            for i, fname in enumerate(fname_batch):
                if not torch.isfinite(patch_emb[i]).all():
                    tqdm.write(f"NaN — skipping {fname} at ({cx_np[i]}, {cy_np[i]})")
                    continue

                if fname not in h5_files:
                    pbar.update(1)
                    active_fname = fname
                    hf = h5py.File(out_dir / f"{fname}.h5", "w")
                    hf.attrs["marker_names"] = fname.split("_")[-1]
                    h5_files[fname] = {
                        "file": hf,
                        "patch_ds": hf.create_dataset("feats", shape=(0, embed_dim), maxshape=(None, embed_dim), dtype="f"),
                        "coord_x_ds": hf.create_dataset("coord_x", shape=(0,), maxshape=(None,), dtype="i"),
                        "coord_y_ds": hf.create_dataset("coord_y", shape=(0,), maxshape=(None,), dtype="i"),
                    }
                    file_count += 1

                entry = h5_files[fname]
                n = entry["patch_ds"].shape[0]
                entry["patch_ds"].resize(n + 1, axis=0)
                entry["coord_x_ds"].resize(n + 1, axis=0)
                entry["coord_y_ds"].resize(n + 1, axis=0)
                entry["patch_ds"][n] = patch_np[i]
                entry["coord_x_ds"][n] = cx_np[i]
                entry["coord_y_ds"][n] = cy_np[i]

            patches_done += len(fname_batch)
            elapsed = time.time() - t_start
            patch_rate = patches_done / elapsed if elapsed > 0 else 0
            eta_total = str(timedelta(seconds=int(elapsed / pbar.n * total))) if pbar.n > 0 else "?"
            pbar.set_postfix({"patches/s": f"{patch_rate:.0f}", "ETA total": eta_total, "current": active_fname}, refresh=False)

    finally:
        pbar.close()
        for entry in h5_files.values():
            entry["file"].close()

    elapsed_total = timedelta(seconds=int(time.time() - t_start))
    print(f"\nDone. {len(h5_files)} H5 files written to {out_dir} in {elapsed_total}")


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args.config)

    # Apply CLI overrides
    if args.device:
        config["device"] = args.device
    if args.markers:
        all_markers = config["marker_order"]
        names = [m for m in all_markers if m in args.markers]
        idxs = [config["marker_order"].index(m) for m in names]
        config["marker_order"] = names
        config["marker_means"] = [config["marker_means"][i] for i in idxs]
        config["marker_stds"] = [config["marker_stds"][i] for i in idxs]

    # Auto-parallel: split markers across all available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.device is None and args.markers is None:
        all_markers = config["marker_order"]
        chunks = [all_markers[i::n_gpus] for i in range(n_gpus)]
        print(f"Launching {n_gpus} parallel workers across GPUs:")
        procs = []
        for gpu_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            print(f"  cuda:{gpu_idx} → {chunk}")
            cmd = [
                sys.executable, __file__,
                "--config", args.config,
                "--device", f"cuda:{gpu_idx}",
                "--markers", *chunk,
            ]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_idx)}
            procs.append(subprocess.Popen(cmd, env=env))
        for p in procs:
            p.wait()
    else:
        run(config)
