# Multiplex images feature extraction

Standalone feature extraction pipeline for multiplexed slide images using the
[KRONOS](https://github.com/mahmoodlab/KRONOS) Vision Transformer.
Supports `.qptiff`, `.tiff`, `.tif` input.

## Overview

Multiplexed tissue imaging produces multi-channel image files where each channel
corresponds to one protein marker (e.g. HER2, DAPI, PanCK). This project uses the
[KRONOS](https://github.com/mahmoodlab/KRONOS) pretrained ViT to embed image patches
into feature vectors for downstream analysis.

The H5 output files follow the [STAMP](https://github.com/KatherLab/STAMP) feature
format and can be used directly with STAMP for MIL training, cross-validation,
statistics, and heatmap generation.

Supported input formats: `.qptiff`, `.tiff`, `.tif`

Two extraction modes are supported:

| Mode | Script | Output per patient |
|------|--------|--------------------|
| **Split-channel** | `extract_features_split_channels.py` | One H5 per `(patient, marker)` |
| **Multi-channel** | `extract_features_multi_channels.py` | One H5 per patient (all markers) |

### H5 contents

**Split-channel** (`{patient_stem}_{marker}.h5`):
```
feats    (N, embed_dim)   float32   — patch CLS embeddings
coord_x  (N,)             int32
coord_y  (N,)             int32
attrs:   marker_names = marker_name
```

**Multi-channel** (`{patient_stem}.h5`):
```
feats              (N, embed_dim)                              float32
marker_embeddings  (N, num_markers, embed_dim)                 float32
token_embeddings   (N, num_markers, T, T, embed_dim)           float32
coord_x            (N,)                                        int32
coord_y            (N,)                                        int32
attrs:             marker_names = [list of marker names]
```

---

## Data structure

### Input

Slides are discovered recursively under `multiplex_image_path`. Supported formats:

```
Tumorpanel_WSI/
├── patient_001/
│   └── slide.qptiff          # multi-channel (C, H, W) — one channel per marker
├── patient_002/
│   └── slide.tiff
└── ...
```

Files inside hidden directories (e.g. `.temp/`) are automatically ignored.
Channel order in the image **must match** the order of `markers` in the config.

### Output — split-channel mode

One H5 file per `(patient, marker)`:

```
h5_path/
├── patient_001_DAPI.h5
├── patient_001_HER2.h5
├── patient_002_DAPI.h5
└── ...
```

Each H5 contains:
```
feats    (N, embed_dim)   float32   — patch CLS token embeddings
coord_x  (N,)             int32     — patch x coordinate (pixels)
coord_y  (N,)             int32     — patch y coordinate (pixels)
attrs:   marker_names = "DAPI"
```

### Output — multi-channel mode

One H5 file per patient:

```
h5_path/
├── patient_001.h5
├── patient_002.h5
└── ...
```

Each H5 contains:
```
feats              (N, embed_dim)                              float32  — CLS embeddings
marker_embeddings  (N, num_markers, embed_dim)                 float32  — per-marker embeddings
token_embeddings   (N, num_markers, T, T, embed_dim)           float32  — spatial token embeddings
coord_x            (N,)                                        int32
coord_y            (N,)                                        int32
attrs:             marker_names = ["DAPI", "HER2", ...]
```

Where `N` = number of patches, `T` = `patch_size // 16`, `embed_dim` = 384 (vits16) or 1024 (vitl16).

---

## Setup

### Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- A CUDA-capable GPU (recommended; CPU works but is slow)
- HuggingFace account with access to [MahmoodLab/kronos](https://huggingface.co/MahmoodLab/kronos)

### Install

```bash
uv sync
source .venv/bin/activate
```

GPU (CUDA) torch is installed by default. No extra flags needed.

---

## Configuration

The provided `.yaml` files are starting-point examples — copy and adapt one for your
own panel and paths.

Key fields:

```yaml
multiplex_image_path: "/path/to/folder/containing/slides"  # *.qptiff, *.tiff, *.tif
device: "cuda"          # or "cpu" / "cuda:1"

model:
  checkpoint_path: "hf_hub:MahmoodLab/kronos"
  hf_auth_token: "hf_..."           # HuggingFace token — get one at https://huggingface.co/MahmoodLab/KRONOS
  cache_dir: "./model_assets"        # local model cache
  model_type: "vits16"               # "vits16" (embed_dim=384) or "vitl16" (embed_dim=1024) currently only vits16 is available
  token_overlap: false               # true → stride 8 (denser, slower)

markers:
  - name: "DAPI"
    mean: 36.49        # per-channel normalization mean
    std:  71.71        # per-channel normalization std
  # ... one entry per channel, in qptiff channel order

split_channels:
  h5_path: "/path/to/output/h5"
  markers_to_extract: ["HER2"]       # null to extract all markers
  patch_size: 64                     # must be divisible by 16
  batch_size: 128
  num_workers: 8

multi_channels:
  h5_path: "/path/to/output/h5"
  markers_to_extract: null           # null = all markers
  patch_size: 64
  batch_size: 128
  num_workers: 8
```

---

## Usage

### Split-channel extraction

Writes one H5 per `(patient, marker)` pair — ideal for per-marker MIL:

```bash
python extract_features_split_channels.py --config multiplex_config.yaml
```

- **Multi-GPU**: automatically detected — markers are distributed round-robin across all GPUs, one subprocess per GPU. No flags needed.
- Skips any `(patient, marker)` pair whose H5 already exists — safe to re-run
- Progress bar shows elapsed, remaining, ETA total, patches/s, and current file:
  ```
  Progress:  12%|█▏  | 6/50 files [00:42<05:01, patches/s=312, ETA total=0:05:43, current=E-21-1234_HER2]
  ```

Override GPU or markers from the command line:
```bash
# force a specific GPU (disables auto-parallel)
python extract_features_split_channels.py --config multiplex_config.yaml --device cuda:1

# extract only specific markers (disables auto-parallel)
python extract_features_split_channels.py --config multiplex_config.yaml --markers HER2 DAPI
```

### Multi-channel extraction

Writes one H5 per patient with all marker embeddings:

```bash
python extract_features_multi_channels.py --config multiplex_config.yaml
```

- **Multi-GPU**: automatically detected — slides are split evenly across all GPUs, one subprocess per GPU. No flags needed.
- Skips slides whose H5 already exists — safe to re-run
- Channels without `mean`/`std` in the config are passed through without normalization
- Images with more channels than defined markers are sliced to match; images with fewer are skipped

Override GPU or slide range from the command line:
```bash
# force a specific GPU (disables auto-parallel)
python extract_features_multi_channels.py --config multiplex_config.yaml --device cuda:0

# process a specific slice of slides (used internally by the parallel launcher)
python extract_features_multi_channels.py --config multiplex_config.yaml --device cuda:0 --slide-start 0 --slide-end 10
```

### Quick test run (subset of slides)

```bash
python run_split_channels_test.py \
    --base-config multiplex_config.yaml \
    --num-slides 3 \
    --markers HER2 DAPI \
    --device cuda \
    --work-dir temp/test_run
```

Options:
- `--num-slides N` — how many slides to process
- `--markers` — subset of markers (space-separated)
- `--patch-size`, `--batch-size`, `--num-workers` — override config values
- `--overwrite` — delete previous test output before running
- `--seed N` — deterministic slide selection
- `--slide-pattern GLOB` — restrict to one format (e.g. `'*.qptiff'`); default discovers all formats

---

## Training (optional)

After we have extracted feaure we can use [STAMP](https://github.com/KatherLab/STAMP) CLI for training, cross-validation, statistics, and heatmaps. It requires STAMP to be installed

---

## References

- **KRONOS** — pretrained ViT for multiplexed imaging: https://github.com/mahmoodlab/KRONOS
- **KRONOS model weights** — HuggingFace Hub: https://huggingface.co/MahmoodLab/KRONOS
- **STAMP** — MIL training, statistics, heatmaps: https://github.com/KatherLab/STAMP

---

## Notes

- **Resumability**: both extraction scripts skip slides whose H5 already exists.
  If a run is interrupted, simply re-run the same command.
- **NaN patches**: patches where the model produces non-finite embeddings are
  silently skipped with a warning.
- **xFormers**: if installed (`uv sync --extra xformers` after the base install),
  memory-efficient attention is used automatically for faster GPU inference.
- **Model weights**: downloaded from HuggingFace Hub on first run and cached in
  `cache_dir` (default `./model_assets`). Requires `hf_auth_token`.
