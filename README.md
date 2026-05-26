# Multiplex pathology images feature extraction

Standalone feature extraction pipeline for multiplexed slide images using the
[KRONOS](https://github.com/mahmoodlab/KRONOS) Vision Transformer.
Supports `.qptiff`, `.tiff`, `.tif` input.

## What is Multiplex Imaging?

**Multiplex imaging** (or **multiplexed imaging**) is a advanced tissue analysis technique that captures **multiple protein markers simultaneously** from a single tissue sample. Each marker is detected in a separate channel, allowing researchers to study the spatial relationships and co-localization patterns of different proteins in tissue.

### Key Concepts

```
Traditional Histology          Multiplex Imaging
(Single Channel)               (Multi-Channel)

       Slide                          Slide
         │                              │
    Protein A                      Protein A (Channel 1)
      only                         Protein B (Channel 2)
                                   Protein C (Channel 3)
                                   Protein D (Channel 4)
                                   DAPI nuclei (Channel 5)
                                        │
                                    One file with
                                   multiple layers
```

### Example Marker Panels

**Breast Cancer Immune Profiling:**
```
DAPI (nuclei) → HER2 (tumor) → CD3 (T cells) → CD8 (cytotoxic) → FOXP3 (regulatory)
  └─ Reveals tumor infiltration and immune landscape
```

**Colorectal Cancer T Cell Assessment:**
```
DAPI (nuclei) → PanCK (epithelium) → CD3 (T cells) → CD8 (cytotoxic) → CD4 (helper)
  └─ Maps immune cell types relative to tumor boundaries
```

**Multiplex Immune Phenotyping (6+ markers):**
```
DAPI → CD3 → CD8 → CD4 → FOXP3 → CD20 → Ki67
  └─ Complete immune ecosystem: T cells, B cells, activation status, suppression
```

### Workflow Overview

```
Multi-channel TIFF/QPTIFF file
(raw tissue stain data)
         │
         ├─ Channel 1 (DAPI) ─┐
         ├─ Channel 2 (HER2) ──┤
         ├─ Channel 3 (PanCK)──┼──→ KRONOS ViT Feature Extraction
         ├─ Channel 4 (CD8) ───┤
         └─ Channel 5 (FOXP3)─┘
                                    │
                                    ↓
                        High-dimensional 
                        embedding vectors
                        (N patches × 384/1024 dims)
                                    │
                                    ↓
                        H5 files (for downstream ML/stats)
```

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

### Extraction Modes Explained

#### Split-Channel Mode
Extracts features **per marker separately** — useful for per-marker machine learning:
```
Input:  patient_001.qptiff (5 channels: DAPI, HER2, PanCK, CD8, FOXP3)
                     │
        ┌────────────┼────────────┬──────────────┬──────────────┐
        ↓            ↓            ↓              ↓              ↓
    DAPI only    HER2 only    PanCK only    CD8 only      FOXP3 only
    ViT encoder  ViT encoder  ViT encoder  ViT encoder   ViT encoder
        │            │            │              │              │
        ↓            ↓            ↓              ↓              ↓
Output: patient_001_DAPI.h5, patient_001_HER2.h5, ... (one file per marker)
```

**Best for**: Per-marker analysis, training separate models per biomarker, comparing marker distributions

#### Multi-Channel Mode
Extracts features **using all markers jointly** — preserves multi-marker context:
```
Input:  patient_001.qptiff (5 channels: DAPI, HER2, PanCK, CD8, FOXP3)
                     │
                  Combined input
                (all 5 channels together)
                     │
                ViT encoder
                (single pass)
                     │
    ┌────────────────┼────────────────┐
    ↓                ↓                 ↓
CLS embedding   Marker embeddings  Token embeddings
(global)        (per-marker)       (spatial tokens)
    │                │                 │
    └────────────────┼─────────────────┘
                     ↓
            patient_001.h5 (multi-channel)
            Contains all features from one pass
```

**Best for**: Multi-marker analysis, learning joint representations, spatial relationships between markers

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
  num_workers: 0   
```

**⚠️ Important**: `num_workers` should be `0` for **multi-channel mode**. Each worker loads the entire slide into memory to yield patches — setting `num_workers > 0` causes out-of-memory errors. For split-channel mode, `num_workers > 0` is safe.

---

## Usage

### Split-channel extraction

Writes one H5 per `(patient, marker)` pair — ideal for per-marker MIL:

```bash
python extract_features_split_channels.py --config multiplex_config.yaml
```

- **Default**: one-by-one over all remaining slides on one process.
- **Parallel**: opt-in with `--parallel` (markers distributed round-robin across GPUs).
- Skips any `(patient, marker)` pair whose H5 already exists — safe to re-run
- Progress bar shows elapsed, remaining, ETA total, patches/s, and current file:
  ```
  Progress:  12%|█▏  | 6/50 files [00:42<05:01, patches/s=312, ETA total=0:05:43, current=E-21-1234_HER2]
  ```

Common options:
```bash
# choose GPU by index (simple form)
python extract_features_split_channels.py --config multiplex_config.yaml --device 1

# extract only specific markers
python extract_features_split_channels.py --config multiplex_config.yaml --markers HER2 DAPI

# opt in to multi-GPU parallel workers
python extract_features_split_channels.py --config multiplex_config.yaml --parallel
```

### Multi-channel extraction

Writes one H5 per patient with all marker embeddings:

```bash
python extract_features_multi_channels.py --config multiplex_config.yaml
```

- **Default**: one-by-one over all remaining slides on one process.
- **Parallel**: opt-in with `--parallel` (slides split across GPUs).
- Skips slides whose H5 already exists — safe to re-run
- Channels without `mean`/`std` in the config are passed through without normalization
- Images with more channels than defined markers are sliced to match; images with fewer are skipped

Common options:
```bash
# choose GPU by index (simple form)
python extract_features_multi_channels.py --config multiplex_config.yaml --device 0

# opt in to multi-GPU parallel workers
python extract_features_multi_channels.py --config multiplex_config.yaml --parallel
```

Options:
- `--num-slides N` — how many slides to process
- `--markers` — subset of markers (space-separated)
- `--patch-size`, `--batch-size`, `--num-workers` — override config values
- `--overwrite` — delete previous test output before running
- `--seed N` — deterministic slide selection
- `--slide-pattern GLOB` — restrict to one format (e.g. `'*.qptiff'`); default discovers all formats

---

## Downstream Training (optional)

After we have extracted feaure we can use [STAMP](https://github.com/KatherLab/STAMP) CLI for training, cross-validation, statistics, and heatmaps. It requires STAMP to be installed

---

## Visual Examples & Real Data

### Real Workflow & Channel Images

**Multiplex Immunofluorescence Algorithm Workflow:**
![Algorithm Workflow](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/4911/9271766/262603756c8e/fonc-12-889886-g006.jpg)

**Multi-Channel Separation Example (Activation Panel):**
![Channel Separation](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/4911/9271766/1486322af81c/fonc-12-889886-g001.jpg)

**Image Preparation & Tissue Segmentation:**
![Image Prep](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/4911/9271766/ab1f3d8d7b62/fonc-12-889886-g003.jpg)

### Research Publications

These publications contain additional real multiplex tissue images, visualizations, and datasets:

- **[CyCIF Method Overview](https://www.tissue-atlas.org/cycif-method)** — Harvard Tissue Atlas with interactive visualizations of cyclic immunofluorescence imaging
- **[Nature Cancer - High-plex imaging biomarkers](https://www.nature.com/articles/s43018-023-00576-1)** — Side-by-side traditional and multiplex histology with HER2, immune markers
- **[Nature Methods - 3D multiplexed profiling (2025)](https://www.nature.com/articles/s41592-025-02824-x)** — Latest high-multiplex 3D imaging techniques
- **[npj Breast Cancer - HER2 multiplexed imaging](https://www.nature.com/articles/s41523-023-00605-3)** — Multiplex detection of HER2 patterns with DAPI, CD8, CD20 panels
- **[eLife - t-CyCIF imaging techniques](https://elifesciences.org/articles/31657)** — Highly multiplexed immunofluorescence with conventional microscopes
- **[Scientific Data - Immune markers dataset](https://www.nature.com/articles/s41597-019-0332-y)** — Multiplexed images + single-cell data of tonsil and lung cancer
- **[CIO DFCI - Multiplex IF Technology](https://ciopath.dfci.harvard.edu/technology-platforms/multiplex-immunofluorescence)** — Harvard facility with example workflows

### KRONOS Foundation Model

- **[KRONOS GitHub](https://github.com/mahmoodlab/KRONOS)** — Vision Transformer trained on 47M patches spanning 175 protein markers
- **[KRONOS on HuggingFace](https://huggingface.co/MahmoodLab/KRONOS)** — Pretrained weights (requires registration)
- **[KRONOS Paper](https://arxiv.org/html/2506.03373v1)** — Foundation Model for Spatial Proteomics

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
