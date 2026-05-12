# Multiplex Feature Extraction

Feature extraction pipeline for multiplexed slide images using [KRONOS](https://github.com/mahmoodlab/KRONOS) Vision Transformer.

Supports: `.qptiff`, `.tiff`, `.tif`

## Two modes

| Mode | Output | Use case |
|------|--------|----------|
| **Split-channel** | One H5 per (patient, marker) | Per-marker MIL |
| **Multi-channel** | One H5 per patient (all markers) | Multi-marker analysis |

---

## Setup

```bash
uv sync
source .venv/bin/activate
```

Requirements: Python 3.13, GPU, HuggingFace account with access to [MahmoodLab/kronos](https://huggingface.co/MahmoodLab/kronos)

---

## Configuration

Create a `.yaml` config file (copy and edit an example):

```yaml
multiplex_image_path: "/path/to/slides"   # *.qptiff, *.tiff, *.tif files

device: "cuda"

model:
  checkpoint_path: "hf_hub:MahmoodLab/kronos"
  hf_auth_token: "hf_..."                # Get from HuggingFace
  cache_dir: "./model_assets"
  model_type: "vits16"                   # vits16 or vitl16
  token_overlap: false

markers:
  - name: "DAPI"
    mean: 36.49
    std: 71.71
  # ... one per channel, in image order

split_channels:
  h5_path: "/path/to/output"
  markers_to_extract: null
  patch_size: 32
  batch_size: 128
  num_workers: 8

multi_channels:
  h5_path: "/path/to/output"
  markers_to_extract: null
  patch_size: 32
  batch_size: 128
  num_workers: 0
```

⚠️ **OOM with num_workers**: If you get out-of-memory errors with multi-channel extraction, change `num_workers: 0` in the config (multi-channel loads full slides per worker).

---

## Usage

### Split-channel (one H5 per marker per patient)

```bash
python extract_features_split_channels.py --config config.yaml
```

- Auto-detects GPUs and distributes markers across them
- Skips existing H5 files — safe to re-run

Override GPU or markers:
```bash
python extract_features_split_channels.py --config config.yaml --device cuda:0
python extract_features_split_channels.py --config config.yaml --markers DAPI HER2
```

### Multi-channel (one H5 per patient with all markers)

```bash
python extract_features_multi_channels.py --config config.yaml
```

- Auto-detects GPUs and splits slides across them
- Skips existing H5 files — safe to re-run

Override GPU:
```bash
python extract_features_multi_channels.py --config config.yaml --device cuda:0
```

### Output H5 structure

**Split-channel**: `{patient_stem}_{marker}.h5`
- `feats` (N, embed_dim) — patch embeddings
- `coord_x`, `coord_y` — patch coordinates

**Multi-channel**: `{patient_stem}.h5`
- `feats` (N, embed_dim) — patch embeddings
- `marker_embeddings` (N, num_markers, embed_dim) — per-marker embeddings
- `token_embeddings` (N, num_markers, T, T, embed_dim) — spatial token embeddings
- `coord_x`, `coord_y` — patch coordinates

Where `N` = patches, `T` = patch_size/16, `embed_dim` = 384 (vits16) or 1024 (vitl16)

---

## Training (optional)

After feature extraction, use [STAMP](https://github.com/KatherLab/STAMP) for MIL training, cross-validation, statistics, and heatmaps.

---

## Notes

- **Resumability**: Re-run the same command after interruption — existing H5 files are skipped
- **Marker stats**: If `mean`/`std` are null in config, auto-fetches from `marker_metadata.csv` in config directory (case-insensitive match)
- **NaN patches**: Patches with non-finite embeddings are skipped with a warning
- **Model caching**: Downloaded to `cache_dir` on first run

---

## References

- [KRONOS](https://github.com/mahmoodlab/KRONOS) — ViT for multiplexed imaging
- [STAMP](https://github.com/KatherLab/STAMP) — MIL training & analysis
