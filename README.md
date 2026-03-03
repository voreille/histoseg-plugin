# 🧬 Histoseg Plugin

**Histoseg Plugin** is a modular preprocessing and embedding framework for **Whole Slide Images (WSIs)** in digital pathology.  
It currently provides a complete **tiling pipeline** (segmentation → patch coordinate extraction → stitch/mask rendering) and a growing **embedding module** for deep feature extraction.

> Status: **Tiling complete ✅** · **Embeddings in progress 🚧** · **FastAPI & viewer adapters planned 🌐**

---

## ✨ Features

- **Tiling pipeline**
  - Tissue segmentation with contour/holes handling
  - Patch coordinate extraction in **level‑0 pixel space**
  - Atomic, chunked **HDF5** storage for large coord sets
  - Mask & stitched mosaic visualizations (PNG/JPG)
  - Multiprocessing for fast coordinate generation
- **Configuration**
  - YAML configs validated by **Pydantic**
  - Preset inheritance and runtime overrides
- **Jobs**
  - Batch/parallel processing
  - Slide‑level status persistence and resume
- **Storage abstraction**
  - Writer interface decoupled from FS/HDF5 (future DB-friendly)

Planned next:
- **Embedding extraction** (UNI/DINOv2/CLIP, etc.)
- **Attention MIL & heatmaps**
- **FastAPI service** + viewer adapters (Sectra, QuPath)

---

## 📦 Project Layout

```
src/histoseg_plugin/
├── embedding/                # Embedding and feature extraction
│   ├── datasets.py           # Tile and patch datasets
│   ├── encoders.py           # Feature extractor wrappers
│   ├── extract_features.py   # (WIP) Embedding pipeline entrypoint
│   ├── _torch_utils.py       # PyTorch helpers
│   └── test_handles_workers.py
│
├── tiling/                   # Core WSI tiling and preprocessing
│   ├── config_ops.py
│   ├── contours_processing.py
│   ├── contour_utils.py
│   ├── create_patches.py     # CLI for batch tiling
│   ├── jobs/                 # Job orchestration & persistence
│   │   ├── domain.py
│   │   ├── exceptions.py
│   │   ├── factory.py
│   │   ├── inspector.py
│   │   ├── runner.py
│   │   ├── run_options.py
│   │   └── store.py
│   ├── parameter_models.py   # Pydantic models for configs
│   └── process_wsi.py        # Single‑slide processing pipeline
│
├── storage/                  # File I/O and HDF5 management
│   ├── config.py             # Pydantic configuration models
│   ├── factory.py            # Store builders with config injection
│   ├── fs_writer.py          # Atomic writes (.part → final)
│   ├── h5_store.py
│   └── interfaces.py         # TilingWriter interface
│
├── wsi_core/                 # OpenSlide & visualization utilities
│   ├── annotations.py
│   ├── contour_checker.py
│   ├── geometry.py
│   ├── patch_sampling.py
│   ├── segmentation.py
│   ├── stitch.py             # Stitched mosaic rendering
│   └── visualization.py      # Mask overlays
│
└── utils/
    ├── file_utils.py
    ├── h5_utils.py
    └── __init__.py
```

**Outputs** are organized as:
```
output_dir/
├── jobs.yaml          # Job list & per‑slide status
├── manifest.yaml      # Effective merged configuration
├── masks/             # Tissue overlays (.png/.jpg)
├── patches/           # Patch coordinates (.h5)
└── stitches/          # Downsampled mosaics (.jpg)
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-org/histoseg-plugin.git
cd histoseg-plugin

# Recommended: Python 3.10+
python -m venv .venv
source .venv/bin/activate

# Editable install
pip install -e .

# Core dependencies (if not pulled via setup.cfg/pyproject)
pip install openslide-python h5py numpy pillow pydantic tqdm
# For embeddings (WIP):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **OpenSlide**: ensure system libraries are installed (e.g., `libopenslide0`).

---

## 🚀 Quickstart (Tiling)

Run the tiling CLI on a folder of WSIs:
```bash
histoseg-process   --source /data/slides   --output /results   --config configs/tiling/default.yaml
```

Use a preset tailored to a slide type:
```bash
histoseg-process   --source /data/biopsies   --output /results   --config configs/tiling/presets/biopsy.yaml
```

Resume an interrupted run (auto-skips completed slides):
```bash
histoseg-process   --source /data/slides   --output /results   --auto-skip
```

Process a curated YAML list:
```bash
histoseg-process   --process-list jobs.yaml   --output /results
```

---

## ⚙️ Configuration

Configs are **YAML** validated by **Pydantic**. They support preset inheritance and runtime overrides.

```python
from histoseg_plugin.tiling.config_ops import load_config_with_presets

config = load_config_with_presets(
    "configs/tiling/default.yaml",
    ["configs/tiling/presets/resection.yaml"],
)

# Optional runtime overrides
config = config.with_tiling_overrides(
    tile_size=512,
    target_tile_mpp=0.25,
    level_policy="lower",
)
```

Key parameters live in `tiling/parameter_models.py` and map directly to the pipeline.

---

## 🧠 Processing Pipeline (Tiling)

Core entrypoint:
```python
```python
from histoseg_plugin.tiling.process_wsi import process_single_wsi
from histoseg_plugin.storage.config import TilingStoreConfig

# Load storage configuration
store_config = TilingStoreConfig.from_yaml("configs/storage.yaml", root_key="tiling")

result = process_single_wsi(
    wsi_path="/path/slide.svs",
    tile_rootdir="/results",
    slide_rootdir="/path/to/slides",
    config=my_config,                       # Pydantic TilingConfig
    store_config=store_config,
    generate_mask=True,
    generate_patches=True,
    generate_stitch=True,
    verbose=True,
)
```
print(result)
```

Design highlights:
- Coordinates stored in **level‑0** ints
- Level selection by **MPP tolerance** and **policy** (`closest|lower|higher`)
- `FSTilingWriter` handles atomic HDF5/image writes
- Per‑contour `cont_idx` stored alongside `/coords` for fast partial reruns

---

## 🧬 Embedding Pipeline (WIP)

The **embedding** module reads HDF5 coords, extracts tiles from WSIs, and computes embeddings with pretrained encoders.

Planned workflow:
```bash
histoseg-embed   --source /results/patches   --output /results/embeddings   --model UNI2   --batch-size 64   --num-workers 4   --device cuda
```

Where:
- `embedding/datasets.py` builds tile datasets from WSI + coord HDF5
- `embedding/encoders.py` wraps torch-based encoders
- `embedding/extract_features.py` orchestrates batching & I/O
- Embeddings saved as HDF5 with metadata (model, patch size, MPP, etc.)

Schema (example):
```
/embeddings/<slide_id>   (N, D) float32
/coords/<slide_id>       (N, 2) int32          # optional copy for convenience
/meta
  model: str
  weights: str
  patch_size: int
  tile_level: int
  target_mpp: float
  created_at: str (ISO8601)
```

---

## 🧰 Troubleshooting

- **“dataclass() got an unexpected keyword argument 'slots'”**  
  You’re on Python < 3.10. Remove `slots=True` or use a compatibility shim.

- **OpenSlide errors**  
  Verify system libs: e.g., Ubuntu `sudo apt-get install libopenslide0`.

- **Huge images / DecompressionBombError**  
  Increase downscale for stitching (`downscale=64`) or adjust PIL limits if safe.

- **Chained tracebacks**  
  Exceptions are re-raised with `from e`. Use `logging.exception(...)` at your entrypoint for full stack traces.

---

## 🧭 Roadmap

- [ ] Embedding CLI (`histoseg-embed`) and unified HDF5 schema
- [ ] Attention MIL, heatmap exports
- [ ] FastAPI service (job submission, status, downloads)
- [ ] Viewer adapters (Sectra/QuPath), interactive contour edits

---

## 📜 License

MIT License — free to use for research and development.

---
## TODO

- [ ] Add mpp x and y to coords
- [ ] return a dict for the coords instead of a tuple

## 🙌 Acknowledgements

- **CLAM (Lu et al., 2021)** for MIL concepts  
- **OpenSlide** for WSI I/O  
- **HDF5** for scalable on-disk arrays

**Maintainer:** Valentin