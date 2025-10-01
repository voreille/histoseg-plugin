# Histoseg Plugin - AI Coding Instructions

## Project Overview
You're building **histoseg-plugin**: a viewer-agnostic histopathology inference core exposed as a FastAPI service with thin adapters (QuPath now, Sectra later).

The core runs a modular pipeline—**tiling → feature extraction → classification/attention → heatmap** (and optional polygons)—implemented as stateless, function-first code inside a single Python package (`src/histoseg_plugin`) and mirrored by a small CLI.

All geometry uses **level-0 pixel coordinates**; intermediate data lives in **HDF5**; final overlays are **level-0–aligned PNG heatmaps** (plus optional vector contours) served from `/static`.

Adapters only pass a **WSI URI + ROI** and render the returned overlays; if the backend can't read the file, they export a temp OME-TIFF or stream tiles as fallback.

**Constraints**: keep adapters thin, keep the API stable, no CLAM coupling (CLAM is inspiration only), and make the core Docker-ready and easily swappable across viewers.

## Architecture

### Core Components
- **FastAPI Service**: Main application framework (planned - not yet implemented)
- **Modular Pipeline**: Stateless, function-first processing in `src/histoseg_plugin/`
- **CLAM Resources**: Reference implementation in `resources/` (inspiration only, not coupled)
- **Adapter System**: Thin viewer adapters (QuPath implemented, Sectra planned)

### Key Data Flow
1. **Tiling** → **Feature Extraction** → **Classification/Attention** → **Heatmap** (+ optional polygons)
2. **Coordinate System**: All geometry uses level-0 pixel coordinates
3. **Data Storage**: Intermediate data in HDF5, final overlays as level-0–aligned PNG heatmaps
4. **API Interface**: WSI URI + ROI input → overlay outputs served from `/static`

## Critical Patterns

### Pipeline Architecture
**Stateless, function-first design** in `src/histoseg_plugin/`:
- Each pipeline stage as independent, composable functions
- No CLAM coupling (inspiration only from `resources/`)
- Docker-ready, viewer-agnostic core

### Contour Processing API
Optimized multiprocessing for patch coordinate generation in `src/histoseg_plugin/preprocessing/contour_proc.py`:
```python
# Single contour processing
process_single_contour(contour, contour_holes, level_dim0, level_downsamples, ...)
# Batch processing to HDF5
process_contours_to_hdf5(contours_tissue, holes_tissue, ...)
```

### Adapter Pattern
**Keep adapters thin** - they only:
- Pass WSI URI + ROI to core
- Render returned overlays
- Handle fallbacks (temp OME-TIFF, tile streaming) if core can't read files


### File Formats & Conventions
- **Geometry**: Level-0 pixel coordinates throughout pipeline
- **Intermediate Data**: HDF5 storage with `coords` dataset and metadata attributes
- **Final Outputs**: Level-0–aligned PNG heatmaps + optional vector contours
- **API Outputs**: Served from `/static` endpoint
- **Configuration**: YAML configs in `resources/CLAM/heatmaps/configs/` (reference)
- **Presets**: CSV parameter templates in `resources/CLAM/presets/` (reference)

## Development Workflows

### Custom Preprocessing
- Import from `wsi_core.util_classes` for contour checking functions
- Use multiprocessing patterns from `contour_proc.py` for performance
- Always specify absolute paths for HDF5 operations

### Typical WSI Processing Commands
```bash
# Fast patching (coordinates only)
python resources/CLAM/create_patches_fp.py --source DATA_DIR --save_dir RESULTS_DIR --patch_size 256 --seg --patch --stitch

# Feature extraction
python resources/CLAM/extract_features_fp.py --data_h5_dir PATCHES_DIR --data_slide_dir SLIDES_DIR --feat_dir FEATURES_DIR

# Training
python resources/CLAM/main.py --data_root_dir FEATURES_DIR --split_dir SPLITS_DIR

# Heatmap generation
python resources/CLAM/create_heatmaps.py --config config_template.yaml
```

## Key Dependencies & Integration
- **OpenSlide**: WSI format support (.svs, .ndpi, .tiff)
- **PyTorch + Timm**: Deep learning models and pretrained encoders
- **H5py**: Efficient patch coordinate and feature storage
- **OpenCV**: Image processing and contour operations
- **FastAPI + Pydantic**: API framework (project goal)

## Important Notes
- The main FastAPI application is not yet implemented - this is framework preparation
- CLAM uses weak supervision (slide-level labels only, no patch annotations)
- Memory-efficient: new pipeline stores coordinates, loads patches on-demand
- Default patch size: 256x256, resized to 224x224 for feature extraction
- Supports multiple encoder models: ResNet50, UNI, CONCH

## File Structure Conventions
```
├── src/histoseg_plugin/          # Main plugin code
├── resources/CLAM/               # CLAM reference implementation (inspiration)
├── adapters/                     # Viewer adapter implementations (empty)
├── configs/                      # Plugin configurations (empty)
├── scripts/                      # Automation scripts (empty)
```

When implementing new features, follow the HDF5 + multiprocessing patterns established in `contour_proc.py` and integrate with CLAM's coordinate-based pipeline for efficiency.