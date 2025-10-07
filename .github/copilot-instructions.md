# Histoseg Plugin - AI Coding Instructions

## Project Overview
You're building **histoseg-plugin**: a WSI preprocessing and tiling pipeline with a planned FastAPI service and pluggable viewer adapters.

Currently implements the **tiling stage** with tissue segmentation, patch coordinate extraction, and visualization stitching. The pipeline uses **level-0 pixel coordinates** throughout, stores intermediate data in **HDF5**, and outputs masks/stitches as images.

The project follows a **job-based processing model** with YAML configurations, parameter validation via Pydantic, and CLI tools for batch processing.

## Current Implementation Status

### Implemented Components
- **Tiling Module**: Complete WSI preprocessing pipeline in `src/histoseg_plugin/tiling/`
- **Job System**: Parallel processing with YAML job persistence and status tracking
- **CLI Tools**: `histoseg-process` for batch processing, `histoseg-params` (planned) for parameter management
- **Configuration**: YAML-based configs with preset inheritance system
- **OpenSlide Integration**: WSI format support with multi-level pyramid handling

### Planned/Missing Components
- **FastAPI Service**: REST API for external integrations (dependencies declared but not implemented)
- **Feature Extraction**: Deep learning pipeline for patch embeddings  
- **Classification/Attention**: ML models for generating attention heatmaps
- **Viewer Adapters**: QuPath and Sectra integration modules

## Critical Architecture Patterns

### Job Processing System
The core pattern is **job-based batch processing** with persistent state:
```python
# Create jobs from directory or YAML
joblist = jobs_from_dir(source_dir, config, exts=(".svs", ".tiff"))
# or
joblist = jobs_from_yaml("jobs.yaml", slides_root=source_dir)

# Run with status persistence
run_tiling_jobs(joblist, output_dir, store=YamlJobStore(...), ...)
```

### Configuration Hierarchy
**Pydantic models** with **YAML config inheritance**:
```python
# Base config (configs/tiling/default.yaml)
Config(tiling=Tiling(...), seg_params=SegmentationParams(...), ...)

# Apply presets (configs/tiling/presets/*.yaml) 
config = load_config_with_presets(default_yaml, presets=[...])

# CLI overrides via .with_tiling_overrides()
```

### WSI Processing Pipeline
**Stateless functions** with **level-aware coordinate handling**:
```python
# Auto-select pyramid level based on target MPP
level, mpp, within_tolerance, reason = select_tile_level(
    wsi_obj, tile_level=-1, target_tile_mpp=0.30, level_policy="closest"
)

# Process single WSI through full pipeline
result = process_single_wsi(wsi_path, output_dir, config, 
                           generate_mask=True, generate_patches=True, generate_stitch=True)
```

## Key File Locations & Patterns

### Configuration System
- **Base Config**: `configs/tiling/default.yaml` - complete parameter specification
- **Presets**: `configs/tiling/presets/*.yaml` - partial overrides for specific slide types (biopsy, resection, tcga, high_quality)
- **Job Files**: Output `jobs.yaml` contains per-slide configs and processing status
- **Parameter Models**: `src/histoseg_plugin/tiling/parameter_models.py` - Pydantic validation

### Core Processing Files
- **Main CLI**: `src/histoseg_plugin/tiling/create_patches.py` - batch processing entry point
- **WSI Processing**: `src/histoseg_plugin/tiling/jobs/process_wsi.py` - core single-slide logic
- **Job Management**: `src/histoseg_plugin/tiling/jobs/` - parallel execution, persistence, error handling
- **WSI Wrapper**: `src/histoseg_plugin/tiling/WholeSlideImage.py` - OpenSlide integration with tissue segmentation

### Output Structure
```
output_dir/
├── jobs.yaml          # Job definitions with status tracking
├── manifest.yaml      # Final effective configuration
├── masks/             # Tissue segmentation visualizations (.jpg)
├── patches/           # Patch coordinates in HDF5 (.h5)
└── stitches/          # Downsampled patch mosaics (.jpg)
```

## Development Workflows

### CLI Usage Patterns
```bash
# Basic batch processing
histoseg-process --source /data/slides --output /results --config configs/tiling/default.yaml

# With preset for specific slide type
histoseg-process --source /data/biopsies --output /results --config configs/tiling/presets/biopsy.yaml

# Resume interrupted processing (auto-skip completed slides)
histoseg-process --source /data/slides --output /results --auto-skip

# Process specific slides from YAML list
histoseg-process --process-list jobs.yaml --output /results
```

### Configuration Development
```python
# Load config with preset merging
config = load_config_with_presets(
    "configs/tiling/default.yaml", 
    ["configs/tiling/presets/biopsy.yaml"]
)

# Runtime parameter overrides (immutable - returns new Config)
config = config.with_tiling_overrides(
    tile_size=512, target_tile_mpp=0.25, level_policy="lower"
)
```

### Extending Processing Logic
When adding new functionality:
1. **Add parameters** to appropriate Pydantic models in `parameter_models.py`
2. **Update validation** in model `@model_validator` methods
3. **Modify processing** in `process_single_wsi()` function
4. **Test with presets** to ensure configuration inheritance works
5. **Update job result tracking** in `TilingResult` dataclass

## Important Implementation Details

### Level Selection Logic
The pipeline auto-selects WSI pyramid levels based on target MPP and policy:
```python
# Auto-select closest level to target MPP
select_tile_level(wsi_obj, tile_level=-1, target_tile_mpp=0.30, 
                 mpp_tolerance=0.1, level_policy="closest")
# Returns: (chosen_level, actual_mpp, within_tolerance, reason)
```

### Multiprocessing Coordination
- **Patch extraction** uses `multiprocessing.Pool` for coordinate generation
- **Job execution** supports both serial and parallel processing via `ProcessPoolExecutor`
- **File I/O** is thread-safe with atomic writes and locking for shared job state

### Error Handling Strategy
- **Typed exceptions** in `jobs/exceptions.py` for different failure modes
- **Per-slide error tracking** with detailed error messages in job state
- **Partial failure tolerance** - individual slide failures don't stop batch processing
- **Resume capability** - failed jobs can be retried without reprocessing successful ones

### Memory Management
- **Streaming HDF5 writes** for large patch coordinate datasets
- **Level-aware memory limits** for tissue segmentation (configurable pixel thresholds)
- **On-demand patch loading** - coordinates stored, patches extracted when needed

## Integration Points

### CLAM Compatibility
The `resources/CLAM/` directory contains reference implementations but is **not coupled** to the main pipeline. Use for inspiration only:
- Config formats differ (YAML vs hardcoded parameters)
- Processing logic is rewritten for modularity
- HDF5 schemas are compatible for downstream feature extraction

### Future FastAPI Integration
When implementing the API service:
- Reuse `process_single_wsi()` function as core processing logic
- Expose job status endpoints using existing `TilingJobCollection` and store patterns
- Serve static outputs from `/static` endpoint (masks, stitches)
- Use existing Pydantic models for request/response validation