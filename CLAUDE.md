# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**histoseg-plugin** is a digital pathology project centered on **whole slide image (WSI) processing**.

There are two relevant parts of the codebase:

1. **Existing tiling / preprocessing pipeline**
   - tissue segmentation
   - patch coordinate extraction
   - stitched visualization
   - storage in HDF5 / YAML

2. **Active Prototype v2 development**
   - FastAPI-based WSI segmentation service
   - tile-level model inference
   - stitched whole-slide predictions
   - GeoJSON annotation output
   - quantitative feature computation
   - prototype-based segmentation from user-selected support ROIs

The active development target is **Prototype v2**, not the older batch tiling CLI.

## Current Priorities (Prototype v2)

Focus on these tasks first:

1. Refactor `POST /segment/wsi` into a reusable `WSIPipeline`
2. Add quantitative feature computation to responses
3. Implement support ROI storage on the filesystem
4. Implement full-WSI prototype segmentation using saved support sets

Explicitly **out of scope** for v2:

- batch processing of multiple WSIs
- database integration
- async job queues
- persistent caching of full segmentation outputs
- polished model-zoo UI
- Sectra-specific integration logic

## Critical Invariants (DO NOT BREAK)

These rules are essential.

- **All coordinates are expressed in level-0 space**
- Tile coordinates are always **level-0 top-left coordinates**
- GeoJSON outputs must always be in **level-0 coordinates**
- Stitching uses **weighted accumulation**:
  - accumulate logits into a sum canvas
  - accumulate weights into a weight canvas
  - final logits = sum / weight
- Background classes must **not** be converted to output polygons unless explicitly requested
- Do not silently change MPP / level-selection conventions
- Do not introduce a second competing tiling or stitching implementation

Violating these invariants will break downstream consumers such as QuPath and feature computation.

## Prototype v2 Pipeline

Prototype v2 performs whole-slide segmentation with the following steps:

1. Tissue segmentation
2. Tile generation inside tissue regions
3. Tile-level prediction using a predictor
4. Logit stitching into a whole-slide canvas
5. Conversion of logits to GeoJSON annotations
6. Feature computation per head / class

High-level flow:

```text
WSI
 └── Tissue segmentation
      └── Tile generation
           └── Predictor
                └── Logit stitching
                     └── GeoJSON conversion
                     └── Feature computation
```

## Predictor-Agnostic Design

The WSI pipeline must be **predictor-agnostic**.

Standard model inference and prototype inference must reuse the same orchestration code. The only difference should be the predictor implementation.

### TilePredictor contract

```python
predict_tiles(tiles: Tensor) -> dict[str, Tensor]
```

Expected behavior:

- Input: batch of image tiles with shape `(B, C, H, W)`
- Output: dictionary mapping `head_name -> logits`
- Each logit tensor should normally have shape `(B, K, H, W)`
- Batch dimension must be preserved
- Spatial alignment with the input tile must be preserved
- Predictor implementations should not silently change coordinate space

Planned implementations:

- `ModelRunnerPredictor` → standard deep learning model
- `PrototypePredictor` → prototype-based inference from saved support ROIs

## Whole-Slide Prototype Segmentation

Prototype inference is required to run on the **full WSI**, not only on selected ROIs.

Reason:
ROI-only inference hides important failure modes such as:
- false positives in non-cancer tissue
- poor generalization outside support regions
- spurious prototype activation in unrelated tissue structures

Expected workflow:

1. User selects ROIs in QuPath
2. ROIs are exported to the server with class labels
3. Server stores crops as a support set
4. Server computes prototypes from the support crops
5. Full WSI is segmented using `PrototypePredictor`

## API Endpoints (Prototype v2)

### `POST /segment/wsi`
Standard WSI segmentation.

Returns:
- tissue GeoJSON
- per-head annotation GeoJSON
- per-head quantitative features

### `POST /prototype/wsi`
Prototype-based WSI segmentation using a saved support set.

Returns:
- same structure as `/segment/wsi`

### `POST /prototype/support/save`
Save selected support ROIs from a WSI.

Behavior:
- crop ROIs from slide
- store them on disk
- associate them with a support set and class labels

### `GET /prototype/support_sets`
List available support sets and their classes.

## Feature Computation

For each output head and class, compute at least:

- `area_px_level0`
- `area_mm2` when slide MPP is available
- `n_objects` (connected components)

Feature schema should stay separate from GeoJSON annotations.

## Support ROI Storage

Use a simple **filesystem-based** storage scheme for prototype v2.

Suggested layout:

```text
support_store/
  support_set_<id>/
    manifest.json
    class_A/
      roi_001.png
      roi_001.json
    class_B/
      roi_002.png
      roi_002.json
```

Each ROI metadata file should contain at least:

- `slide_uri`
- level-0 bounding box
- class label
- optional MPP metadata

Do not introduce a database for this stage.

## Existing Codebase Context

This repository already contains an older WSI tiling / preprocessing pipeline.

Important existing behaviors:

- coordinates are level-0
- MPP-based level selection already exists
- storage and job abstractions exist for tiling
- some CLI / YAML job logic exists for older workflows

That older code is useful context, but Prototype v2 should not be forced into the older batch/job architecture if that makes the FastAPI service harder to build.

## Where To Implement Changes

Preferred areas for new work:

- API routes → `api/routes/`
- Core WSI orchestration → `core/pipeline/`
- Prediction abstractions → `core/inference/` or `core/predictors/`
- GeoJSON conversion → `core/segmentation/geojson.py`
- Feature computation → `core/features/`
- Support ROI storage → `core/support/`
- Model runtime / manifests → `core/model_runtime/`

If refactoring existing code:
- keep routes thin
- move logic out of route handlers
- reuse existing utilities instead of duplicating them

## Design Rules

- Keep FastAPI routes thin
- Put core logic in `core/`
- Avoid duplicating tiling or stitching logic
- Prefer modular predictors over branching pipelines
- Keep storage filesystem-based for Prototype v2
- Preserve coordinate conventions
- Preserve existing working behavior unless the task explicitly asks for redesign

## Practical Guidance For Agents

When making changes:

1. Propose structure first if the task is architectural
2. Keep edits local and minimal when possible
3. Do not rewrite unrelated legacy tiling code unless necessary
4. If adding a new abstraction, explain how it connects to existing modules
5. Prefer incremental refactors over broad rewrites

## Testing Conventions

- Framework: `pytest`
- One test file per module: `test_<module>.py` in `tests/`
- Use fixtures for shared setup
- Never require real WSI files — mock all slide I/O
- Run `pytest -m "not slow"` after each implementation step

## Non-Goals For This File

This file is not intended to document:
- all CLI usage
- all legacy batch-processing features
- all install instructions
- all experiments
