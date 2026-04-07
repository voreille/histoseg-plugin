# Prototype v2 — Implementation Plan

## What Already Exists

Do not rewrite these unless a step explicitly requires it:

- Stitching engine: `core/segmentation/segment.py` (`run_model_and_stitch_logits`)
- Inference planning: `core/inference/planning.py` (`resolve_wsi_inference_plan`)
- Model runtime: `core/model_runtime/` (manifest, loader, torchscript, onnx, preprocessor)
- Tissue detection + GeoJSON: `core/tissue/`, `core/segmentation/geojson.py`
- Tile generation: `core/tiling/tile.py`
- Working routes: `api/routes/segment.py` (`POST /segment/wsi`, `POST /segment/tissue`)

---

## Step 1 — `TilePredictor` interface + `ModelRunnerPredictor` adapter

**Files to create:**
- `core/predictors/base.py` — abstract `TilePredictor`
- `core/predictors/model_runner.py` — `ModelRunnerPredictor` wrapping `BaseModelRunner`

**`TilePredictor` contract:**
```python
predict_tiles(tiles: Tensor) -> dict[str, Tensor]
```

- Input: `(B, C, H, W)` batch of image tiles
- Output: `head_name -> logits`, each `(B, K, H, W)`
- Batch dimension and spatial alignment with input must be preserved

**Update:** `run_model_and_stitch_logits` in `core/segmentation/segment.py` to accept a `TilePredictor` instead of `BaseModelRunner` directly.

**Notes:**
- Keep the change minimal: generalize the existing inference entry point rather than rewriting the stitching logic
- Preserve current behavior for the standard model path
- Avoid renaming public functions too early unless it clearly improves readability

---

## Step 2 — `WSIPipeline`

**Files to create:**
- `core/pipeline/wsi_pipeline.py` — `WSIPipelineParams`, `PipelineResult`, `WSIPipeline`

**Interface:**
```python
@dataclass
class WSIPipelineParams:
    seg_level: int
    sthresh: int
    sthresh_up: int
    mthresh: int
    close: int
    use_otsu: bool
    filter_params: dict
    ref_patch_size: int
    min_area_px_level0: int
    contour_fn: str
    step_size: int
    output_target_mpp: float
    batch_size: int
    num_workers: int
    # ... extend as needed

WSIPipeline.run(slide_path: Path, params: WSIPipelineParams, predictor: TilePredictor) -> PipelineResult
```

`PipelineResult` holds:
- tissue GeoJSON
- per-head annotation GeoJSON
- per-head features (placeholder until Step 3)
- metadata:
  - `coords_space`
  - `tile_count`
  - `slide_uri`
  - `selected_level`
  - `runtime`

**Refactor:** Extract all logic currently inlined in the `POST /segment/wsi` route handler into `WSIPipeline`.

The route becomes:

1. resolve slide URI → `slide_path: Path` (API concern: allowed roots, HTTP errors)
2. map `WSISegmentationRequest` → `WSIPipelineParams` (thin mapping, no logic)
3. call `WSIPipeline.run(slide_path, params, predictor)`
4. serialize `PipelineResult` → response

`WSIPipeline` should own:
- slide opening
- inference planning
- tissue segmentation
- tile generation
- predictor execution
- logit stitching
- GeoJSON conversion
- feature computation

`WSIPipeline` must not import anything from `api/`. `WSIPipelineParams` is a plain dataclass — no FastAPI or Pydantic API-schema dependencies — so it can be constructed directly in tests without HTTP machinery.

---

## Step 3 — Feature computation

**Files to create:**
- `core/features/compute.py`

Optional split if the file grows:
- `core/features/per_head.py`
- `core/features/derived.py`

**Interface:**
```python
compute_features(
    logits_by_head: dict[str, Tensor],
    manifest,
    fx: float,
    fy: float,
    mpp_x: float | None = None,
    mpp_y: float | None = None,
) -> FeatureCollection
```

### Scope

Step 3 must support both:

1. **Per-head features**
2. **Derived / cross-head features**

### Per-head features

For each head and each class:
- `area_px_level0`
- `area_mm2` (when MPP is available)
- `n_objects` (connected components)

### Derived / cross-head features

These features may require combining multiple heads, for example:
- tumor/stroma ratio by pattern
- pattern area inside tumor regions
- overlap of one head with selected classes from another head

### Implementation guidance

- Start from hard masks (`argmax`) for Prototype v2 unless a soft-probability feature is explicitly needed
- Keep feature computation separate from GeoJSON generation
- Pass `mpp_x` and `mpp_y` separately to avoid assuming isotropic resolution
- Keep the schema extensible, since future features may combine heads or metadata

**Update:**
- `api/schemas.py` — add `features` field to `WSISegmentationResponse`
- `core/pipeline/wsi_pipeline.py` — call `compute_features(...)` and include results in `PipelineResult`

---

## Step 4 — Support ROI storage

**Files to create:**
- `core/support/store.py`

**Interface:**
```python
SupportStore(root_dir)
  save_roi(support_set_id, class_label, image, metadata) -> roi_id
  list_support_sets() -> list[SupportSetInfo]
  load_class_records(support_set_id, class_label) -> list[SupportROIRecord]
```

**Filesystem layout:**
```text
support_store/
  support_set_<id>/
    manifest.json
    class_A/
      roi_001.png
      roi_001.json   # slide_uri, level-0 bbox, class label, optional MPP
    class_B/
      ...
```

### Notes

- Filesystem only
- No database
- No async jobs
- Store image files and metadata records; leave tensor loading / preprocessing to the prototype inference code
- Prefer a simple sidecar JSON format per ROI

---

## Step 5 — Prototype API routes

**Files to create:**
- `api/routes/prototype.py`

**Endpoints:**
- `POST /prototype/support/save` — crop ROIs from WSI, delegate to `SupportStore.save_roi()`
- `GET /prototype/support_sets` — delegate to `SupportStore.list_support_sets()`
- `POST /prototype/wsi` — initially a stub returning `501 Not Implemented`, then filled in Step 6

**Update:**
- `api/schemas.py` — add:
  - `SupportSaveRequest`
  - `SupportSaveResponse`
  - `SupportSetListResponse`
  - `PrototypeWSISegmentationRequest`
- `api/main.py` (or app factory) — register the new router

### Notes

- Implement support-save and support-list after Step 4
- Keep `POST /prototype/wsi` thin and reuse `WSIPipeline`
- Prototype mode may return the same top-level response structure as standard segmentation, even if its head layout is simpler

---

## Step 6 — `PrototypePredictor`

**Files to create:**
- `core/predictors/prototype.py`

**Interface:**
```python
PrototypePredictor(TilePredictor)
  __init__(support_set_id, store, encoder, preprocessor=None)
  _compute_prototypes() -> dict[class_label, Tensor]
  predict_tiles(tiles) -> dict[str, Tensor]
```

### Behavior

- Load support ROI records via `SupportStore`
- Read support crops from disk
- Apply the same preprocessing / normalization required by the encoder
- Encode support crops with the encoder used for inference
- Compute one prototype per class (initially mean embedding per class)
- Predict on target tiles using nearest-prototype similarity / distance

### Important design note

Prototype inference usually needs access to an encoder, not only a final model runner.

If helpful, define or reuse an explicit encoder-facing abstraction such as:

```python
encode_tiles(tiles: Tensor) -> Tensor
```

to avoid duplicating preprocessing or reaching awkwardly into model-runner internals.

### Output shape

For Prototype v2, keep this simple:
- return a single prototype segmentation head, for example `"prototype"`
- labels are derived from the support set classes

**Fill in `POST /prototype/wsi`:**
- load support set
- build `PrototypePredictor`
- call `WSIPipeline.run(slide_uri, req, predictor)`

The WSI pipeline remains the same; only the predictor changes.

---

## Step 7 — Move allowed roots to settings

`api/settings.py` is currently empty. Move the hardcoded `ALLOWED_ROOTS` list out of `api/routes/segment.py` and into a proper settings object (env-var-backed).

**Update:**
- define settings in `api/settings.py`
- store settings in `app.state.settings`
- update routes / pipeline to read allowed roots from settings

This step is independent and can be done at any point. It is useful cleanup, but not critical for the Prototype v2 demo.

---

## Step 8 — Minimal tests

Add a small but useful test layer to stabilize the refactor.

### Suggested tests

- `tests/test_features_compute.py`
  - area calculation
  - object count
  - derived feature examples

- `tests/test_support_store.py`
  - ROI save
  - support set listing
  - metadata round-trip

- `tests/test_wsi_pipeline.py`
  - pipeline orchestration with mocked predictor and mocked slide utilities

- optionally:
  - `tests/test_prototype_predictor.py`
  - basic prototype computation with synthetic embeddings

### Testing guidance

- Mock WSI / slide access; do not require real slide files
- Keep unit tests focused on interfaces and invariants
- Prefer lightweight synthetic tensors for feature and prototype tests

---

## Dependency Order

```text
1 (TilePredictor)
 └── 2 (WSIPipeline)
      └── 3 (Features)
4 (SupportStore)
 └── 5 (support routes)
 └── 6 (PrototypePredictor + /prototype/wsi)
7 (Settings) — independent
8 (Tests) — can be added incrementally, but should be in place before the demo if possible
```

---

## Practical Notes For Agents

- Do not rewrite the existing stitching engine unless required by Step 1
- Keep routes thin
- Reuse existing planning / tiling / tissue utilities
- Preserve level-0 coordinate conventions everywhere
- Keep prototype support storage filesystem-based
- Prefer incremental refactors over broad rewrites
- Avoid introducing parallel job architecture into Prototype v2