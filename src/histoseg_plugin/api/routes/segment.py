from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import openslide
from fastapi import APIRouter, HTTPException, Request

from histoseg_plugin.api.schemas import (
    GeoJSONFeatureCollection,
    TissueContoursRequest,
    WSISegmentationRequest,
    WSISegmentationResponse,
)
from histoseg_plugin.core.segmentation.geojson import logits_argmax_to_geojson
from histoseg_plugin.core.segmentation.segment import run_model_and_stitch_logits
from histoseg_plugin.core.tiling.tile import generate_tiles_from_tissue
from histoseg_plugin.core.tissue.geojson import contours_to_geojson_features
from histoseg_plugin.io.slide import assert_allowed_root, slide_uri_to_path
from histoseg_plugin.wsi_core.segmentation import segment_tissue

router = APIRouter(prefix="/segment", tags=["segmentation"])

ALLOWED_ROOTS = [Path("/mnt/nas6"), Path("/mnt/nas7")]  # TODO: move to settings


def resolve_and_check_slide(slide_uri: str) -> Path:
    slide_path = slide_uri_to_path(slide_uri)
    try:
        assert_allowed_root(slide_path, ALLOWED_ROOTS)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return slide_path


def normalize_seg_level(seg_level: int, level_count: int) -> int:
    if seg_level == -1:
        return level_count - 1
    if seg_level < 0 or seg_level >= level_count:
        raise HTTPException(
            status_code=400,
            detail=f"seg_level {seg_level} out of range for this slide",
        )
    return seg_level


@contextmanager
def open_wsi(path: Path):
    try:
        wsi = openslide.OpenSlide(str(path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open slide: {e}")
    try:
        yield wsi
    finally:
        wsi.close()


def run_tissue_segmentation(
    wsi: openslide.OpenSlide,
    req: TissueContoursRequest | WSISegmentationRequest,
) -> tuple[list[Any], list[Any], int]:
    seg_level = normalize_seg_level(req.seg_level, wsi.level_count)

    try:
        contours, holes = segment_tissue(
            wsi,
            seg_level=seg_level,
            sthresh=req.sthresh,
            sthresh_up=req.sthresh_up,
            mthresh=req.mthresh,
            close=req.close,
            use_otsu=req.use_otsu,
            filter_params=req.filter_params,
            ref_patch_size=req.ref_patch_size,
            exclude_ids=req.exclude_ids,
            keep_ids=req.keep_ids,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"segment_tissue failed: {e}")

    return contours, holes, seg_level


def build_tissue_geojson(
    *,
    contours: list[Any],
    holes: list[Any],
    seg_level: int,
    slide_uri: str,
    min_area_px_level0: int,
) -> GeoJSONFeatureCollection:
    props = {
        "class": "tissue",
        "seg_level_used": seg_level,
        "coords_space": "level0",
        "slide_uri": slide_uri,
        "algorithm": "segment_tissue_v1",
    }
    features = contours_to_geojson_features(
        contours=contours,
        holes=holes,
        downsample=1.0,
        props=props,
        min_area_px_level0=min_area_px_level0,
    )
    return GeoJSONFeatureCollection(features=features)


@router.post("/tissue", response_model=GeoJSONFeatureCollection)
def segment_tissue_route(req: TissueContoursRequest) -> GeoJSONFeatureCollection:
    slide_path = resolve_and_check_slide(req.slide_uri)

    with open_wsi(slide_path) as wsi:
        contours, holes, seg_level = run_tissue_segmentation(wsi=wsi, req=req)

    return build_tissue_geojson(
        contours=contours,
        holes=holes,
        seg_level=seg_level,
        slide_uri=req.slide_uri,
        min_area_px_level0=req.min_area_px_level0,
    )


@router.post("/wsi", response_model=WSISegmentationResponse)
def segment_wsi(
    req: WSISegmentationRequest,
    request: Request,
) -> WSISegmentationResponse:
    slide_path = resolve_and_check_slide(req.slide_uri)

    bundle = request.app.state.model_bundle
    model = bundle["model"]
    manifest = bundle["manifest"]
    device = request.app.state.device

    with open_wsi(slide_path) as wsi:
        # 1) Tissue segmentation
        contours, holes, seg_level = run_tissue_segmentation(wsi=wsi, req=req)

        tissue_geojson = build_tissue_geojson(
            contours=contours,
            holes=holes,
            seg_level=seg_level,
            slide_uri=req.slide_uri,
            min_area_px_level0=req.min_area_px_level0,
        )

        # 2) Tile generation
        coords = generate_tiles_from_tissue(
            wsi=wsi,
            contours_tissue=contours,
            holes_tissue=holes,
            tile_level=req.patch_level,
            tile_size=req.patch_size,
            step_size=req.step_size,
            contour_fn=req.contour_fn,
            center_shift=req.center_shift,
            use_padding=req.use_padding,
            top_left=req.top_left,
            bot_right=req.bot_right,
            max_workers=req.max_workers,
        )

    # 3) Inference + stitching
    stitch_result = run_model_and_stitch_logits(
        slide_path=str(slide_path.resolve()),
        coords=coords,
        tile_level=req.patch_level,
        tile_size=req.patch_size,
        model=model,
        device=device,
        output_target_mpp=req.output_target_mpp,
        batch_size=32,
        num_workers=32,
    )

    # 4) Logits -> GeoJSON
    outputs: dict[str, GeoJSONFeatureCollection] = {}
    for head_name, avg_logits in stitch_result.avg_logits_by_head.items():
        spec = manifest["output"][head_name]
        class_names = [label["name"] for label in spec["labels"]]
        background_id = spec.get("background_id", 0)

        geojson = logits_argmax_to_geojson(
            avg_logits=avg_logits,
            class_names=class_names,
            head_name=head_name,
            fx=stitch_result.meta.fx,
            fy=stitch_result.meta.fy,
            skip_class_ids=[background_id],
            simplify_epsilon=2.0,
            close_kernel=5,
            open_kernel=3,
            min_object_area=200,
            max_hole_area=200,
        )
        outputs[head_name] = GeoJSONFeatureCollection(**geojson)

    return WSISegmentationResponse(
        coords_space="level0",
        tissue=tissue_geojson,
        outputs=outputs,
    )
