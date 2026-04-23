from __future__ import annotations

import logging
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
from histoseg_plugin.core.inference.bundle import InferenceBundle
from histoseg_plugin.core.tissue.geojson import contours_to_geojson_features
from histoseg_plugin.io.slide import assert_allowed_root, slide_uri_to_path
from histoseg_plugin.core.tissue.segmentation import segment_tissue
from histoseg_plugin.core.pipeline.wsi_segmentation import run_wsi_segmentation
from histoseg_plugin.core.pipeline.contracts import (
    WSISegmentationInput,
    TissueSegmentationParams,
    TilingParams,
    InferenceParams,
)

router = APIRouter(prefix="/segment", tags=["segmentation"])
logger = logging.getLogger(__name__)

ALLOWED_ROOTS = [
    Path("/mnt/nas6"),
    Path("/mnt/nas7"),
    Path("/home/val/data"),
]  # TODO: move to settings


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
    seg_level = normalize_seg_level(req.tissue.seg_level, wsi.level_count)

    try:
        contours, holes = segment_tissue(
            wsi,
            seg_level=seg_level,
            sthresh=req.tissue.sthresh,
            sthresh_up=req.tissue.sthresh_up,
            mthresh=req.tissue.mthresh,
            close=req.tissue.close,
            use_otsu=req.tissue.use_otsu,
            filter_params=req.tissue.filter_params,
            ref_patch_size=req.tissue.ref_patch_size,
            exclude_ids=req.tissue.exclude_ids,
            keep_ids=req.tissue.keep_ids,
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
        min_area_px_level0=req.tissue.min_area_px_level0,
    )


def build_wsi_segmentation_input(req: WSISegmentationRequest) -> WSISegmentationInput:
    slide_path = resolve_and_check_slide(req.slide_uri)

    return WSISegmentationInput(
        slide_path=slide_path,
        tissue=TissueSegmentationParams(
            seg_level=req.tissue.seg_level,
            sthresh=req.tissue.sthresh,
            sthresh_up=req.tissue.sthresh_up,
            mthresh=req.tissue.mthresh,
            close=req.tissue.close,
            use_otsu=req.tissue.use_otsu,
            filter_params=dict(req.tissue.filter_params),
            ref_patch_size=req.tissue.ref_patch_size,
            exclude_ids=list(req.tissue.exclude_ids),
            keep_ids=list(req.tissue.keep_ids),
            min_area_px_level0=req.tissue.min_area_px_level0,
        ),
        tiling=TilingParams(
            contour_fn=req.tiling.contour_fn,
            center_shift=req.tiling.center_shift,
            use_padding=req.tiling.use_padding,
            top_left=tuple(req.tiling.top_left) if req.tiling.top_left else None,
            bot_right=tuple(req.tiling.bot_right) if req.tiling.bot_right else None,
            max_workers=req.tiling.max_workers,
        ),
        inference=InferenceParams(
            output_target_mpp=req.inference.output_target_mpp,
            batch_size=req.inference.batch_size,
            num_workers=req.inference.num_workers,
        ),
    )


@router.post("/wsi", response_model=WSISegmentationResponse)
def segment_wsi_route(
    req: WSISegmentationRequest,
    request: Request,
) -> WSISegmentationResponse:

    bundle: InferenceBundle = request.app.state.inference_bundle
    try:
        result = run_wsi_segmentation(
            wsi_segmentation_input=build_wsi_segmentation_input(req),
            inference_bundle=bundle,
        )
    except Exception as e:
        logger.error(f"Error during WSI segmentation: {e}")
        raise HTTPException(
            status_code=500, detail=f"WSI segmentation failed: {e}"
        ) from e

    return WSISegmentationResponse(
        coords_space=result.coords_space,
        tissue=result.tissue,
        outputs=result.outputs,
        statistics=result.statistics,
    )
