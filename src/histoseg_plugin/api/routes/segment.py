from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import openslide
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from histoseg_plugin.api.schemas import (
    GeoJSONFeatureCollection,
    TissueContoursRequest,
    WSISegmentationRequest,
    WSISegmentationResponse,
)
from histoseg_plugin.core.inference.bundle import InferenceBundle
from histoseg_plugin.core.inference.loader import load_inference_bundle
from histoseg_plugin.core.pipeline.wsi_segmentation import (
    run_wsi_segmentation,
    run_tissue_segmentation,
)
from histoseg_plugin.core.tissue.geojson import contours_to_geojson_features
from histoseg_plugin.settings import Settings, get_settings
from histoseg_plugin.api.adapters.segment import (
    build_wsi_segmentation_input,
    build_tissue_segmentation_input,
)

router = APIRouter(prefix="/segment", tags=["segmentation"])
logger = logging.getLogger(__name__)


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
def segment_tissue_route(
    req: Request, tissue_req: TissueContoursRequest
) -> GeoJSONFeatureCollection:
    wsi_segmentation_input = build_tissue_segmentation_input(
        req=tissue_req,
        allowed_roots=req.app.state.allowed_roots,
    )

    with open_wsi(wsi_segmentation_input.slide_path) as wsi:
        contours, holes, seg_level = run_tissue_segmentation(
            wsi=wsi, wsi_segmentation_input=wsi_segmentation_input
        )

    return build_tissue_geojson(
        contours=contours,
        holes=holes,
        seg_level=seg_level,
        slide_uri=tissue_req.slide_uri,
        min_area_px_level0=tissue_req.tissue.min_area_px_level0,
    )


def get_debug_inference_bundle(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> InferenceBundle:
    if not hasattr(request.app.state, "debug_inference_bundle"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_dir = settings.default_model_dir.resolve()

        request.app.state.debug_inference_bundle = load_inference_bundle(
            model_dir,
            device=device,
        )
        request.app.state.debug_device = device

    return request.app.state.debug_inference_bundle


@router.post("/wsi", response_model=WSISegmentationResponse)
def segment_wsi_route(
    wsi_req: WSISegmentationRequest,
    request: Request,
    inference_bundle: InferenceBundle = Depends(get_debug_inference_bundle),
) -> WSISegmentationResponse:

    try:
        result = run_wsi_segmentation(
            wsi_segmentation_input=build_wsi_segmentation_input(
                wsi_req, request.app.state.allowed_roots
            ),
            inference_bundle=inference_bundle,
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
