from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager

import openslide
from fastapi import APIRouter, HTTPException

from histoseg_plugin.api.schemas import GeoJSONFeatureCollection, TissueContoursRequest
from histoseg_plugin.core.tissue.geojson import contours_to_geojson_features
from histoseg_plugin.io.slide import assert_allowed_root, slide_uri_to_path
from histoseg_plugin.wsi_core.segmentation import segment_tissue

router = APIRouter(prefix="/tissue", tags=["tissue"])

ALLOWED_ROOTS = [Path("/mnt/nas6"), Path("/mnt/nas7")]  # TODO: settings/env


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


def normalize_seg_level(seg_level: int, level_count: int) -> int:
    if seg_level == -1:
        return level_count - 1
    if seg_level < 0 or seg_level >= level_count:
        raise HTTPException(
            status_code=400,
            detail=f"seg_level {seg_level} out of range for this slide",
        )
    return seg_level


@router.post("/contours", response_model=GeoJSONFeatureCollection)
def tissue_contours(req: TissueContoursRequest) -> GeoJSONFeatureCollection:
    slide_path = slide_uri_to_path(req.slide_uri)

    try:
        assert_allowed_root(slide_path, ALLOWED_ROOTS)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    with open_wsi(slide_path) as wsi:
        seg_level = normalize_seg_level(req.seg_level, wsi.level_count)

        # If you want: only catch expected errors here; otherwise let it raise
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

    # TODO: add more properties like the params for tissue segmentation
    props = {
        "class": "tissue",
        "seg_level_used": seg_level,
        "coords_space": "level0",
        "slide_uri": req.slide_uri,
        "algorithm": "segment_tissue_v1",
    }
    features = contours_to_geojson_features(
        contours=contours,
        holes=holes,
        downsample=1.0,
        props=props,
        min_area_px_level0=req.min_area_px_level0,
    )
    return GeoJSONFeatureCollection(features=features)
