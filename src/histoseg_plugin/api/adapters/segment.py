from pathlib import Path

from histoseg_plugin.api.schemas import TissueContoursRequest, WSISegmentationRequest
from histoseg_plugin.api.utils.paths import resolve_allowed_path
from histoseg_plugin.core.pipeline.contracts import (
    InferenceParams,
    TilingParams,
    TissueSegmentationParams,
    WSISegmentationInput,
)


def build_tissue_segmentation_input(
    req: TissueContoursRequest,
    allowed_roots: list[Path],
) -> WSISegmentationInput:
    slide_path = resolve_allowed_path(req.slide_uri, allowed_roots)

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
    )


def build_wsi_segmentation_input(
    req: WSISegmentationRequest,
    allowed_roots: list[Path],
) -> WSISegmentationInput:
    slide_path = resolve_allowed_path(req.slide_uri, allowed_roots)

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
