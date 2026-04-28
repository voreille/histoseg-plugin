from pathlib import Path

from histoseg_plugin.api.schemas import JobItem
from histoseg_plugin.api.utils.paths import resolve_allowed_path
from histoseg_plugin.core.pipeline.contracts import (
    InferenceParams,
    TilingParams,
    TissueSegmentationParams,
    WSISegmentationInput,
)


def build_job_item_input(
    item: JobItem,
    allowed_roots: list[Path],
) -> dict:
    slide_path = resolve_allowed_path(item.slide_uri, allowed_roots)

    return WSISegmentationInput(
        model_id=item.model_id,
        slide_path=slide_path,
        tissue=TissueSegmentationParams(
            seg_level=item.tissue.seg_level,
            sthresh=item.tissue.sthresh,
            sthresh_up=item.tissue.sthresh_up,
            mthresh=item.tissue.mthresh,
            close=item.tissue.close,
            use_otsu=item.tissue.use_otsu,
            filter_params=dict(item.tissue.filter_params),
            ref_patch_size=item.tissue.ref_patch_size,
            exclude_ids=list(item.tissue.exclude_ids),
            keep_ids=list(item.tissue.keep_ids),
            min_area_px_level0=item.tissue.min_area_px_level0,
        ),
        tiling=TilingParams(
            contour_fn=item.tiling.contour_fn,
            center_shift=item.tiling.center_shift,
            use_padding=item.tiling.use_padding,
            top_left=tuple(item.tiling.top_left) if item.tiling.top_left else None,
            bot_right=tuple(item.tiling.bot_right) if item.tiling.bot_right else None,
            max_workers=item.tiling.max_workers,
        ),
        inference=InferenceParams(
            output_target_mpp=item.inference.output_target_mpp,
            batch_size=item.inference.batch_size,
            num_workers=item.inference.num_workers,
        ),
    ).as_dict()


def build_job_inputs(
    items: list[JobItem],
    allowed_roots: list[Path],
) -> list[dict]:
    return [build_job_item_input(item, allowed_roots) for item in items]
