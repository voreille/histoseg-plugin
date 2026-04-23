from pathlib import Path

import logging
from typing import Any

import openslide
import torch

from histoseg_plugin.core.tissue.geojson import contours_to_geojson_features

from ..geojson.schemas import GeoJSONFeatureCollection
from ..inference.bundle import InferenceBundle
from ..inference.planning import resolve_wsi_inference_plan
from ..postprocessing.schemas import DemoPatternStatistics
from ..postprocessing.stats import compute_demo_pattern_statistics
from ..segmentation.segment import (
    StitchResult,
    run_model_and_stitch_logits,
)
from ..tiling.tile import generate_tiles_from_tissue
from ..tissue.segmentation import segment_tissue
from ..wsi.utils import open_wsi
from .contracts import WSISegmentationInput, WSISegmentationResult
from ..segmentation.geojson import logits_argmax_to_geojson_multipolygon

logger = logging.getLogger(__name__)


def build_tissue_geojson(
    *,
    contours: list[Any],
    holes: list[Any],
    seg_level: int,
    slide_path: Path,
    min_area_px_level0: int,
) -> GeoJSONFeatureCollection:
    props = {
        "class": "tissue",
        "seg_level_used": seg_level,
        "coords_space": "level0",
        "slide_path": str(slide_path),
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


def maybe_compute_demo_statistics(
    *,
    pixel_area_um2: float,
    stitched_outputs: dict[str, torch.Tensor],
    processed_metadata: dict[str, Any] | None,
    manifest: Any,
    covered_mask: torch.Tensor | None,
) -> DemoPatternStatistics | None:
    required_keys = {
        "logits_a",
        "logits_b",
        "logits_b_conformal.safe",
        "logits_b_conformal.max_possible",
    }

    if not required_keys.issubset(stitched_outputs.keys()):
        return None

    metadata = dict(processed_metadata or {})
    if covered_mask is not None:
        metadata["logits_a.covered_mask"] = covered_mask
        metadata["logits_b.covered_mask"] = covered_mask

    return compute_demo_pattern_statistics(
        outputs=stitched_outputs,
        metadata=metadata,
        head_a_labels=manifest.output["logits_a"].labels,
        head_b_labels=manifest.output["logits_b"].labels,
        head_a_name="logits_a",
        head_b_name="logits_b",
        head_b_max_name="logits_b_conformal.max_possible",
        covered_mask_a_name="logits_a.covered_mask",
        covered_mask_b_name="logits_b.covered_mask",
        pixel_area_um2=pixel_area_um2,
    )


def normalize_seg_level(seg_level: int, level_count: int) -> int:
    if seg_level == -1:
        return level_count - 1
    if seg_level < 0 or seg_level >= level_count:
        raise ValueError(f"seg_level {seg_level} out of range for this slide")
    return seg_level


def run_tissue_segmentation(
    wsi: openslide.OpenSlide,
    wsi_segmentation_input: WSISegmentationInput,
) -> tuple[list[Any], list[Any], int]:
    seg_level = normalize_seg_level(wsi_segmentation_input.tissue.seg_level, wsi.level_count)

    try:
        contours, holes = segment_tissue(
            wsi,
            seg_level=seg_level,
            sthresh=wsi_segmentation_input.tissue.sthresh,
            sthresh_up=wsi_segmentation_input.tissue.sthresh_up,
            mthresh=wsi_segmentation_input.tissue.mthresh,
            close=wsi_segmentation_input.tissue.close,
            use_otsu=wsi_segmentation_input.tissue.use_otsu,
            filter_params=wsi_segmentation_input.tissue.filter_params,
            ref_patch_size=wsi_segmentation_input.tissue.ref_patch_size,
            exclude_ids=wsi_segmentation_input.tissue.exclude_ids,
            keep_ids=wsi_segmentation_input.tissue.keep_ids,
        )
    except Exception as e:
        logger.error(f"Error during tissue segmentation: {e}")
        raise RuntimeError(f"Tissue segmentation failed: {e}") from e
    return contours, holes, seg_level


def run_wsi_segmentation(
    *,
    wsi_segmentation_input: WSISegmentationInput,
    inference_bundle: InferenceBundle,
) -> WSISegmentationResult:
    slide_path = wsi_segmentation_input.slide_path

    model_runner = inference_bundle.model_runner
    postprocessor = inference_bundle.postprocessor
    manifest = model_runner.manifest

    with open_wsi(slide_path) as wsi:
        logger.info(
            "Running tissue segmentation for slide %s",
            wsi_segmentation_input.slide_path,
        )
        plan = resolve_wsi_inference_plan(
            wsi=wsi,
            model_tile_size=manifest.input.tile_size,
            model_tile_mpp=manifest.input.tile_mpp,
            mpp_tolerance=manifest.input.mpp_tolerance or 0.10,
            overlap_ratio=0.5,
        )

        contours, holes, seg_level = run_tissue_segmentation(
            wsi=wsi, wsi_segmentation_input=wsi_segmentation_input
        )

        logger.info(
            "Found %d tissue contours and %d holes at seg_level %d",
            len(contours),
            len(holes),
            seg_level,
        )

        tissue_geojson = build_tissue_geojson(
            contours=contours,
            holes=holes,
            seg_level=seg_level,
            slide_path=wsi_segmentation_input.slide_path,
            min_area_px_level0=wsi_segmentation_input.tissue.min_area_px_level0,
        )

        logger.info(
            "Generating tiles from tissue contours for slide %s",
            wsi_segmentation_input.slide_path,
        )
        coords = generate_tiles_from_tissue(
            wsi=wsi,
            contours_tissue=contours,
            holes_tissue=holes,
            tile_level=plan.chosen_level,
            tile_size=plan.level_tile_size,
            step_size=plan.level_stride,
            contour_fn=wsi_segmentation_input.tiling.contour_fn,
            center_shift=wsi_segmentation_input.tiling.center_shift,
            use_padding=wsi_segmentation_input.tiling.use_padding,
            top_left=wsi_segmentation_input.tiling.top_left,
            bot_right=wsi_segmentation_input.tiling.bot_right,
            max_workers=wsi_segmentation_input.tiling.max_workers,
        )
        logger.info(
            "Generated %d tile coordinates for slide %s",
            len(coords),
            wsi_segmentation_input.slide_path,
        )

    logger.info(
        "Running model inference and stitching logits for slide %s",
        wsi_segmentation_input.slide_path,
    )
    stitch_result: StitchResult = run_model_and_stitch_logits(
        slide_path=str(slide_path.resolve()),
        coords=coords,
        tile_level=plan.chosen_level,
        level_tile_size=plan.level_tile_size,
        model_runner=model_runner,
        output_target_mpp=wsi_segmentation_input.inference.output_target_mpp,
        batch_size=wsi_segmentation_input.inference.batch_size,
        num_workers=wsi_segmentation_input.inference.num_workers,
        resample=plan.needs_resampling,
        model_tile_size=plan.model_tile_size,
    )

    stitched_outputs = dict(stitch_result.avg_logits_by_head)
    processed_metadata: dict[str, Any] = {}

    covered_mask = None
    if stitch_result.weight_map is not None:
        covered_mask = stitch_result.weight_map > 0

    if postprocessor is not None:
        processed = postprocessor(
            stitched_outputs=stitched_outputs,
            aux={f"{k}.covered_mask": covered_mask for k in stitched_outputs.keys()}
            if covered_mask is not None
            else None,
        )
        stitched_outputs.update(processed.outputs)
        processed_metadata.update(processed.metadata)

    statistics = maybe_compute_demo_statistics(
        stitched_outputs=stitched_outputs,
        processed_metadata=processed_metadata,
        manifest=manifest,
        covered_mask=covered_mask,
        pixel_area_um2=stitch_result.meta.output_target_mpp**2,
    )

    logger.info(
        "Converting outputs to GeoJSON for slide %s", wsi_segmentation_input.slide_path
    )
    outputs: dict[str, GeoJSONFeatureCollection] = {}

    for head_name, avg_logits in stitch_result.avg_logits_by_head.items():
        spec = manifest.output[head_name]
        class_names = [label.name for label in spec.labels]
        background_id = spec.background_id

        logger.info(
            "Converting to GeoJSON for head '%s' with %d classes (background_id=%s) for slide %s",
            head_name,
            len(class_names),
            background_id,
            wsi_segmentation_input.slide_path,
        )

        geojson = logits_argmax_to_geojson_multipolygon(
            avg_logits=avg_logits,
            class_names=class_names,
            head_name=head_name,
            fx=stitch_result.meta.fx,
            fy=stitch_result.meta.fy,
            skip_class_ids=[background_id] if background_id is not None else [],
            simplify_epsilon=2.0,
            close_kernel=5,
            open_kernel=3,
            min_object_area=200,
            max_hole_area=200,
            props_common={
                "head_display_name": spec.display_name or head_name,
                "object_name": spec.display_name or head_name,
            },  # TODO: if display name becomes central, make it more explicit than relying on props_common
        )
        outputs[head_name] = GeoJSONFeatureCollection(**geojson)

    logger.info(
        "Completed processing all heads for slide %s",
        wsi_segmentation_input.slide_path,
    )

    return WSISegmentationResult(
        coords_space="level0",
        tissue=tissue_geojson,
        outputs=outputs,
        statistics=statistics,
    )
