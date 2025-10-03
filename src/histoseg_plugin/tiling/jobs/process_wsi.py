from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Union

from histoseg_plugin.tiling.parameter_models import Config
from histoseg_plugin.tiling.WholeSlideImage import WholeSlideImage
from histoseg_plugin.tiling.wsi_utils import StitchCoords

from .domain import TilingResult
from .exceptions import LoadError, PatchError, SegmentationError, StitchError, MaskSavingError


def _parse_ids(ids: str | List[int]) -> List[int]:
    if isinstance(ids, list):
        return [int(x) for x in ids]
    if ids is None:
        return []
    s = str(ids).strip().lower()
    if s in ("none", ""):
        return []
    return [int(x) for x in s.split(",") if str(x).strip() != ""]


def _resolve_levels(wsi_obj, seg_level: int, vis_level: int) -> Tuple[int, int]:
    def best_level() -> int:
        wsi = wsi_obj.getOpenSlide()
        return int(wsi.get_best_level_for_downsample(64))
    s_lvl = best_level() if seg_level < 0 else int(seg_level)
    v_lvl = best_level() if vis_level < 0 else int(vis_level)
    return s_lvl, v_lvl


def _too_large_for_seg(wsi_obj, seg_level: int, px_limit: float = 1e8) -> bool:
    w, h = wsi_obj.level_dim[seg_level]
    return (w * h) > px_limit


def process_single_wsi(
    wsi_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Config,
    *,
    generate_mask: bool = True,
    generate_patches: bool = True,
    generate_stitch: bool = True,
    verbose: bool = True,
) -> TilingResult:
    """
    Process one WSI and (optionally) produce: tissue mask, patch coordinates, and a stitched visualization.
    Fatal policy: any requested stage that fails raises a typed exception; the runner sets job status/error.

    Returns:
        TilingResult with timings and output paths (on success).
    """
    wsi_path = Path(wsi_path)
    output_dir = Path(output_dir)
    slide_id = wsi_path.stem

    # Output layout
    masks_dir    = output_dir / "masks"
    patches_dir  = output_dir / "patches"
    stitches_dir = output_dir / "stitches"
    for d in (masks_dir, patches_dir, stitches_dir):
        d.mkdir(parents=True, exist_ok=True)

    mask_path   = masks_dir / f"{slide_id}.jpg"
    patch_path  = patches_dir / f"{slide_id}.h5"
    stitch_path = stitches_dir / f"{slide_id}.jpg"

    if verbose:
        print(f"[{slide_id}] start")

    # Open WSI (fatal)
    try:
        wsi_obj = WholeSlideImage(str(wsi_path))
    except Exception as e:
        raise LoadError(f"Failed to load {wsi_path}: {e}") from e

    # Params from Config
    seg_params    = config.seg_params.model_dump()
    filter_params = config.filter_params.model_dump()
    vis_params    = config.vis_params.model_dump()
    patch_params  = config.patch_params.model_dump()

    # Parse keep/exclude ids
    seg_params["keep_ids"]    = _parse_ids(seg_params.get("keep_ids"))     # type: ignore[arg-type]
    seg_params["exclude_ids"] = _parse_ids(seg_params.get("exclude_ids"))  # type: ignore[arg-type]

    # Resolve levels
    seg_level, vis_level = _resolve_levels(
        wsi_obj,
        seg_params.get("seg_level", -1),
        vis_params.get("vis_level", -1),
    )
    seg_params["seg_level"] = seg_level
    vis_params["vis_level"] = vis_level

    # Sanity: segmentation size
    if _too_large_for_seg(wsi_obj, seg_level):
        raise SegmentationError(f"WSI too large for segmentation at level {seg_level}")

    seg_time = 0.0
    patch_time = 0.0
    stitch_time = 0.0

    # Segmentation (fatal if any downstream is requested)
    if generate_mask or generate_patches or generate_stitch:
        if verbose:
            print(f"[{slide_id}] segment tissue …")
        t0 = time.perf_counter()
        try:
            wsi_obj.segmentTissue(**seg_params, filter_params=filter_params)
        except Exception as e:
            raise SegmentationError(f"Segmentation failed: {e}") from e
        finally:
            seg_time = time.perf_counter() - t0

    # Mask (fatal if requested)
    if generate_mask:
        if verbose:
            print(f"[{slide_id}] render mask …")
        try:
            pil_img = wsi_obj.visWSI(**vis_params)
            pil_img.save(str(mask_path))
        except Exception as e:
            raise MaskSavingError(f"Mask saving failed: {e}") from e

    # Patches (fatal if requested)
    if generate_patches:
        if verbose:
            print(f"[{slide_id}] extract patch coords …")
        t1 = time.perf_counter()
        try:
            _ = wsi_obj.process_contours(save_path=str(patches_dir), **patch_params)
        except Exception as e:
            raise PatchError(f"Patch extraction failed: {e}") from e
        finally:
            patch_time = time.perf_counter() - t1

    # Stitch (fatal if requested; if generate_patches was True, we assume the file now exists)
    if generate_stitch:
        if verbose:
            print(f"[{slide_id}] stitch heatmap …")
        t2 = time.perf_counter()
        try:
            heatmap = StitchCoords(
                str(patch_path),
                wsi_obj,
                downscale=64,
                bg_color=(0, 0, 0),
                alpha=-1,
                draw_grid=False,
            )
            heatmap.save(str(stitch_path))
        except Exception as e:
            raise StitchError(f"Stitching failed: {e}") from e
        finally:
            stitch_time = time.perf_counter() - t2

    if verbose:
        print(f"[{slide_id}] done | seg={seg_time:.2f}s patch={patch_time:.2f}s stitch={stitch_time:.2f}s")

    return TilingResult(
        seg_time=seg_time,
        patch_time=patch_time,
        stitch_time=stitch_time,
        # mask_path=mask_path if generate_mask else None,
        # patch_path=patch_path if generate_patches else None,
        # stitch_path=stitch_path if generate_stitch else None,
    )
