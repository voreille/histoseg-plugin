from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..parameter_models import Config, LevelPolicy
from ..WholeSlideImage import WholeSlideImage
from ..wsi_utils import StitchCoords
from .domain import TilingResult
from .exceptions import (
    LoadError,
    MaskSavingError,
    PatchError,
    SegmentationError,
    StitchError,
)


def _get_level_mpps(wsi_obj) -> List[float]:
    """
    Return MPP per level. If OpenSlide properties are missing, derive from downsample.
    """
    wsi = wsi_obj.getOpenSlide()
    props = wsi.properties
    base = None
    try:
        bx = float(props.get("openslide.mpp-x", ""))
        by = float(props.get("openslide.mpp-y", ""))
        base = (bx + by) / 2.0
    except Exception:
        base = None

    mpps: List[float] = []
    for lvl in range(wsi.level_count):
        ds = float(wsi.level_downsamples[lvl])
        mpps.append(base * ds if base is not None else ds)
    return mpps


def _bounds(target_mpp: float, tol: float) -> Tuple[float, float]:
    lo = target_mpp * (1 - tol)
    hi = target_mpp * (1 + tol)
    return lo, hi


def select_tile_level_fixed(
    wsi_obj,
    *,
    tile_level: int,
    target_tile_mpp: Optional[float] = None,
    mpp_tolerance: float = 0.10,
) -> Tuple[int, float, bool, str]:
    """
    Fixed mode: clamp level into available range; if target MPP is provided, check tolerance.
    Returns: (level, mpp, within_tolerance, reason)
    """
    mpps = _get_level_mpps(wsi_obj)
    if not mpps:
        raise ValueError("No pyramid levels.")
    lvl = max(0, min(tile_level, len(mpps) - 1))
    mpp = mpps[lvl]
    if target_tile_mpp is None:
        return lvl, mpp, True, "fixed level; no target MPP"
    lo, hi = _bounds(target_tile_mpp, mpp_tolerance)
    ok = (lo <= mpp <= hi)
    reason = "within tolerance" if ok else f"mpp {mpp:.3f} not in [{lo:.3f}, {hi:.3f}]"
    return lvl, mpp, ok, reason


def select_tile_level_auto(
    wsi_obj,
    *,
    target_tile_mpp: float,
    mpp_tolerance: float,
    level_policy: LevelPolicy,
) -> Tuple[int, float, bool, str]:
    """
    Auto mode: choose level according to policy and target MPP; check tolerance.
    Returns: (level, mpp, within_tolerance, reason)
    """
    mpps = _get_level_mpps(wsi_obj)
    if not mpps:
        raise ValueError("No pyramid levels.")

    # find closest candidate(s)
    diffs = [abs(m - target_tile_mpp) for m in mpps]
    closest = min(range(len(mpps)), key=lambda i: diffs[i])

    if level_policy == "closest":
        lvl = closest
    elif level_policy == "lower":
        lowers = [i for i, m in enumerate(mpps) if m <= target_tile_mpp]
        lvl = min(lowers, key=lambda i: abs(mpps[i] - target_tile_mpp)
                  ) if lowers else closest
    else:  # "higher"
        highers = [i for i, m in enumerate(mpps) if m >= target_tile_mpp]
        lvl = min(highers, key=lambda i: abs(mpps[i] - target_tile_mpp)
                  ) if highers else closest

    mpp = mpps[lvl]
    lo, hi = _bounds(target_tile_mpp, mpp_tolerance)
    ok = (lo <= mpp <= hi)
    reason = "within tolerance" if ok else f"mpp {mpp:.3f} not in [{lo:.3f}, {hi:.3f}] (target {target_tile_mpp:.3f})"
    return lvl, mpp, ok, reason


def select_tile_level_from_config(
        wsi_obj,
        tiling,  # your Tiling model (with level_mode)
) -> Tuple[int, float, bool, str]:
    """
    Dispatcher that reads tiling.level_mode and calls the right selector.
    """
    if tiling.level_mode == "fixed":
        return select_tile_level_fixed(
            wsi_obj,
            tile_level=tiling.tile_level,
            target_tile_mpp=tiling.target_tile_mpp,  # optional QA check
            mpp_tolerance=tiling.mpp_tolerance,
        )
    # auto
    if tiling.target_tile_mpp is None:
        raise ValueError("Auto mode requires target_tile_mpp.")
    return select_tile_level_auto(
        wsi_obj,
        target_tile_mpp=tiling.target_tile_mpp,
        mpp_tolerance=tiling.mpp_tolerance,
        level_policy=tiling.level_policy,
    )


def _parse_ids(ids: str | List[int]) -> List[int]:
    if isinstance(ids, list):
        return [int(x) for x in ids]
    if ids is None:
        return []
    s = str(ids).strip().lower()
    if s in ("none", ""):
        return []
    return [int(x) for x in s.split(",") if str(x).strip() != ""]


def _resolve_levels(wsi_obj, seg_level: int,
                    vis_level: int) -> Tuple[int, int]:

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
    masks_dir = output_dir / "masks"
    patches_dir = output_dir / "patches"
    stitches_dir = output_dir / "stitches"
    for d in (masks_dir, patches_dir, stitches_dir):
        d.mkdir(parents=True, exist_ok=True)

    mask_path = masks_dir / f"{slide_id}.jpg"
    patch_path = patches_dir / f"{slide_id}.h5"
    stitch_path = stitches_dir / f"{slide_id}.jpg"

    if verbose:
        print(f"[{slide_id}] start")

    # Open WSI (fatal)
    try:
        wsi_obj = WholeSlideImage(str(wsi_path))
    except Exception as e:
        raise LoadError(f"Failed to load {wsi_path}: {e}") from e

    # Params from Config
    seg_params = config.seg_params.model_dump()
    filter_params = config.filter_params.model_dump()
    vis_params = config.vis_params.model_dump()
    patch_params = config.patch_params.model_dump()

    # Parse keep/exclude ids
    seg_params["keep_ids"] = _parse_ids(
        seg_params.get("keep_ids"))  # type: ignore[arg-type]
    seg_params["exclude_ids"] = _parse_ids(
        seg_params.get("exclude_ids"))  # type: ignore[arg-type]

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
        raise SegmentationError(
            f"WSI too large for segmentation at level {seg_level}")

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

    tile_level, tile_mpp, mpp_within_tolerance, mpp_reason = select_tile_level_from_config(
        wsi_obj,
        tiling=config.tiling,
    )

    # Patches (fatal if requested)
    if generate_patches:
        if verbose:
            print(f"[{slide_id}] extract patch coords …")
        t1 = time.perf_counter()
        try:
            _ = wsi_obj.process_contours(save_path=str(patches_dir),
                                         patch_level=tile_level,
                                         **patch_params)
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
        print(
            f"[{slide_id}] done | seg={seg_time:.2f}s patch={patch_time:.2f}s stitch={stitch_time:.2f}s"
        )

    return TilingResult(
        seg_time=seg_time,
        patch_time=patch_time,
        stitch_time=stitch_time,
        tile_level=tile_level,
        tile_mpp=tile_mpp,
        mpp_within_tolerance=mpp_within_tolerance,
        mpp_reason=mpp_reason,
    )
