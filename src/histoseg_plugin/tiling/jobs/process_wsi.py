from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

from histoseg_plugin.tiling.parameter_models import Config, LevelPolicy
from histoseg_plugin.tiling.WholeSlideImage import WholeSlideImage
from histoseg_plugin.tiling.wsi_utils import StitchCoords

from .domain import TilingResult
from .exceptions import (
    LoadError,
    MaskSavingError,
    PatchError,
    SegmentationError,
    StitchError,
)


def _get_level_mpps(wsi_obj) -> list[float]:
    """
    Return MPP per level. If OpenSlide properties are missing, derive from downsample.
    """
    wsi = wsi_obj.getOpenSlide()
    props = wsi.properties
    base_mpp_x = props.get("openslide.mpp-x")
    base_mpp_y = props.get("openslide.mpp-y")
    base_mpp: Optional[float] = None
    if base_mpp_x and base_mpp_y:
        try:
            base_mpp = (float(base_mpp_x) + float(base_mpp_y)) / 2.0
        except Exception:
            base_mpp = None

    mpps: list[float] = []
    for lvl in range(wsi.level_count):
        ds = float(wsi.level_downsamples[lvl])
        if base_mpp is not None:
            mpps.append(base_mpp * ds)
        else:
            # Fallback: approximate via downsample, normalized to level 0 = 1.0
            mpps.append(ds)  # unitless; still useful for relative selection
    return mpps


def select_tile_level(
    wsi_obj,
    *,
    tile_level: int,
    target_tile_mpp: Optional[float],
    mpp_tolerance: float,
    level_policy: LevelPolicy,
) -> Tuple[int, float, bool, str]:
    """
    Returns: (chosen_level, chosen_mpp, within_tolerance, reason)
    """
    mpps = _get_level_mpps(wsi_obj)
    n = len(mpps)
    # If explicit level is given, use it and check tolerance (if target provided)
    if tile_level >= 0:
        lvl = min(max(tile_level, 0), n - 1)
        mpp = mpps[lvl]
        if target_tile_mpp is None:
            return lvl, mpp, True, "explicit level, no target mpp"
        lo = target_tile_mpp * (1 - mpp_tolerance)
        hi = target_tile_mpp * (1 + mpp_tolerance)
        ok = (lo <= mpp <= hi)
        return lvl, mpp, ok, ("within tolerance" if ok else
                              f"mpp {mpp:.3f} not in [{lo:.3f}, {hi:.3f}]")

    # Auto-select based on policy and target MPP
    if target_tile_mpp is None:
        # Underspecified; fall back to closest to a reasonable scale (e.g., level 0)
        lvl = 0
        mpp = mpps[lvl]
        return lvl, mpp, True, "auto fallback to level 0 (no target mpp)"

    # Find candidate levels
    diffs = [abs(m - target_tile_mpp) for m in mpps]
    closest = min(range(n), key=lambda i: diffs[i])

    if level_policy == "closest":
        lvl = closest
    elif level_policy == "lower":
        # choose level whose mpp <= target, closest on that side; fallback to closest
        lower_idxs = [i for i, m in enumerate(mpps) if m <= target_tile_mpp]
        lvl = min(lower_idxs, key=lambda i: abs(mpps[i] - target_tile_mpp)
                  ) if lower_idxs else closest
    elif level_policy == "higher":
        higher_idxs = [i for i, m in enumerate(mpps) if m >= target_tile_mpp]
        lvl = min(higher_idxs, key=lambda i: abs(mpps[i] - target_tile_mpp)
                  ) if higher_idxs else closest
    elif level_policy == "exact":
        # only valid if there exists a level within tolerance; else signal not ok
        lvl = closest
    else:
        lvl = closest  # defensive default

    mpp = mpps[lvl]
    lo = target_tile_mpp * (1 - mpp_tolerance)
    hi = target_tile_mpp * (1 + mpp_tolerance)
    ok = (lo <= mpp <= hi)
    reason = (
        "within tolerance" if ok else
        f"mpp {mpp:.3f} not in [{lo:.3f}, {hi:.3f}] (target {target_tile_mpp:.3f})"
    )
    if level_policy == "exact" and not ok:
        reason = "exact policy: no level within tolerance"
    return lvl, mpp, ok, reason


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
    tiling_params = config.tiling.model_dump()
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

    tile_level, tile_mpp, mpp_within_tolerance, mpp_reason = select_tile_level(
        wsi_obj,
        tile_level=tiling_params.get("tile_level", -1),
        target_tile_mpp=tiling_params.get("target_tile_mpp"),
        mpp_tolerance=tiling_params.get("mpp_tolerance", 0.1),
        level_policy=tiling_params.get("level_policy", "closest"),
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
        # mask_path=mask_path if generate_mask else None,
        # patch_path=patch_path if generate_patches else None,
        # stitch_path=stitch_path if generate_stitch else None,
    )
