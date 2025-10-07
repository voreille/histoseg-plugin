# contour_proc.py
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# reuse your existing helpers (unchanged)
from histoseg_plugin.tiling.wsi_utils import save_hdf5
from histoseg_plugin.tiling.util_classes import (
    isInContourV1,
    isInContourV2,
    isInContourV3_Easy,
    isInContourV3_Hard,
    Contour_Checking_fn,
)

Coord = Tuple[int, int]
ROI = Optional[Tuple[int, int, int, int]]  # (x, y, w, h)

# ---------- Small helpers (top-level for mp pickling) ----------


def _is_in_holes(holes: List[np.ndarray], pt: Coord, patch_size: int) -> bool:
    """True if (patch center) lies inside any hole polygon."""
    cx = pt[0] + patch_size / 2.0
    cy = pt[1] + patch_size / 2.0
    for hole in holes or []:
        if cv2.pointPolygonTest(hole, (cx, cy), False) > 0:
            return True
    return False


def _is_in_contours(cont_check_fn, pt: Coord,
                    holes: Optional[List[np.ndarray]],
                    patch_size: int) -> bool:
    """Match original logic: inside foreground contour & not inside any hole."""
    if cont_check_fn(pt):
        if holes is not None:
            return not _is_in_holes(holes, pt, patch_size)
        return True
    return False


def _build_contour_check_fn(
    contour: np.ndarray,
    ref_patch_size_px: int,
    contour_fn: Union[str, Contour_Checking_fn] = "four_pt",
):
    """Construct the point-in-contour predicate with the same options as original code."""
    if isinstance(contour_fn, str):
        if contour_fn == "four_pt":
            return isInContourV3_Easy(contour=contour,
                                      patch_size=ref_patch_size_px,
                                      center_shift=0.5)
        elif contour_fn == "four_pt_hard":
            return isInContourV3_Hard(contour=contour,
                                      patch_size=ref_patch_size_px,
                                      center_shift=0.5)
        elif contour_fn == "center":
            return isInContourV2(contour=contour, patch_size=ref_patch_size_px)
        elif contour_fn == "basic":
            return isInContourV1(contour=contour)
        else:
            raise NotImplementedError(f"Unknown contour_fn: {contour_fn}")
    # already a Contour_Checking_fn
    assert isinstance(contour_fn, Contour_Checking_fn)
    return contour_fn


def _process_coord_candidate(
    coord: Coord,
    contour_holes: Optional[List[np.ndarray]],
    ref_patch_size_px: int,
    cont_check_fn,
) -> Optional[Coord]:
    """Top-level function for multiprocessing: returns coord if valid, else None."""
    return coord if _is_in_contours(cont_check_fn, coord, contour_holes,
                                    ref_patch_size_px) else None


# ---------- Public API ----------


def process_single_contour(
    contour: np.ndarray,
    contour_holes: Optional[List[np.ndarray]],
    level_dim0: Tuple[int, int],
    level_downsamples: List[Tuple[float, float]],
    patch_level: int,
    patch_size: int = 256,
    step_size: int = 256,
    contour_fn: Union[str, Contour_Checking_fn] = "four_pt",
    use_padding: bool = True,
    roi_top_left: Optional[Tuple[int, int]] = None,
    roi_bot_right: Optional[Tuple[int, int]] = None,
    max_workers: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """
    Compute valid top-left coordinates for patches within a single contour (and not in its holes),
    matching the original process_contour() behavior.

    Returns:
      asset_dict: {'coords': np.ndarray[N, 2]}
      attr_dict:  {'coords': {...attrs...}}
    """
    # ----- bounding box for this contour, clipped by ROI if provided -----
    start_x, start_y, w, h = cv2.boundingRect(contour)
    img_w, img_h = level_dim0

    patch_downsample = (
        int(level_downsamples[patch_level][0]),
        int(level_downsamples[patch_level][1]),
    )
    ref_patch_w = patch_size * patch_downsample[0]
    ref_patch_h = patch_size * patch_downsample[1]
    ref_patch_size_px = ref_patch_w  # square patches; original uses width

    if use_padding:
        stop_x = start_x + w
        stop_y = start_y + h
    else:
        stop_x = min(start_x + w, img_w - ref_patch_w + 1)
        stop_y = min(start_y + h, img_h - ref_patch_h + 1)

    # ROI crop
    if roi_bot_right is not None:
        stop_x = min(roi_bot_right[0], stop_x)
        stop_y = min(roi_bot_right[1], stop_y)
    if roi_top_left is not None:
        start_x = max(roi_top_left[0], start_x)
        start_y = max(roi_top_left[1], start_y)

    w_eff, h_eff = stop_x - start_x, stop_y - start_y
    if w_eff <= 0 or h_eff <= 0:
        # nothing to do for this contour in the given ROI
        return {}, {}

    # ----- grid of candidate coords at level-0 -----
    step_x = step_size * patch_downsample[0]
    step_y = step_size * patch_downsample[1]

    x_range = np.arange(start_x, stop_x, step=step_x, dtype=int)
    y_range = np.arange(start_y, stop_y, step=step_y, dtype=int)
    x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
    candidates = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)

    # ----- build the contour check function (same choices as original) -----
    cont_check_fn = _build_contour_check_fn(contour, ref_patch_size_px,
                                            contour_fn)

    # ----- filter candidates with multiprocessing -----
    if max_workers is None:
        max_workers = min(4, mp.cpu_count() or 1)

    with mp.Pool(max_workers) as pool:
        iterable = [(tuple(coord), contour_holes, ref_patch_size_px,
                     cont_check_fn) for coord in candidates]
        results = pool.starmap(_process_coord_candidate, iterable)

    coords = np.array([r for r in results if r is not None], dtype=np.int32)
    # Return empty if nothing found
    if coords.size == 0:
        return {}, {}

    # ----- package outputs like original -----
    asset_dict = {"coords": coords}
    attr = {
        "patch_size": patch_size,
        "patch_level": patch_level,
        "downsample": level_downsamples[patch_level],
        "downsampled_level_dim":
        tuple(level_dim0
              ),  # mirrors original (at patch_level they used scaled dims)
        "level_dim": (img_w, img_h),
        "save_path": "",  # left blank here; caller can fill if desired
    }
    attr_dict = {"coords": attr}
    return asset_dict, attr_dict


def process_contours_to_hdf5(
    contours_tissue: List[np.ndarray],
    holes_tissue: List[List[np.ndarray]],
    level_dim0: Tuple[int, int],
    level_downsamples: List[Tuple[float, float]],
    save_path_hdf5: Union[str, Path],
    patch_level: int = 0,
    patch_size: int = 256,
    step_size: int = 256,
    contour_fn: Union[str, Contour_Checking_fn] = "four_pt",
    use_padding: bool = True,
    roi_top_left: Optional[Tuple[int, int]] = None,
    roi_bot_right: Optional[Tuple[int, int]] = None,
    max_workers: Optional[int] = None,
    verbose: bool = True,
) -> Optional[Path]:
    """
    Loop over all contours, compute coordinates per contour, and append to a single HDF5 file,
    matching original process_contours() semantics.

    Returns:
      Path to the written HDF5 file, or None if nothing was written.
    """
    save_path_hdf5 = Path(save_path_hdf5)
    n_contours = len(contours_tissue)
    if verbose:
        print(f"Total number of contours to process: {n_contours}")
    if n_contours == 0:
        return None

    fp_chunk_size = max(1, math.ceil(n_contours * 0.05))
    wrote_any = False

    for idx, cont in enumerate(contours_tissue):
        if verbose and ((idx + 1) % fp_chunk_size == 0 or idx == 0):
            print(f"Processing contour {idx+1}/{n_contours}")

        asset_dict, attr_dict = process_single_contour(
            contour=cont,
            contour_holes=holes_tissue[idx]
            if idx < len(holes_tissue) else None,
            level_dim0=level_dim0,
            level_downsamples=level_downsamples,
            patch_level=patch_level,
            patch_size=patch_size,
            step_size=step_size,
            contour_fn=contour_fn,
            use_padding=use_padding,
            roi_top_left=roi_top_left,
            roi_bot_right=roi_bot_right,
            max_workers=max_workers,
        )

        if asset_dict:
            mode = "w" if not wrote_any else "a"
            save_hdf5(str(save_path_hdf5),
                      asset_dict,
                      attr_dict if mode == "w" else None,
                      mode=mode)
            wrote_any = True

    return save_path_hdf5 if wrote_any else None
