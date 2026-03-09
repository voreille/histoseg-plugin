from typing import Optional, Tuple
import numpy as np
import openslide

from histoseg_plugin.tiling.contours_processing import process_contour


def generate_tiles_from_tissue(
    wsi: openslide.OpenSlide,
    contours_tissue,
    holes_tissue,
    *,
    tile_level: int,
    tile_size: int,
    step_size: int,
    contour_fn: str = "four_pt",
    center_shift: float = 0.5,
    use_padding: bool = True,
    top_left: Optional[Tuple[int, int]] = None,
    bot_right: Optional[Tuple[int, int]] = None,
    max_workers: int = 4,
)-> np.ndarray:
    """
    Generate tile coordinates inside tissue contours.

    Returns
    -------
    coords : np.ndarray
        shape (N,2) level0 tile origins
    attrs : dict
        metadata required for tile reading
    """

    if len(contours_tissue) == 0:
        return np.empty((0, 2), dtype=np.int32)

    coords_all = []

    for idx, cont in enumerate(contours_tissue):
        coords = process_contour(
            wsi,
            cont,
            holes_tissue[idx],
            patch_level=tile_level,
            patch_size=tile_size,
            step_size=step_size,
            contour_fn=contour_fn,
            center_shift=center_shift,
            use_padding=use_padding,
            top_left=top_left,
            bot_right=bot_right,
            max_workers=max_workers,
        )

        coords = np.asarray(coords, dtype=np.int32).reshape(-1, 2)

        if coords.size == 0:
            continue

        coords_all.append(coords)

    if not coords_all:
        coords = np.empty((0, 2), dtype=np.int32)
    else:
        coords = np.concatenate(coords_all, axis=0)
        coords = np.unique(coords, axis=0)

    return coords
