from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def _to_xy(contour: np.ndarray) -> np.ndarray:
    # Accept Nx2 or Nx1x2
    c = np.asarray(contour)
    if c.ndim == 3 and c.shape[1] == 1 and c.shape[2] == 2:
        c = c[:, 0, :]
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"Unexpected contour shape: {c.shape}")
    return c


def _ensure_closed(ring: np.ndarray) -> np.ndarray:
    if len(ring) == 0:
        return ring
    if not np.array_equal(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    return ring


def polygon_area_signed(xy: np.ndarray) -> float:
    # Shoelace; signed area
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))


def contours_to_geojson_features(
    contours: List[np.ndarray],
    holes: List[List[np.ndarray]],
    downsample: float,
    props: Dict[str, Any],
    min_area_px_level0: int = 0,
) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []

    for i, outer in enumerate(contours):
        outer_xy = _ensure_closed(_to_xy(outer).astype(np.float64) * downsample)

        if len(outer_xy) < 4:
            continue

        # Ensure outer ring is CCW (GeoJSON convention is not strict, but helps)
        if polygon_area_signed(outer_xy) < 0:
            outer_xy = outer_xy[::-1]

        # Filter by area in level-0 pixels (absolute area)
        area = abs(polygon_area_signed(outer_xy))
        if area < float(min_area_px_level0):
            continue

        rings = [outer_xy.tolist()]

        # Holes: ensure opposite winding (CW)
        for h in holes[i] if i < len(holes) else []:
            hole_xy = _ensure_closed(_to_xy(h).astype(np.float64) * downsample)
            if len(hole_xy) < 4:
                continue
            if polygon_area_signed(hole_xy) > 0:
                hole_xy = hole_xy[::-1]
            rings.append(hole_xy.tolist())

        features.append(
            {
                "type": "Feature",
                "properties": {**props, "region_index": i, "area_px_level0": area},
                "geometry": {"type": "Polygon", "coordinates": rings},
            }
        )

    return features
