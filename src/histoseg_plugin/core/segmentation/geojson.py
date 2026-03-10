from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import cv2
except ImportError as e:
    raise ImportError("This function requires opencv-python (cv2).") from e


def _remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area. binary is 0/255 uint8."""
    if min_area <= 0:
        return binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return binary
    out = np.zeros_like(binary)
    # stats: [label, x, y, w, h, area] (area is stats[i, cv2.CC_STAT_AREA])
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == i] = 255
    return out


def _fill_small_holes(binary: np.ndarray, max_hole_area: int) -> np.ndarray:
    """
    Fill holes smaller than max_hole_area inside a binary mask.
    binary is 0/255 uint8.
    """
    if max_hole_area <= 0:
        return binary

    # Find contours with hierarchy so we can identify holes.
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None or len(contours) == 0:
        return binary

    out = binary.copy()
    # hierarchy: (1, N, 4) -> [next, prev, first_child, parent]
    hier = hierarchy[0]
    for idx, h in enumerate(hier):
        parent = h[3]
        # In RETR_CCOMP, holes are contours that have a parent (i.e., inside an outer contour)
        if parent != -1:
            area = abs(cv2.contourArea(contours[idx]))
            if area <= float(max_hole_area):
                cv2.drawContours(out, contours, idx, 255, thickness=-1)  # fill hole
    return out


def logits_argmax_to_geojson(
    avg_logits: torch.Tensor,  # (C,H,W) float (CPU or GPU)
    class_names: Sequence[str],  # length C, index matches argmax id
    *,
    head_name: str = "a",
    fx: float = 1.0,  # level0 pixels per mask pixel in x
    fy: float = 1.0,  # level0 pixels per mask pixel in y
    include_classes: Optional[Sequence[int]] = None,  # e.g. [1] for tumor only
    skip_class_ids: Sequence[int] = (0,),  # commonly skip background
    close_kernel: int = 0,  # e.g. 5 or 7 to merge small gaps
    open_kernel: int = 0,  # e.g. 3 to remove salt noise
    min_object_area: int = 0,  # in mask pixels^2 (after argmax)
    max_hole_area: int = 0,  # fill holes smaller than this (mask px^2)
    simplify_epsilon: float = 0.0,  # in mask pixels; 1-3 can reduce vertices
    props_common: Optional[Dict[str, Any]] = None,  # extra properties in each feature
) -> Dict[str, Any]:
    """
    Converts stitched logits to GeoJSON polygons (with holes), scaling to level0 via fx,fy.

    Notes:
    - avg_logits is assumed to correspond to a downsampled "mask grid".
    - fx,fy map mask-grid coordinates -> level0 coordinates: x0 = x_mask * fx, y0 = y_mask * fy.
    """
    if avg_logits.ndim != 3:
        raise ValueError(
            f"Expected avg_logits (C,H,W), got shape {tuple(avg_logits.shape)}"
        )

    C, H, W = map(int, avg_logits.shape)
    if len(class_names) != C:
        raise ValueError(f"class_names length {len(class_names)} must match C={C}")

    # Argmax
    pred = torch.argmax(avg_logits, dim=0).to(torch.int32).cpu().numpy()  # (H,W)

    props_common = props_common or {}
    include_set = set(include_classes) if include_classes is not None else None
    skip_set = set(skip_class_ids)

    features: List[Dict[str, Any]] = []

    for cid in range(C):
        if cid in skip_set:
            continue
        if include_set is not None and cid not in include_set:
            continue

        mask = (pred == cid).astype(np.uint8) * 255  # uint8 0/255

        # Morphology to reduce fragmentation
        if close_kernel and close_kernel > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        if open_kernel and open_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        # Remove tiny islands
        mask = _remove_small_components(mask, min_object_area)

        # Optionally fill small holes (prevents tons of inner contours)
        mask = _fill_small_holes(mask, max_hole_area)

        if mask.max() == 0:
            continue

        # Extract contours with holes
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None or len(contours) == 0:
            continue

        hier = hierarchy[0]

        # Optional simplification in mask pixel space
        if simplify_epsilon and simplify_epsilon > 0:
            simplified = []
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                eps = float(simplify_epsilon)
                # If you want relative epsilon: eps = simplify_epsilon * peri
                simp = cv2.approxPolyDP(cnt, eps, True)
                simplified.append(simp)
            contours = simplified

        # Build polygons: outer contours (parent == -1), with holes (children)
        for i, h in enumerate(hier):
            parent = h[3]
            if parent != -1:
                continue  # not an outer contour

            outer = contours[i]
            if outer.shape[0] < 3:
                continue

            # Collect holes: children of this outer contour
            rings: List[List[List[float]]] = []

            def cnt_to_ring(cnt: np.ndarray) -> List[List[float]]:
                pts = cnt.reshape(-1, 2).astype(np.float64)
                # Scale to level0
                pts[:, 0] *= float(fx)
                pts[:, 1] *= float(fy)
                ring = pts.tolist()
                # Ensure closed ring
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                return ring

            rings.append(cnt_to_ring(outer))

            # iterate children chain
            child = h[2]
            while child != -1:
                hole = contours[child]
                if hole.shape[0] >= 3:
                    rings.append(cnt_to_ring(hole))
                child = hier[child][0]  # next sibling

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": rings,  # [outer, hole1, hole2, ...]
                },
                "properties": {
                    **props_common,
                    "class": class_names[cid],
                    "class_id": int(cid),
                    "head": head_name,
                    "coords_space": "level0",
                },
            }
            features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def logits_argmax_to_geojson_multipolygon(
    avg_logits: torch.Tensor,  # (C,H,W)
    class_names: Sequence[str],
    *,
    head_name: str = "a",
    fx: float = 1.0,
    fy: float = 1.0,
    include_classes: Optional[Sequence[int]] = None,
    skip_class_ids: Sequence[int] = (0,),
    close_kernel: int = 0,
    open_kernel: int = 0,
    min_object_area: int = 0,
    max_hole_area: int = 0,
    simplify_epsilon: float = 0.0,
    props_common: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert stitched logits (C,H,W) to a GeoJSON FeatureCollection.

    Output convention:
    - one Feature per class
    - geometry is:
        * Polygon if the class has exactly one connected component
        * MultiPolygon if the class has multiple disconnected components
    - holes are encoded as interior rings of each polygon

    Coordinates are scaled from mask space to level-0 slide space with:
        x_level0 = x_mask * fx
        y_level0 = y_mask * fy
    """
    if avg_logits.ndim != 3:
        raise ValueError(
            f"Expected avg_logits with shape (C,H,W), got {tuple(avg_logits.shape)}"
        )

    c, _, _ = map(int, avg_logits.shape)
    if len(class_names) != c:
        raise ValueError(
            f"class_names length ({len(class_names)}) must match number of channels ({c})"
        )

    pred = torch.argmax(avg_logits, dim=0).to(torch.int32).cpu().numpy()

    props_common = props_common or {}
    include_set = set(include_classes) if include_classes is not None else None
    skip_set = set(skip_class_ids)

    features: List[Dict[str, Any]] = []

    for class_id in range(c):
        if class_id in skip_set:
            continue
        if include_set is not None and class_id not in include_set:
            continue

        class_mask = _build_class_mask(
            pred=pred,
            class_id=class_id,
            close_kernel=close_kernel,
            open_kernel=open_kernel,
            min_object_area=min_object_area,
            max_hole_area=max_hole_area,
        )
        if class_mask is None or class_mask.max() == 0:
            continue

        polygons = _mask_to_polygons_with_holes(
            mask=class_mask,
            fx=fx,
            fy=fy,
            simplify_epsilon=simplify_epsilon,
        )
        if not polygons:
            continue

        geometry = _polygons_to_geojson_geometry(polygons)

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                **props_common,
                "class": class_names[class_id],
                "class_id": int(class_id),
                "head": head_name,
                "coords_space": "level0",
            },
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _build_class_mask(
    pred: np.ndarray,
    class_id: int,
    close_kernel: int,
    open_kernel: int,
    min_object_area: int,
    max_hole_area: int,
) -> np.ndarray:
    """Build and clean a binary uint8 mask (0/255) for one class."""
    mask = (pred == class_id).astype(np.uint8) * 255

    if close_kernel and close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel, close_kernel),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if open_kernel and open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel, open_kernel),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = _remove_small_components(mask, min_object_area)
    mask = _fill_small_holes(mask, max_hole_area)

    return mask


def _mask_to_polygons_with_holes(
    mask: np.ndarray,
    fx: float,
    fy: float,
    simplify_epsilon: float,
) -> List[List[List[List[float]]]]:
    """
    Convert a binary mask to a list of polygons with holes.

    Returns:
        polygons = [
            [outer_ring, hole1, hole2, ...],
            [outer_ring, hole1, ...],
            ...
        ]

    Each ring is a list of [x, y] coordinates in level-0 space.
    """
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None or len(contours) == 0:
        return []

    hierarchy = hierarchy[0]

    if simplify_epsilon and simplify_epsilon > 0:
        contours = [
            cv2.approxPolyDP(cnt, float(simplify_epsilon), True)
            for cnt in contours
        ]

    polygons: List[List[List[List[float]]]] = []

    for contour_idx, h in enumerate(hierarchy):
        parent_idx = h[3]
        if parent_idx != -1:
            continue  # only outer contours start a polygon

        outer = contours[contour_idx]
        if not _is_valid_contour(outer):
            continue

        rings: List[List[List[float]]] = [_contour_to_ring(outer, fx=fx, fy=fy)]

        child_idx = h[2]
        while child_idx != -1:
            hole = contours[child_idx]
            if _is_valid_contour(hole):
                rings.append(_contour_to_ring(hole, fx=fx, fy=fy))
            child_idx = hierarchy[child_idx][0]  # next sibling hole

        polygons.append(rings)

    return polygons


def _polygons_to_geojson_geometry(
    polygons: List[List[List[List[float]]]],
) -> Dict[str, Any]:
    """
    Convert a list of polygons-with-holes to GeoJSON geometry.

    Input:
        polygons = [
            [outer, hole1, hole2, ...],
            [outer, hole1, ...],
        ]
    """
    if len(polygons) == 1:
        return {
            "type": "Polygon",
            "coordinates": polygons[0],
        }

    return {
        "type": "MultiPolygon",
        "coordinates": polygons,
    }


def _is_valid_contour(contour: np.ndarray) -> bool:
    """A valid polygon contour needs at least 3 vertices."""
    return contour is not None and contour.shape[0] >= 3


def _contour_to_ring(
    contour: np.ndarray,
    *,
    fx: float,
    fy: float,
) -> List[List[float]]:
    """Convert an OpenCV contour to a closed GeoJSON ring in level-0 coordinates."""
    points = contour.reshape(-1, 2).astype(np.float64)
    points[:, 0] *= float(fx)
    points[:, 1] *= float(fy)

    ring: List[List[float]] = points.tolist()

    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])

    return ring