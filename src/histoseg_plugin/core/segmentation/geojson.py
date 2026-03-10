from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import cv2
except ImportError as e:
    raise ImportError("This function requires opencv-python (cv2).") from e

logger = logging.getLogger(__name__)


def _remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area. binary is 0/255 uint8."""
    if min_area <= 0:
        return binary

    t0 = time.perf_counter()

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8),
        connectivity=8,
    )
    if n <= 1:
        logger.debug(
            "_remove_small_components: no foreground components | dt=%.3fs",
            time.perf_counter() - t0,
        )
        return binary

    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= int(min_area)
    keep[0] = False  # background

    out = (keep[labels] * 255).astype(np.uint8)

    logger.debug(
        "_remove_small_components: total=%d kept=%d removed=%d min_area=%d dt=%.3fs",
        n - 1,
        int(keep[1:].sum()),
        int((n - 1) - keep[1:].sum()),
        min_area,
        time.perf_counter() - t0,
    )
    return out


def _fill_small_holes(binary: np.ndarray, max_hole_area: int) -> np.ndarray:
    """
    Fill holes smaller than max_hole_area inside a binary mask.
    binary is 0/255 uint8.
    """
    if max_hole_area <= 0:
        return binary

    t0 = time.perf_counter()

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None or len(contours) == 0:
        logger.debug(
            "_fill_small_holes: no contours | dt=%.3fs",
            time.perf_counter() - t0,
        )
        return binary

    out = binary.copy()
    hier = hierarchy[0]
    filled = 0

    for idx, h in enumerate(hier):
        parent = h[3]
        if parent != -1:
            area = abs(cv2.contourArea(contours[idx]))
            if area <= float(max_hole_area):
                cv2.drawContours(out, contours, idx, 255, thickness=-1)
                filled += 1

    logger.debug(
        "_fill_small_holes: filled=%d max_hole_area=%d dt=%.3fs",
        filled,
        max_hole_area,
        time.perf_counter() - t0,
    )
    return out


def logits_argmax_to_geojson(
    avg_logits: torch.Tensor,  # (C,H,W) float (CPU or GPU)
    class_names: Sequence[str],  # length C, index matches argmax id
    *,
    head_name: str = "a",
    fx: float = 1.0,  # level0 pixels per mask pixel in x
    fy: float = 1.0,  # level0 pixels per mask pixel in y
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
    Backward-compatible wrapper returning one Feature per polygon.
    Internally uses the same faster class-wise cropped pipeline.
    """
    geojson = logits_argmax_to_geojson_multipolygon(
        avg_logits=avg_logits,
        class_names=class_names,
        head_name=head_name,
        fx=fx,
        fy=fy,
        include_classes=include_classes,
        skip_class_ids=skip_class_ids,
        close_kernel=close_kernel,
        open_kernel=open_kernel,
        min_object_area=min_object_area,
        max_hole_area=max_hole_area,
        simplify_epsilon=simplify_epsilon,
        props_common=props_common,
    )

    flat_features: List[Dict[str, Any]] = []

    for feature in geojson["features"]:
        geometry = feature["geometry"]

        if geometry["type"] == "Polygon":
            flat_features.append(feature)
            continue

        if geometry["type"] != "MultiPolygon":
            continue

        for polygon_coords in geometry["coordinates"]:
            flat_features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": polygon_coords,
                    },
                    "properties": dict(feature["properties"]),
                }
            )

    return {
        "type": "FeatureCollection",
        "features": flat_features,
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

    Pipeline:
    - argmax once
    - for each class:
        build binary mask
        crop to bbox
        optional morphology
        connected-components filter
        fill small holes
        contours
        simplify
        rescale to level-0 coordinates
    """
    t_total = time.perf_counter()

    if avg_logits.ndim != 3:
        raise ValueError(
            f"Expected avg_logits with shape (C,H,W), got {tuple(avg_logits.shape)}"
        )

    c, h, w = map(int, avg_logits.shape)
    if len(class_names) != c:
        raise ValueError(
            f"class_names length ({len(class_names)}) must match number of channels ({c})"
        )

    logger.info(
        "logits_argmax_to_geojson_multipolygon: start | head=%s shape=(%d,%d,%d)",
        head_name,
        c,
        h,
        w,
    )

    t0 = time.perf_counter()
    pred = torch.argmax(avg_logits, dim=0).to(torch.int32).cpu().numpy()
    logger.info(
        "logits_argmax_to_geojson_multipolygon: argmax done | shape=%s dt=%.3fs",
        pred.shape,
        time.perf_counter() - t0,
    )

    props_common = props_common or {}
    include_set = set(include_classes) if include_classes is not None else None
    skip_set = set(skip_class_ids)

    features: List[Dict[str, Any]] = []

    for class_id in range(c):
        if class_id in skip_set:
            logger.debug("class %d skipped because in skip_class_ids", class_id)
            continue
        if include_set is not None and class_id not in include_set:
            logger.debug("class %d skipped because not in include_classes", class_id)
            continue

        t_class = time.perf_counter()

        cropped_mask, x0, y0 = _build_class_mask_crop(
            pred=pred,
            class_id=class_id,
            close_kernel=close_kernel,
            open_kernel=open_kernel,
            min_object_area=min_object_area,
            max_hole_area=max_hole_area,
        )

        if cropped_mask is None or cropped_mask.max() == 0:
            logger.debug(
                "class %d (%s): empty after mask construction | dt=%.3fs",
                class_id,
                class_names[class_id],
                time.perf_counter() - t_class,
            )
            continue

        polygons = _mask_to_polygons_with_holes(
            mask=cropped_mask,
            fx=fx,
            fy=fy,
            simplify_epsilon=simplify_epsilon,
            x_offset=x0,
            y_offset=y0,
        )

        if not polygons:
            logger.debug(
                "class %d (%s): no polygons found | dt=%.3fs",
                class_id,
                class_names[class_id],
                time.perf_counter() - t_class,
            )
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

        logger.info(
            "class %d (%s): bbox=(x0=%d,y0=%d,w=%d,h=%d) polygons=%d geom=%s dt=%.3fs",
            class_id,
            class_names[class_id],
            x0,
            y0,
            cropped_mask.shape[1],
            cropped_mask.shape[0],
            len(polygons),
            geometry["type"],
            time.perf_counter() - t_class,
        )

    out = {
        "type": "FeatureCollection",
        "features": features,
    }

    logger.info(
        "logits_argmax_to_geojson_multipolygon: done | features=%d total_dt=%.3fs",
        len(features),
        time.perf_counter() - t_total,
    )
    return out


def _build_class_mask_crop(
    pred: np.ndarray,
    class_id: int,
    close_kernel: int,
    open_kernel: int,
    min_object_area: int,
    max_hole_area: int,
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Build and clean a cropped binary uint8 mask (0/255) for one class.

    Returns:
        cropped_mask, x_offset, y_offset
    """
    t0 = time.perf_counter()

    ys, xs = np.where(pred == class_id)
    if len(xs) == 0:
        logger.debug(
            "_build_class_mask_crop: class=%d absent from argmax | dt=%.3fs",
            class_id,
            time.perf_counter() - t0,
        )
        return None, 0, 0

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    pred_crop = pred[y0:y1, x0:x1]
    mask = (pred_crop == class_id).astype(np.uint8) * 255

    logger.debug(
        "_build_class_mask_crop: class=%d raw_bbox=(x0=%d,y0=%d,w=%d,h=%d)",
        class_id,
        x0,
        y0,
        x1 - x0,
        y1 - y0,
    )

    if close_kernel and close_kernel > 1:
        t_morph = time.perf_counter()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel, close_kernel),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        logger.debug(
            "_build_class_mask_crop: class=%d close_kernel=%d dt=%.3fs",
            class_id,
            close_kernel,
            time.perf_counter() - t_morph,
        )

    if open_kernel and open_kernel > 1:
        t_morph = time.perf_counter()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel, open_kernel),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        logger.debug(
            "_build_class_mask_crop: class=%d open_kernel=%d dt=%.3fs",
            class_id,
            open_kernel,
            time.perf_counter() - t_morph,
        )

    t_cc = time.perf_counter()
    mask = _remove_small_components(mask, min_object_area)
    logger.debug(
        "_build_class_mask_crop: class=%d remove_small_components dt=%.3fs",
        class_id,
        time.perf_counter() - t_cc,
    )

    t_holes = time.perf_counter()
    mask = _fill_small_holes(mask, max_hole_area)
    logger.debug(
        "_build_class_mask_crop: class=%d fill_small_holes dt=%.3fs",
        class_id,
        time.perf_counter() - t_holes,
    )

    if mask.max() == 0:
        logger.debug(
            "_build_class_mask_crop: class=%d empty after cleaning | dt=%.3fs",
            class_id,
            time.perf_counter() - t0,
        )
        return None, x0, y0

    logger.debug(
        "_build_class_mask_crop: class=%d done | bbox=(x0=%d,y0=%d,w=%d,h=%d) dt=%.3fs",
        class_id,
        x0,
        y0,
        mask.shape[1],
        mask.shape[0],
        time.perf_counter() - t0,
    )
    return mask, x0, y0


def _mask_to_polygons_with_holes(
    mask: np.ndarray,
    fx: float,
    fy: float,
    simplify_epsilon: float,
    x_offset: int = 0,
    y_offset: int = 0,
) -> List[List[List[List[float]]]]:
    """
    Convert a cropped binary mask to a list of polygons with holes.

    Returns:
        polygons = [
            [outer_ring, hole1, hole2, ...],
            [outer_ring, hole1, ...],
            ...
        ]

    Each ring is a list of [x, y] coordinates in level-0 space.
    """
    t0 = time.perf_counter()

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None or len(contours) == 0:
        logger.debug(
            "_mask_to_polygons_with_holes: no contours | dt=%.3fs",
            time.perf_counter() - t0,
        )
        return []

    hierarchy = hierarchy[0]

    if simplify_epsilon and simplify_epsilon > 0:
        t_simplify = time.perf_counter()
        contours = [
            cv2.approxPolyDP(cnt, float(simplify_epsilon), True) for cnt in contours
        ]
        logger.debug(
            "_mask_to_polygons_with_holes: simplified %d contours eps=%.3f dt=%.3fs",
            len(contours),
            simplify_epsilon,
            time.perf_counter() - t_simplify,
        )

    polygons: List[List[List[List[float]]]] = []

    for contour_idx, h in enumerate(hierarchy):
        parent_idx = h[3]
        if parent_idx != -1:
            continue

        outer = contours[contour_idx]
        if not _is_valid_contour(outer):
            continue

        rings: List[List[List[float]]] = [
            _contour_to_ring(
                outer,
                fx=fx,
                fy=fy,
                x_offset=x_offset,
                y_offset=y_offset,
            )
        ]

        child_idx = h[2]
        while child_idx != -1:
            hole = contours[child_idx]
            if _is_valid_contour(hole):
                rings.append(
                    _contour_to_ring(
                        hole,
                        fx=fx,
                        fy=fy,
                        x_offset=x_offset,
                        y_offset=y_offset,
                    )
                )
            child_idx = hierarchy[child_idx][0]

        polygons.append(rings)

    logger.debug(
        "_mask_to_polygons_with_holes: contours=%d polygons=%d dt=%.3fs",
        len(contours),
        len(polygons),
        time.perf_counter() - t0,
    )
    return polygons


def _polygons_to_geojson_geometry(
    polygons: List[List[List[List[float]]]],
) -> Dict[str, Any]:
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
    x_offset: int = 0,
    y_offset: int = 0,
) -> List[List[float]]:
    """
    Convert an OpenCV contour from cropped mask coordinates to a closed GeoJSON ring
    in level-0 coordinates.
    """
    points = contour.reshape(-1, 2).astype(np.float64)
    points[:, 0] += float(x_offset)
    points[:, 1] += float(y_offset)
    points[:, 0] *= float(fx)
    points[:, 1] *= float(fy)

    ring: List[List[float]] = points.tolist()

    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])

    return ring
