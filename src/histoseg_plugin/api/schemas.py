from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class TissueSegmentationParams(BaseModel):
    slide_uri: str = Field(..., description="file:/... or file:///... or absolute path")
    seg_level: int = -1

    sthresh: int = 20
    sthresh_up: int = 255
    mthresh: int = 7
    close: int = 0
    use_otsu: bool = False

    filter_params: Dict[str, int] = Field(
        default_factory=lambda: {"a_t": 100, "a_h": 16, "max_n_holes": 10}
    )

    ref_patch_size: int = 512
    exclude_ids: Optional[List[int]] = None
    keep_ids: Optional[List[int]] = None

    min_area_px_level0: int = 0
    simplify_tol_px_level0: float = 0.0


class TissueContoursRequest(TissueSegmentationParams):
    pass


class WSISegmentationRequest(TissueSegmentationParams):
    patch_level: int = 0
    patch_size: int = 512
    step_size: int = 512

    contour_fn: str = "four_pt"
    center_shift: float = 0.5
    use_padding: bool = True
    top_left: Optional[List[int]] = None
    bot_right: Optional[List[int]] = None
    max_workers: int = 4  # for the tiling

    output_target_mpp: float = 4.0
    batch_size: int = 16
    num_workers: int = 0


class PolygonGeometry(BaseModel):
    type: Literal["Polygon"]
    coordinates: List[List[List[float]]]


class MultiPolygonGeometry(BaseModel):
    type: Literal["MultiPolygon"]
    coordinates: List[List[List[List[float]]]]


class GeoJSONFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    properties: Dict[str, Any] = Field(default_factory=dict)
    geometry: Union[PolygonGeometry, MultiPolygonGeometry]


class GeoJSONFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoJSONFeature]


class WSISegmentationResponse(BaseModel):
    coords_space: Literal["level0"] = "level0"
    tissue: GeoJSONFeatureCollection
    outputs: Dict[str, GeoJSONFeatureCollection]
