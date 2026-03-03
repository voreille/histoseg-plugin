from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TissueContoursRequest(BaseModel):
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

    # geojson post options (MVP)
    min_area_px_level0: int = 0
    simplify_tol_px_level0: float = 0.0


class GeoJSONGeometry(BaseModel):
    type: Literal["Polygon"]
    coordinates: List[List[List[float]]]  # [[[x,y], ...]]


class GeoJSONFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    properties: Dict[str, Any] = Field(default_factory=dict)
    geometry: GeoJSONGeometry


class GeoJSONFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoJSONFeature]
