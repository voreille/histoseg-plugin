from __future__ import annotations

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field


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
