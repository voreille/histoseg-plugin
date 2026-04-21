from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PolygonGeometry:
    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


@dataclass
class MultiPolygonGeometry:
    type: Literal["MultiPolygon"]
    coordinates: list[list[list[list[float]]]]


@dataclass
class GeoJSONFeature:
    geometry: PolygonGeometry | MultiPolygonGeometry
    properties: dict[str, Any] = field(default_factory=dict)
    type: Literal["Feature"] = "Feature"


@dataclass
class GeoJSONFeatureCollection:
    features: list[GeoJSONFeature] = field(default_factory=list)
    type: Literal["FeatureCollection"] = "FeatureCollection"
