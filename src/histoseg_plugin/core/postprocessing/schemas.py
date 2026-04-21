from __future__ import annotations


from pydantic import BaseModel, Field


class PatternBoundStats(BaseModel):
    pattern_id: int
    safe_area_px: int
    argmax_area_px: int
    max_area_px: int
    safe_ratio: float
    argmax_ratio: float
    max_ratio: float


class CompartmentPatternStats(BaseModel):
    compartment_id: int
    area_px: int
    area_um2: float
    patterns: dict[str, PatternBoundStats] = Field(default_factory=dict)


class DemoPatternStatistics(BaseModel):
    head_b_foreground_area_px: int
    head_b_foreground_area_um2: float
    patterns: dict[str, PatternBoundStats] = Field(default_factory=dict)
    compartments: dict[str, CompartmentPatternStats] = Field(default_factory=dict)
