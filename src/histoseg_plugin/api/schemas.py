from __future__ import annotations

from typing import Dict, Literal, List

from pydantic import BaseModel, Field

from histoseg_plugin.core.geojson.schemas import GeoJSONFeatureCollection
from histoseg_plugin.core.postprocessing.schemas import DemoPatternStatistics


class TissueSegmentationConfig(BaseModel):
    seg_level: int = -1
    sthresh: int = 20
    sthresh_up: int = 255
    mthresh: int = 7
    close: int = 0
    use_otsu: bool = False
    filter_params: dict[str, int] = Field(
        default_factory=lambda: {"a_t": 100, "a_h": 16, "max_n_holes": 10}
    )
    ref_patch_size: int = 512
    exclude_ids: list[int] = Field(default_factory=list)
    keep_ids: list[int] = Field(default_factory=list)
    min_area_px_level0: int = 0
    simplify_tol_px_level0: float = 0.0


class TilingConfig(BaseModel):
    contour_fn: str = "four_pt"
    center_shift: float = 0.5
    use_padding: bool = True
    top_left: list[int] | None = None
    bot_right: list[int] | None = None
    max_workers: int = 4


class InferenceConfig(BaseModel):
    output_target_mpp: float = 2.0
    batch_size: int = 32
    num_workers: int = 8


class TissueContoursRequest(BaseModel):
    slide_uri: str
    tissue: TissueSegmentationConfig = Field(default_factory=TissueSegmentationConfig)


class WSISegmentationRequest(BaseModel):
    slide_uri: str
    tissue: TissueSegmentationConfig = Field(default_factory=TissueSegmentationConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


class WSISegmentationResponse(BaseModel):
    coords_space: Literal["level0"] = "level0"
    tissue: GeoJSONFeatureCollection
    outputs: Dict[str, GeoJSONFeatureCollection]
    statistics: DemoPatternStatistics | None = None


class JobItem(BaseModel):
    # define properly later
    # for now minimal
    slide_uri: str
    model_id: str | None = "default"
    params: dict = {}


class CreateJobRequest(BaseModel):
    items: List[JobItem]
