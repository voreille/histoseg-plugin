from pathlib import Path

from dataclasses import dataclass
from typing import Any, Literal

from ..geojson.schemas import GeoJSONFeatureCollection
from ..postprocessing.schemas import DemoPatternStatistics


@dataclass
class WSISegmentationInput:
    slide_path: Path
    tissue_seg_level: int
    sthresh: int
    sthresh_up: int
    mthresh: int
    close: int
    use_otsu: bool
    filter_params: dict[str, Any]
    ref_patch_size: int
    exclude_ids: list[int]
    keep_ids: list[int]
    min_area_px_level0: int
    contour_fn: str
    center_shift: float
    use_padding: bool
    top_left: tuple[int, int] | None
    bot_right: tuple[int, int] | None
    max_workers: int
    output_target_mpp: float 
    batch_size: int
    num_workers: int


@dataclass
class WSISegmentationResult:
    coords_space: Literal["level0"]
    tissue: GeoJSONFeatureCollection
    outputs: dict[str, GeoJSONFeatureCollection]
    statistics: DemoPatternStatistics | None
