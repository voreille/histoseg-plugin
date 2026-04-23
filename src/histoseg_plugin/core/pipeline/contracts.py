from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..geojson.schemas import GeoJSONFeatureCollection
from ..postprocessing.schemas import DemoPatternStatistics


@dataclass
class WSISegmentationResult:
    coords_space: Literal["level0"]
    tissue: GeoJSONFeatureCollection
    outputs: dict[str, GeoJSONFeatureCollection]
    statistics: DemoPatternStatistics | None


@dataclass(frozen=True)
class TissueSegmentationParams:
    seg_level: int = -1
    sthresh: int = 20
    sthresh_up: int = 255
    mthresh: int = 7
    close: int = 0
    use_otsu: bool = False
    filter_params: dict[str, int] = field(
        default_factory=lambda: {"a_t": 100, "a_h": 16, "max_n_holes": 10}
    )
    ref_patch_size: int = 512
    exclude_ids: list[int] = field(default_factory=list)
    keep_ids: list[int] = field(default_factory=list)
    min_area_px_level0: int = 0


@dataclass(frozen=True)
class TilingParams:
    contour_fn: str = "four_pt"
    center_shift: float = 0.5
    use_padding: bool = True
    top_left: tuple[int, int] | None = None
    bot_right: tuple[int, int] | None = None
    max_workers: int = 4


@dataclass(frozen=True)
class InferenceParams:
    output_target_mpp: float = 2.0
    batch_size: int = 32
    num_workers: int = 8


@dataclass(frozen=True)
class WSISegmentationInput:
    slide_path: Path
    model_id: str = "default"
    tissue: TissueSegmentationParams = field(default_factory=TissueSegmentationParams)
    tiling: TilingParams = field(default_factory=TilingParams)
    inference: InferenceParams = field(default_factory=InferenceParams)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["slide_path"] = str(self.slide_path)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WSISegmentationInput":
        tiling = data.get("tiling", {})
        return cls(
            slide_path=Path(data["slide_path"]),
            model_id=data.get("model_id", "default"),
            tissue=TissueSegmentationParams(**data.get("tissue", {})),
            tiling=TilingParams(
                **{
                    **tiling,
                    "top_left": tuple(tiling["top_left"])
                    if tiling.get("top_left")
                    else None,
                    "bot_right": tuple(tiling["bot_right"])
                    if tiling.get("bot_right")
                    else None,
                }
            ),
            inference=InferenceParams(**data.get("inference", {})),
        )
