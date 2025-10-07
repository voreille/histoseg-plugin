# parameter_dataclass.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Set, Literal

# ---------- Mutable blocks (per-WSI tunable) ----------


@dataclass
class SegmentationParams:
    """Parameters for tissue segmentation."""
    seg_level: int = -1
    sthresh: int = 8
    mthresh: int = 7
    close: int = 4
    use_otsu: bool = False
    keep_ids: str = "none"
    exclude_ids: str = "none"


@dataclass
class FilterParams:
    """Parameters for contour filtering."""
    a_t: int = 100  # area threshold (tissue)
    a_h: int = 16  # area threshold (holes)
    max_n_holes: int = 8


@dataclass
class VisualizationParams:
    """Parameters for visualization."""
    vis_level: int = -1
    line_thickness: int = 500


@dataclass
class PatchParams:
    """Parameters for patch extraction."""
    use_padding: bool = True
    contour_fn: str = "four_pt"


# ---------- Run-defining tiling block (immutable per run/dir) ----------

LevelPolicy = Literal["closest", "lower", "higher", "exact"]


@dataclass(frozen=True)
class Tiling:
    """
    Run-level tiling configuration (immutable for a given output directory).
    - If tile_level >= 0, use that exact pyramid level.
    - If tile_level < 0, pick a level based on target_tile_mpp and level_policy.
    """
    tile_size: int = 256
    step_size: int = 256
    target_tile_mpp: Optional[float] = 0.50  # µm/px target; None to disable
    mpp_tolerance: float = 0.10  # fraction (0.10 = 10%)
    tile_level: int = -1  # -1 => auto-select via policy
    level_policy: LevelPolicy = "closest"  # "closest" | "lower" | "higher" | "exact"


# ---------- Complete YAML config mirror ----------


@dataclass
class Config:
    """
    Full tiling config as stored in the YAML:
      tiling: {...}
      seg_params: {...}
      filter_params: {...}
      vis_params: {...}
      patch_params: {...}
    """
    tiling: Tiling = field(default_factory=Tiling)
    seg_params: SegmentationParams = field(default_factory=SegmentationParams)
    filter_params: FilterParams = field(default_factory=FilterParams)
    vis_params: VisualizationParams = field(
        default_factory=VisualizationParams)
    patch_params: PatchParams = field(default_factory=PatchParams)

    # ---------- Serialization ----------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        _validate_unknown_keys(
            data,
            {
                "tiling", "seg_params", "filter_params", "vis_params",
                "patch_params"
            },
            ctx="root",
        )

        tiling = Tiling(**_get_block(data, "tiling"))
        seg = SegmentationParams(**_get_block(data, "seg_params"))
        fil = FilterParams(**_get_block(data, "filter_params"))
        vis = VisualizationParams(**_get_block(data, "vis_params"))
        pat = PatchParams(**_get_block(data, "patch_params"))

        cfg = cls(tiling=tiling,
                  seg_params=seg,
                  filter_params=fil,
                  vis_params=vis,
                  patch_params=pat)
        cfg.validate()
        return cfg

    # ---------- Validation ----------

    def validate(self) -> None:
        # Basic type/range checks (extend as you like)
        if self.tiling.tile_size <= 0:
            raise ValueError("tiling.tile_size must be > 0")
        if self.tiling.step_size <= 0:
            raise ValueError("tiling.step_size must be > 0")
        if self.tiling.mpp_tolerance < 0:
            raise ValueError("tiling.mpp_tolerance must be >= 0")
        if self.tiling.level_policy not in ("closest", "lower", "higher",
                                            "exact"):
            raise ValueError(
                "tiling.level_policy must be one of: closest|lower|higher|exact"
            )

        # If exact policy is requested without tile_level, that's ambiguous.
        if self.tiling.level_policy == "exact" and self.tiling.tile_level < 0:
            raise ValueError(
                "tiling.level_policy=exact requires a non-negative tiling.tile_level"
            )

        # If target_tile_mpp is None and tile_level < 0, selection is under-specified.
        if self.tiling.target_tile_mpp is None and self.tiling.tile_level < 0:
            raise ValueError(
                "Provide tiling.target_tile_mpp or a non-negative tiling.tile_level"
            )


# ---------- Helpers ----------


def _get_block(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    blk = data.get(key, {}) or {}
    if not isinstance(blk, dict):
        raise ValueError(
            f"Expected '{key}' to be a mapping, got {type(blk).__name__}")
    return blk


def _validate_unknown_keys(d: Dict[str, Any], allowed: Set[str],
                           ctx: str) -> None:
    unknown = set(d.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {ctx}: {sorted(unknown)}")
