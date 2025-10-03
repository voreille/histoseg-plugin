# src/histoseg_plugin/tiling/parameter_models.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict, model_validator

# -----------------------
# Mutable blocks (per-WSI)
# -----------------------


class SegmentationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seg_level: int = -1
    sthresh: int = 8
    mthresh: int = 7
    close: int = 4
    use_otsu: bool = False
    keep_ids: str = "none"
    exclude_ids: str = "none"


class FilterParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a_t: int = 100  # area threshold (tissue)
    a_h: int = 16  # area threshold (holes)
    max_n_holes: int = 8


class VisualizationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vis_level: int = -1
    line_thickness: int = 500


class PatchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_padding: bool = True
    contour_fn: str = "four_pt"


# ------------------------------------------
# Run-defining tiling block (immutable/“fixed”)
# ------------------------------------------

LevelPolicy = Literal["closest", "lower", "higher", "exact"]


class Tiling(BaseModel):
    """
    Run-level tiling configuration.

    If tile_level >= 0 => use that level (policy still documented, but not used).
    If tile_level < 0  => select level from target_tile_mpp + level_policy.
    """
    # immutable intent: don't mutate in-place — use .model_copy(update=...) to override
    model_config = ConfigDict(extra="forbid", frozen=True)

    tile_size: int = Field(default=256, gt=0)
    step_size: int = Field(default=256, gt=0)

    # Policy fields
    target_tile_mpp: Optional[float] = Field(default=0.50, gt=0)
    mpp_tolerance: float = Field(default=0.10, ge=0, le=1)  # fraction (0.10 = 10%)
    tile_level: int = -1  # -1 => auto-select via policy
    level_policy: LevelPolicy = "closest"  # closest | lower | higher | exact

    @model_validator(mode="after")
    def _validate_policy(self) -> "Tiling":
        # If exact is requested, a concrete non-negative tile_level must be provided
        if self.level_policy == "exact" and self.tile_level < 0:
            raise ValueError(
                "tiling.level_policy='exact' requires tiling.tile_level >= 0")

        # If neither tile_level nor target_tile_mpp is usable, selection is under-specified
        if self.tile_level < 0 and self.target_tile_mpp is None:
            raise ValueError(
                "Provide either tiling.tile_level >= 0 or tiling.target_tile_mpp"
            )

        return self


# -----------------------------
# Full config (YAML mirror)
# -----------------------------


class Config(BaseModel):
    """
    Full config as stored in YAML:

    tiling:
      tile_size: ...
      step_size: ...
      target_tile_mpp: ...
      mpp_tolerance: ...
      tile_level: ...
      level_policy: ...
    seg_params: {...}
    filter_params: {...}
    vis_params: {...}
    patch_params: {...}
    """
    model_config = ConfigDict(extra="forbid")

    tiling: Tiling = Tiling()
    seg_params: SegmentationParams = SegmentationParams()
    filter_params: FilterParams = FilterParams()
    vis_params: VisualizationParams = VisualizationParams()
    patch_params: PatchParams = PatchParams()

    # ---------- I/O helpers ----------

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(), sort_keys=False))

    # ---------- Functional overrides (for CLI) ----------

    def with_tiling_overrides(
        self,
        *,
        tile_size: int | None = None,
        step_size: int | None = None,
        tile_level: int | None = None,
        target_tile_mpp: float | None = None,
        mpp_tolerance: float | None = None,
        level_policy: LevelPolicy | None = None,
    ) -> "Config":
        """
        Returns a new Config with updated tiling fields (does not mutate self).
        Precedence: explicit tile_level dominates target_tile_mpp if both provided.
        """
        updates = {}
        if tile_size is not None:
            updates["tile_size"] = tile_size
        if step_size is not None:
            updates["step_size"] = step_size
        if level_policy is not None:
            updates["level_policy"] = level_policy
        if mpp_tolerance is not None:
            updates["mpp_tolerance"] = mpp_tolerance

        # policy precedence
        if tile_level is not None:
            updates["tile_level"] = tile_level
            updates["target_tile_mpp"] = None
        elif target_tile_mpp is not None:
            updates["target_tile_mpp"] = target_tile_mpp
            # keep tile_level as-is; negative means “auto”

        new_tiling = self.tiling.model_copy(update=updates)
        return self.model_copy(update={"tiling": new_tiling})
