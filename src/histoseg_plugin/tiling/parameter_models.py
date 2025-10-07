# src/histoseg_plugin/tiling/parameter_models.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    a_t: int = 100
    a_h: int = 16
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

LevelPolicy = Literal["closest", "lower", "higher"]


class Tiling(BaseModel):
    """
    Run-level tiling configuration.

    level_mode:
      - "fixed": use tile_level (>=0)
      - "auto": select level from target_tile_mpp + level_policy (+/- mpp_tolerance)
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    level_mode: Literal["auto", "fixed"] = "auto"

    tile_size: int = Field(default=256, gt=0)
    step_size: int = Field(default=256, gt=0)

    # Policy fields
    target_tile_mpp: Optional[float] = Field(default=0.50, gt=0)
    mpp_tolerance: float = Field(default=0.10, ge=0,
                                 le=1)  # fraction (0.10 = 10%)
    tile_level: int = -1  # valid when level_mode="fixed"
    level_policy: LevelPolicy = "closest"  # valid when level_mode="auto"

    @model_validator(mode="after")
    def _validate_policy(self) -> "Tiling":
        if self.level_mode == "fixed":
            if self.tile_level < 0:
                raise ValueError("Fixed mode requires tile_level ≥ 0.")
            # target_tile_mpp not required/used in fixed mode (may be present but ignored)
        else:  # auto
            if self.target_tile_mpp is None:
                raise ValueError("Auto mode requires target_tile_mpp.")
            # tile_level is ignored in auto (kept as -1)
        return self


# -----------------------------
# Full config (YAML mirror)
# -----------------------------


class Config(BaseModel):
    """
    Full config as stored in YAML:

    tiling:
      level_mode: auto|fixed
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

    # ---------- Mode-aware overrides ----------

    def with_tiling_overrides(
        self,
        *,
        # common
        tile_size: int | None = None,
        step_size: int | None = None,
        # explicit mode switch (optional)
        level_mode: Literal["auto", "fixed"] | None = None,
        # fixed
        tile_level: int | None = None,
        # auto
        target_tile_mpp: float | None = None,
        mpp_tolerance: float | None = None,
        level_policy: LevelPolicy | None = None,
    ) -> "Config":
        """
        Create a new Config with updated tiling.
        Precedence:
          1) explicit level_mode if provided
          2) else tile_level -> fixed ; target_tile_mpp -> auto
          3) otherwise: geometry-only update
        """
        t = self.tiling
        updates: dict = {}

        # always allow geometry updates
        if tile_size is not None:
            updates["tile_size"] = tile_size
        if step_size is not None:
            updates["step_size"] = step_size

        # decide target mode
        target_mode: str | None = level_mode
        if target_mode is None:
            if tile_level is not None:
                target_mode = "fixed"
            elif target_tile_mpp is not None or mpp_tolerance is not None or level_policy is not None:
                target_mode = "auto"

        # apply mode-specific updates
        if (target_mode or t.level_mode) == "fixed":
            # If switching to fixed, ensure tile_level is set
            lvl = tile_level if tile_level is not None else (
                t.tile_level if t.level_mode == "fixed" else None)
            if lvl is None or lvl < 0:
                raise ValueError(
                    "Switching to fixed mode requires a non-negative tile_level."
                )
            updates.update({
                "level_mode": "fixed",
                "tile_level": lvl,
                # keep existing geometry; auto-only fields are left as-is/ignored
            })
        elif (target_mode or t.level_mode) == "auto":
            # If switching to auto, ensure target_tile_mpp exists
            tgt = target_tile_mpp if target_tile_mpp is not None else (
                t.target_tile_mpp if t.level_mode == "auto" else None)
            if tgt is None:
                raise ValueError(
                    "Switching to auto mode requires target_tile_mpp.")
            updates.update({
                "level_mode":
                "auto",
                "target_tile_mpp":
                tgt,
                "mpp_tolerance":
                mpp_tolerance
                if mpp_tolerance is not None else t.mpp_tolerance,
                "level_policy":
                level_policy if level_policy is not None else t.level_policy,
                "tile_level":
                -1,  # ignore fixed value in auto mode
            })
        else:
            # No mode change requested and none inferred — geometry-only update
            pass

        new_tiling = t.model_copy(update=updates)
        return self.model_copy(update={"tiling": new_tiling})

    # Convenience one-liners
    def to_fixed(self, *, tile_level: int) -> "Config":
        return self.with_tiling_overrides(level_mode="fixed",
                                          tile_level=tile_level)

    def to_auto(
        self,
        *,
        target_tile_mpp: float,
        mpp_tolerance: float | None = None,
        level_policy: LevelPolicy | None = None,
    ) -> "Config":
        return self.with_tiling_overrides(
            level_mode="auto",
            target_tile_mpp=target_tile_mpp,
            mpp_tolerance=mpp_tolerance,
            level_policy=level_policy,
        )
