from __future__ import annotations

from pathlib import Path

from .base import DerivedOutputProcessor
from .contracts import DerivedOutputSpec, PostprocessingConfig
from .pipeline import WSIPostprocessor
from .processors import (
    ConformalSetFromLogitsProcessor,
    PrototypeFGMergeProcessor,
)


def build_derived_output_processor(
    derived_name: str,
    spec: DerivedOutputSpec,
    *,
    base_dir: Path | None = None,
) -> DerivedOutputProcessor:
    if spec.type == "prototype_fg_merge":
        return PrototypeFGMergeProcessor(
            derived_name=derived_name,
            spec=spec,
        )

    if spec.type == "conformal_set_from_logits":
        return ConformalSetFromLogitsProcessor(
            derived_name=derived_name,
            spec=spec,
            base_dir=base_dir,
        )

    raise ValueError(f"Unknown derived output type='{spec.type}' for '{derived_name}'")


def build_postprocessor(
    cfg: PostprocessingConfig,
    *,
    base_dir: Path | None = None,
) -> WSIPostprocessor:
    processors: dict[str, DerivedOutputProcessor] = {}

    for derived_name, spec in cfg.derived_outputs.items():
        processors[derived_name] = build_derived_output_processor(
            derived_name=derived_name,
            spec=spec,
            base_dir=base_dir,
        )

    return WSIPostprocessor(processors=processors)