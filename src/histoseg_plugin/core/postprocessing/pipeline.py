from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .base import DerivedOutputProcessor


@dataclass
class PostprocessingResult:
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class WSIPostprocessor:
    """
    Runs configured postprocessing nodes on stitched outputs.

    Input:
      stitched_outputs:
        dict[str, Tensor], typically raw stitched outputs like:
          {
            "logits_a": (C,H,W),
            "logits_b": (C,H,W),
          }

      aux:
        optional dictionary for extra runtime information, e.g.:
          {
            "logits_b.covered_mask": (H,W) bool,
          }

    Output:
      PostprocessingResult:
        - outputs: includes raw outputs + derived outputs
        - metadata: extra non-client tensors useful for stats/debugging
    """

    def __init__(
        self,
        processors: dict[str, DerivedOutputProcessor],
    ):
        self.processors = processors

    def __call__(
        self,
        stitched_outputs: dict[str, torch.Tensor],
        aux: dict[str, Any] | None = None,
    ) -> PostprocessingResult:
        outputs = dict(stitched_outputs)
        metadata: dict[str, Any] = {}

        for derived_name, processor in self.processors.items():
            result = processor(outputs, aux=aux)
            for out_name, out_value in result.outputs.items():
                if out_name in outputs:
                    raise ValueError(
                        f"Postprocessor '{derived_name}' tried to overwrite existing "
                        f"output '{out_name}'"
                    )
                outputs[out_name] = out_value

            for meta_name, meta_value in result.metadata.items():
                if meta_name in metadata:
                    raise ValueError(
                        f"Postprocessor '{derived_name}' tried to overwrite existing "
                        f"metadata '{meta_name}'"
                    )
                metadata[meta_name] = meta_value

        return PostprocessingResult(outputs=outputs, metadata=metadata)