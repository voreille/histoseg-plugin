from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from .contracts import DerivedOutputSpec


@dataclass
class ProcessorResult:
    """
    outputs:
        New named outputs produced by the processor.
        Example:
          {"prototype_prediction": tensor}
        or for conformal:
          {
            "luad_conformal.safe": ...,
            "luad_conformal.max_possible": ...,
            "luad_conformal.set_size": ...,
          }

    metadata:
        Optional extra metadata for downstream logic/debugging/stats.
    """

    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class DerivedOutputProcessor(ABC):
    def __init__(self, derived_name: str, spec: DerivedOutputSpec):
        self.derived_name = derived_name
        self.spec = spec

    @abstractmethod
    def __call__(
        self,
        stitched_outputs: dict[str, torch.Tensor],
        aux: dict[str, Any] | None = None,
    ) -> ProcessorResult:
        raise NotImplementedError