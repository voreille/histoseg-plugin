from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from .contracts import ModelManifest


class BaseModelRunner(ABC):
    def __init__(self, model_dir: Path, manifest: ModelManifest):
        self.model_dir = model_dir
        self.manifest = manifest
        self.device = torch.device("cpu")

    @abstractmethod
    def predict(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        pass