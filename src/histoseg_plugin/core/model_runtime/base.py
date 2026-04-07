from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from .contracts import ModelManifest
from .preprocessor import build_preprocessor


class BaseModelRunner(ABC):
    def __init__(self, model_dir: Path, manifest: ModelManifest):
        self.model_dir = model_dir
        self.manifest = manifest
        self.device = torch.device("cpu")

        if manifest.input.preprocessing is not None:
            self.preprocessor = build_preprocessor(
                model_dir=model_dir,
                spec=manifest.input.preprocessing,
            )
        else:
            self.preprocessor = None

    def preprocess(self, batch: torch.Tensor) -> torch.Tensor:
        if self.preprocessor is not None:
            return self.preprocessor(batch)
        return batch

    def get_preprocessor(self):
        return self.preprocessor

    @abstractmethod
    def predict_tiles(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def supports_context(self) -> bool:
        return False

    def fit_context_from_support(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support context fitting"
        )

    def predict_tiles_with_ctx(
        self,
        batch: torch.Tensor,
        ctx: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support context inference"
        )