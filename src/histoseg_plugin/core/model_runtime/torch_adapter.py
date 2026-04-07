from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTorchModelAdapter(ABC):
    def __init__(self, model: nn.Module):
        self.model = model

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


class DefaultTorchModelAdapter(BaseTorchModelAdapter):
    def __init__(self, model, default_output_key: str | None = None):
        super().__init__(model=model)
        self.default_output_key = default_output_key

    def predict_tiles(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(batch)

        if torch.is_tensor(out):
            if self.default_output_key is None:
                raise ValueError(
                    "Model returned a tensor but no default_output_key was provided"
                )
            return {self.default_output_key: out}

        if isinstance(out, dict):
            return {k: v for k, v in out.items() if torch.is_tensor(v)}

        raise TypeError(f"Unsupported torch model output type: {type(out)}")


class TwoStagePrototypeAdapter(BaseTorchModelAdapter):
    def predict_tiles(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(batch)
        if not isinstance(out, dict):
            raise TypeError(
                f"Expected dict output from prototype model, got {type(out)}"
            )
        return {k: v for k, v in out.items() if torch.is_tensor(v)}

    def supports_context(self) -> bool:
        return True

    def fit_context_from_support(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.model.fit_prototypes_from_image_labels(
            images=support_images,
            image_labels=support_labels,
        )

    def predict_tiles_with_ctx(
        self,
        batch: torch.Tensor,
        ctx: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out = self.model(batch, ctx=ctx)
        if not isinstance(out, dict):
            raise TypeError(
                f"Expected dict output from prototype model with ctx, got {type(out)}"
            )
        return {k: v for k, v in out.items() if torch.is_tensor(v)}
