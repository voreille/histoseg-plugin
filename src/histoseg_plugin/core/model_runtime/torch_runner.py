from __future__ import annotations

from pathlib import Path

import torch

from .base import BaseModelRunner
from .contracts import ModelManifest
from .torch_adapters.default import DefaultTorchModelAdapter
from .utils import import_from_string


class TorchRunner(BaseModelRunner):
    def __init__(self, model_dir: Path, manifest: ModelManifest, device: torch.device):
        super().__init__(model_dir=model_dir, manifest=manifest)

        if manifest.model is None:
            raise ValueError("TorchRunner requires manifest.model")

        self.device = device

        model_factory = import_from_string(manifest.model.factory)
        self.model = model_factory(**manifest.model.init_args)

        weights_path = model_dir / manifest.inference.weights
        state = torch.load(weights_path, map_location="cpu")

        # state_dict only
        if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            self.model.load_state_dict(state)
        else:
            raise TypeError(
                "TorchRunner expects weights to be a state_dict saved with torch.save(model.state_dict(), ...)"
            )

        self.model = self.model.to(self.device).eval()

        self.use_amp = manifest.inference.use_amp
        self.amp_dtype = self._resolve_amp_dtype(manifest.inference.amp_dtype)

        output_keys = list(manifest.output.keys())
        default_output_key = output_keys[0] if len(output_keys) == 1 else None

        if manifest.model.adapter_factory is not None:
            adapter_factory = import_from_string(manifest.model.adapter_factory)
            self.adapter = adapter_factory(self.model)
        else:
            self.adapter = DefaultTorchModelAdapter(
                model=self.model,
                default_output_key=default_output_key,
            )

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str | None):
        if amp_dtype is None:
            return torch.float16
        if amp_dtype == "float16":
            return torch.float16
        if amp_dtype == "bfloat16":
            return torch.bfloat16
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")

    def predict_tiles(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = batch.to(self.device, non_blocking=True)
        batch = self.preprocess(batch)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    out = self.adapter.predict_tiles(batch)
            else:
                out = self.adapter.predict_tiles(batch)

        return out

    def supports_context(self) -> bool:
        return self.adapter.supports_context()

    def fit_context_from_support(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.supports_context():
            raise NotImplementedError("This torch model does not support context")

        support_images = support_images.to(self.device, non_blocking=True)
        support_images = self.preprocess(support_images)
        support_labels = support_labels.to(self.device, non_blocking=True)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    return self.adapter.fit_context_from_support(
                        support_images=support_images,
                        support_labels=support_labels,
                    )
            return self.adapter.fit_context_from_support(
                support_images=support_images,
                support_labels=support_labels,
            )

    def predict_tiles_with_ctx(
        self,
        batch: torch.Tensor,
        ctx: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.supports_context():
            raise NotImplementedError("This torch model does not support context")

        batch = batch.to(self.device, non_blocking=True)
        batch = self.preprocess(batch)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    return self.adapter.predict_tiles_with_ctx(batch=batch, ctx=ctx)
            return self.adapter.predict_tiles_with_ctx(batch=batch, ctx=ctx)