from __future__ import annotations

from pathlib import Path

import torch

from .base import BaseModelRunner
from .contracts import ModelManifest


class TorchExportRunner(BaseModelRunner):
    def __init__(
        self,
        model_dir: Path,
        manifest: ModelManifest,
        device: torch.device,
    ):
        super().__init__(
            model_dir=model_dir,
            manifest=manifest,
        )

        weights_path = model_dir / manifest.inference.weights
        self.device = device

        exported = torch.export.load(weights_path)

        exported = torch.export.passes.move_to_device_pass(
            exported,
            device,
        )

        self.model = exported.module().eval()

        self.use_amp = manifest.inference.use_amp
        self.amp_dtype = (
            torch.float16 if manifest.inference.amp_dtype == "float16" else torch.bfloat16
        )

        output_keys = list(manifest.output)

        if not output_keys:
            raise ValueError("Model manifest must define at least one output.")

        self.output_keys = output_keys

    def predict_tiles(
        self,
        batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = batch.to(
            self.device,
            non_blocking=True,
        )
        batch = self.preprocess(batch)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                ):
                    out = self.model(batch)
            else:
                out = self.model(batch)

        if torch.is_tensor(out):
            if len(self.output_keys) != 1:
                raise ValueError(
                    f"Tensor output requires exactly one output key, got {self.output_keys}."
                )

            return {
                self.output_keys[0]: out,
            }

        if isinstance(out, dict):
            return {key: value for key, value in out.items() if torch.is_tensor(value)}

        raise TypeError(f"Unsupported torch.export output type: {type(out)}")
