from __future__ import annotations

from pathlib import Path

import torch

from .base import BaseModelRunner
from .contracts import ModelManifest


class TorchScriptRunner(BaseModelRunner):
    def __init__(self, model_dir: Path, manifest: ModelManifest, device: torch.device):
        super().__init__(model_dir=model_dir, manifest=manifest)
        weights_path = model_dir / manifest.inference.weights
        self.device = device
        self.model = torch.jit.load(weights_path, map_location=device).to(device).eval()
        self.use_amp = manifest.inference.use_amp
        self.amp_dtype = (
            torch.float16
            if manifest.inference.amp_dtype == "float16"
            else torch.bfloat16
        )
        output_keys = list(manifest.output.keys())
        if len(output_keys) != 1:
            raise ValueError(
                f"Tensor-only TorchScript output currently expects exactly 1 output head, got {output_keys}"
            )
        self.default_output_key = output_keys[0]

    def predict_tiles(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = batch.to(self.device, non_blocking=True)
        batch = self.preprocess(batch)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    out = self.model(batch)
            else:
                out = self.model(batch)

        if torch.is_tensor(out):
            return {self.default_output_key: out}

        if isinstance(out, dict):
            return {k: v for k, v in out.items() if torch.is_tensor(v)}

        raise TypeError(f"Unsupported TorchScript output type: {type(out)}")
