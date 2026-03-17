from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is required for ONNX model runtime. "
        "Please install it with `pip install onnxruntime`."
    ) from e

from .base import BaseModelRunner
from .contracts import ModelManifest
from .preprocessor import build_preprocessor


class ONNXRunner(BaseModelRunner):
    def __init__(self, model_dir: Path, manifest: ModelManifest):
        super().__init__(model_dir=model_dir, manifest=manifest)

        weights_path = model_dir / manifest.inference.weights
        self.session = ort.InferenceSession(str(weights_path))
        self.input_name = self.session.get_inputs()[0].name
        self.input_layout = manifest.input.layout

        if manifest.input.preprocessing is not None:
            self.preprocessor = build_preprocessor(
                model_dir=model_dir,
                spec=manifest.input.preprocessing,
            )
        else:
            self.preprocessor = None

    def predict(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Input batch contract:
            - torch.Tensor
            - shape: (B, C, H, W)
            - RGB
            - float32
            - values in [0, 1]

        Output contract:
            - dict[str, torch.Tensor]
            - tensors normalized to BCHW for the rest of the pipeline
        """
        x = batch.detach().cpu().to(dtype=torch.float32)

        if self.preprocessor is not None:
            x = self.preprocessor(x)

        x_np = x.numpy().astype(np.float32)  # BCHW

        if self.input_layout == "BHWC":
            x_np = np.transpose(x_np, (0, 2, 3, 1))
        elif self.input_layout != "BCHW":
            raise ValueError(f"Unsupported input layout: {self.input_layout}")

        outputs = self.session.run(None, {self.input_name: x_np})

        manifest_heads = list(self.manifest.output.items())
        if len(outputs) != len(manifest_heads):
            raise ValueError(
                f"ONNX runtime returned {len(outputs)} outputs, "
                f"but manifest declares {len(manifest_heads)} heads."
            )

        out_dict: dict[str, torch.Tensor] = {}
        for (head_name, head_spec), out_np in zip(manifest_heads, outputs):
            out_t = torch.from_numpy(out_np)

            if head_spec.output_layout == "BHWC":
                out_t = out_t.permute(0, 3, 1, 2)
            elif head_spec.output_layout != "BCHW":
                raise ValueError(
                    f"Unsupported output layout for head '{head_name}': "
                    f"{head_spec.output_layout}"
                )

            out_dict[head_name] = out_t

        return out_dict