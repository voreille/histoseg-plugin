from __future__ import annotations

from pathlib import Path

import torch

from .manifest import load_manifest


def load_model_bundle(model_dir: Path, device: torch.device):
    manifest = load_manifest(model_dir)

    runtime = manifest.inference.runtime
    if runtime == "torchscript":
        from .torchscript import TorchScriptRunner

        return TorchScriptRunner(model_dir=model_dir, manifest=manifest, device=device)
    if runtime == "onnx":
        from .onnx import ONNXRunner

        return ONNXRunner(model_dir=model_dir, manifest=manifest)

    if runtime == "torch":
        from .torch_runner import TorchRunner

        return TorchRunner(model_dir=model_dir, manifest=manifest, device=device)



    raise ValueError(f"Unsupported runtime: {runtime}")
