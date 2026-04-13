from pathlib import Path

import torch
import yaml

from .bundle import InferenceBundle
from histoseg_plugin.core.model_runtime.loader import load_model_bundle
from histoseg_plugin.core.postprocessing.builders import build_postprocessor
from histoseg_plugin.core.postprocessing.contracts import PostprocessingConfig


def load_inference_bundle(model_dir: Path, device: torch.device) -> InferenceBundle:
    model_runner = load_model_bundle(model_dir, device=device)

    postprocessor = None
    post_cfg_path = model_dir / "postprocessing.yaml"
    if post_cfg_path.exists():
        with post_cfg_path.open("r", encoding="utf-8") as f:
            cfg = PostprocessingConfig.model_validate(yaml.safe_load(f))
        postprocessor = build_postprocessor(cfg, base_dir=model_dir)

    return InferenceBundle(
        model_runner=model_runner,
        postprocessor=postprocessor,
    )