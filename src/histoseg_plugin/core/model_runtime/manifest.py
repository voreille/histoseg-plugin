from __future__ import annotations

import json
from pathlib import Path

import yaml

from .contracts import ModelManifest


def load_manifest(model_dir: Path) -> ModelManifest:
    yaml_path = model_dir / "manifest.yaml"

    if yaml_path.exists():
        data = yaml.safe_load(yaml_path.read_text())
    else:
        raise FileNotFoundError(f"No model manifest found in {model_dir}")

    return ModelManifest.model_validate(data)
