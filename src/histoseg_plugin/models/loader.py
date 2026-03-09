from pathlib import Path
import json
import torch


def load_model_bundle(model_dir: Path, device: torch.device):
    model = torch.jit.load(model_dir / "scripted_model.pt", map_location=device)
    model = model.to(device).eval()

    with open(model_dir / "model_manifest.json", "r") as f:
        manifest = json.load(f)

    return {
        "model": model,
        "manifest": manifest,
    }
