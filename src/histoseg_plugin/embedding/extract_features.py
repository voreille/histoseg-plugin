from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..storage.factory import build_embedding_store, build_tiling_store_from_dir
from ..storage.config import EmbeddingStoreConfig
from .datasets import WholeSlidePatch
from .encoders import get_encoder

project_dir = Path(__file__).resolve().parents[3]
DEFAULT_CFG = project_dir / "configs" / "embedding.yaml"
DEFAULT_STORAGE_CFG = project_dir / "configs" / "storage.yaml"


@click.command(context_settings={"show_default": True})
@click.option(
    "--tiles-rootdir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help=
    "Root directory containing tiling outputs (must include .tiling_store.json).",
)
@click.option(
    "--slides-rootdir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Directory containing the raw WSIs.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for features (e.g., h5_files/, pt_files/).",
)
@click.option(
    "--model-name",
    type=click.Choice(["resnet50_trunc", "uni_v1", "conch_v1"]),
    default="resnet50_trunc",
)
@click.option("--batch-size", type=int, default=256)
@click.option("--num-workers", type=int, default=8)
@click.option("--target-patch-size", type=int, default=224)
@click.option(
    "--no-auto-skip",
    is_flag=True,
    help="Process even if .pt already exists.",
)
@click.option(
    "--use-amp",
    is_flag=True,
    help="Use autocast on CUDA for speed.",
)
@click.option(
    "--export-pt",
    is_flag=True,
)
def main(
    tiles_rootdir: Path,
    slides_rootdir: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    num_workers: int,
    target_patch_size: int,
    no_auto_skip: bool,
    use_amp: bool,
    export_pt: bool,
):
    # Build stores from config + runtime roots
    embedding_store_config = EmbeddingStoreConfig.from_yaml(
        path=DEFAULT_STORAGE_CFG,
        root_key="embedding",
    )
    tiling_store = build_tiling_store_from_dir(
        root_dir=tiles_rootdir,
        slides_root=slides_rootdir,
    )
    embedding_store = build_embedding_store(
        config=embedding_store_config,
        slides_root=slides_rootdir,
        root_dir=output_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tx, out_dim = get_encoder(model_name,
                                     target_img_size=target_patch_size)
    model.eval().to(device)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }

    autocast = (
        torch.amp.autocast("cuda")  # type: ignore[attr-defined]
        if (use_amp and device.type == "cuda") else nullcontext())

    for slide_id in tqdm(sorted(tiling_store.slide_ids()),
                         desc="Slides",
                         unit="slide"):
        coords, _, attrs = tiling_store.load_coords(slide_id)

        # If you stored 'wsi_relpath' or 'wsi_basename' in attrs, consider:
        # slide_path = tiling_store.resolve_wsi(slide_id, attrs)
        # For now you used 'relative_wsi_path':
        slide_path = slides_rootdir / attrs["relative_wsi_path"]

        tile_level = int(attrs["patch_level"])
        tile_size = int(attrs["patch_size"])

        ds = WholeSlidePatch(
            coords,
            wsi_path=slide_path,
            tile_level=tile_level,
            tile_size=tile_size,
            transform=tx,
        )
        loader = DataLoader(ds,
                            batch_size=batch_size,
                            shuffle=False,
                            **loader_kwargs)

        embedding_store.begin_slide(slide_id, dim=out_dim, attrs=attrs)

        with torch.inference_mode():
            for batch in tqdm(loader,
                              desc=f"Embedding {slide_id}",
                              leave=False):
                imgs = batch["img"].to(device, non_blocking=True)
                batch_coords = batch["coord"].numpy().astype(np.int32)

                with autocast:
                    feats = model(imgs)

                feats = feats.detach().cpu().numpy().astype(np.float32)
                embedding_store.append_batch(slide_id, feats, batch_coords)

        embedding_store.finalize_slide(slide_id)

        if export_pt:
            pt_dir = output_dir / "pt_files"
            pt_dir.mkdir(parents=False, exist_ok=True)
            out = embedding_store.export_to_pt(slide_id, pt_dir)
            tqdm.write(f"Exported {slide_id} to {out}")


if __name__ == "__main__":
    main()
