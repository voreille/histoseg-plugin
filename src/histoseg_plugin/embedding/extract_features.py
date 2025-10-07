# src/histoseg_plugin/embedding/extract_features.py
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import click
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import WholeSlidePatchH5
from .encoders import get_encoder


def _iter_patch_h5s(patches_dir: Path) -> Iterable[Path]:
    yield from sorted(patches_dir.glob("*.h5"))


def get_h5_dataset(h5: h5py.File, key: str) -> h5py.Dataset:
    obj = h5.get(key)
    if not isinstance(obj, h5py.Dataset):
        raise KeyError(
            f"'{key}' is not an HDF5 dataset (got {type(obj).__name__})")
    return obj


def _save_batch_h5(out_path: Path, features: np.ndarray, coords: np.ndarray,
                   mode: str) -> None:
    """
    Append-friendly HDF5: create datasets on first call, then resize and append.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, mode if out_path.exists() else "w") as f:
        if "features" not in f:
            # first write
            maxshape = (None, features.shape[1])  # (N, D)
            dset_f = f.create_dataset("features",
                                      data=features,
                                      maxshape=maxshape,
                                      chunks=True)
            dset_c = f.create_dataset("coords",
                                      data=coords,
                                      maxshape=(None, coords.shape[1]),
                                      chunks=True)
        else:
            dset_f = get_h5_dataset(f, "features")
            dset_c = get_h5_dataset(f, "coords")
            n0 = dset_f.shape[0]
            dset_f.resize(n0 + features.shape[0], axis=0)
            dset_f[n0:] = features
            dset_c.resize(n0 + coords.shape[0], axis=0)
            dset_c[n0:] = coords


@click.command(context_settings={"show_default": True})
@click.option("--data-h5-dir",
              type=click.Path(path_type=Path),
              required=True,
              help="Root dir containing patches/*.h5 (coords).")
@click.option("--data-slide-dir",
              type=click.Path(path_type=Path),
              required=True,
              help="Root dir containing original WSIs.")
@click.option("--slide-ext",
              type=str,
              default=".svs",
              help="WSI filename extension.")
@click.option("--feat-dir",
              type=click.Path(path_type=Path),
              required=True,
              help="Output dir for features (h5_files/, pt_files/).")
@click.option("--model-name",
              type=click.Choice(["resnet50_trunc", "uni_v1", "conch_v1"]),
              default="resnet50_trunc")
@click.option("--batch-size", type=int, default=256)
@click.option("--num-workers", type=int, default=8)
@click.option("--target-patch-size", type=int, default=224)
@click.option("--no-auto-skip",
              is_flag=True,
              help="Process even if .pt already exists.")
@click.option("--use-amp",
              is_flag=True,
              help="Use autocast on CUDA for speed.")
def main(
    data_h5_dir: Path,
    data_slide_dir: Path,
    slide_ext: str,
    feat_dir: Path,
    model_name: str,
    batch_size: int,
    num_workers: int,
    target_patch_size: int,
    no_auto_skip: bool,
    use_amp: bool,
):
    patches_dir = data_h5_dir / "patches"
    out_h5_dir = feat_dir / "h5_files"
    out_pt_dir = feat_dir / "pt_files"
    out_h5_dir.mkdir(parents=True, exist_ok=True)
    out_pt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tx = get_encoder(model_name, target_img_size=target_patch_size)
    model.eval().to(device)

    # CUDA loader hints
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0
    }

    autocast = (
        torch.amp.autocast("cuda") if  # type: ignore[attr-defined]
        (use_amp and device.type == "cuda") else nullcontext())

    for coords_h5 in tqdm(list(_iter_patch_h5s(patches_dir)), desc="Slides"):
        slide_id = coords_h5.stem  # same as tiling: e.g., "DHMC_0010"
        slide_path = data_slide_dir / f"{slide_id}{slide_ext}"

        out_h5 = out_h5_dir / f"{slide_id}.h5"
        out_pt = out_pt_dir / f"{slide_id}.pt"

        if out_pt.exists() and not no_auto_skip:
            tqdm.write(f"skip {slide_id} (exists)")
            continue

        if not slide_path.exists():
            tqdm.write(f"missing WSI for {slide_id}: {slide_path}")
            continue

        ds = WholeSlidePatchH5(coords_h5_path=coords_h5,
                               wsi_path=slide_path,
                               img_transforms=tx)
        loader = DataLoader(ds,
                            batch_size=batch_size,
                            shuffle=False,
                            **loader_kwargs)

        mode = "w"
        with torch.inference_mode():
            for batch in tqdm(loader,
                              desc=f"Embedding {slide_id}",
                              leave=False):
                imgs = batch["img"].to(device, non_blocking=True)
                coords = batch["coord"].numpy().astype(np.int32)
                with autocast:
                    feats = model(imgs)
                feats = feats.detach().cpu().numpy().astype(np.float32)
                _save_batch_h5(out_h5, feats, coords, mode=mode)
                mode = "a"

        # also save a .pt tensor (compat with older code)

        with h5py.File(out_h5, "r") as f:
            data = {
                "features": torch.from_numpy(get_h5_dataset(f, "features")[:]),
                "coords": torch.from_numpy(get_h5_dataset(f, "coords")[:]),
                "meta": {
                    key: (val.decode() if isinstance(val, bytes) else val)
                    for key, val in f.attrs.items()
                },
            }

        torch.save(data, out_pt)

        tqdm.write(
            f"done {slide_id}: feats {tuple(data['features'].shape)} -> {out_pt}"
        )


if __name__ == "__main__":
    main()
