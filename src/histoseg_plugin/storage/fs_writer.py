# storage/fs_writer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import h5py
import numpy as np

from .interfaces import TilingWriter


class FSTilingWriter(TilingWriter):

    def __init__(
        self,
        coords_dir: Path,
        masks_dir: Path,
        stitches_dir: Path,
        *,
        compression: Optional[str] = None,
        mask_ext: str = ".png",
        stitch_ext: str = ".png",
    ):
        self.coords_dir = Path(coords_dir)
        self.coords_dir.mkdir(parents=True, exist_ok=True)

        self.masks_dir = Path(masks_dir)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        self.stitches_dir = Path(stitches_dir)
        self.stitches_dir.mkdir(parents=True, exist_ok=True)

        self.comp = compression
        self.mask_ext = mask_ext
        self.stitch_ext = stitch_ext

    def _coords_path(self, slide_id: str) -> Path:
        return self.coords_dir / f"{slide_id}.h5"

    def save_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        attrs: Dict[str, Any],
        cont_idx: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Create a new HDF5 file with /coords (Nx2) and /cont_idx (N,).
        If cont_idx is None, it will be created and filled with -1.
        """
        out = self._coords_path(slide_id)
        tmp = out.with_suffix(out.suffix + ".part")

        coords = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        if cont_idx is None:
            cont_idx = np.full((coords.shape[0], ), -1, dtype=np.int32)
        else:
            cont_idx = np.asarray(cont_idx, dtype=np.int32).reshape(-1)

        if cont_idx.shape[0] != coords.shape[0]:
            raise ValueError("cont_idx length must match coords rows.")

        with h5py.File(tmp, "w") as f:
            d = f.create_dataset(
                "coords",
                data=coords,
                maxshape=(None, 2),
                chunks=True,
                compression=self.comp,
            )
            ci = f.create_dataset(
                "cont_idx",
                data=cont_idx,
                maxshape=(None, ),
                chunks=True,
                compression=self.comp,
            )
            # store flat attrs on the coords dataset
            for k, v in attrs.items():
                d.attrs[k] = v

        tmp.replace(out)
        return out

    def append_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        cont_idx: Optional[np.ndarray] = None,
    ) -> None:
        """
        Append rows to /coords and /cont_idx.
        If file does not exist yet, creates it with empty attrs.
        If cont_idx is None, appended rows get -1.
        """
        out = self._coords_path(slide_id)

        coords = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        if cont_idx is None:
            cont_idx = np.full((coords.shape[0], ), -1, dtype=np.int32)
        else:
            cont_idx = np.asarray(cont_idx, dtype=np.int32).reshape(-1)

        if coords.shape[0] == 0:
            return

        if cont_idx.shape[0] != coords.shape[0]:
            raise ValueError("cont_idx length must match coords rows.")

        if not out.exists():
            # First write: empty attrs since we don't have them here.
            self.save_coords(slide_id, coords, attrs={}, cont_idx=cont_idx)
            return

        with h5py.File(out, "a") as f:
            d = f["coords"]
            ci = f["cont_idx"]
            old_n = d.shape[0]
            new_n = old_n + coords.shape[0]
            d.resize((new_n, 2))
            ci.resize((new_n, ))
            d[old_n:new_n, :] = coords
            ci[old_n:new_n] = cont_idx

    def save_mask(self, slide_id: str, image: Image.Image) -> Path:
        out = self.masks_dir / f"{slide_id}{self.mask_ext}"
        image.save(out)
        return out

    def save_stitch(self, slide_id: str, image: Image.Image) -> Path:
        out = self.stitches_dir / f"{slide_id}{self.stitch_ext}"
        image.save(out)
        return out
