from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import h5py
import openslide
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from histoseg_plugin.utils.h5_utils import get_dataset, read_attrs


class SlideLike(Protocol):
    """Minimal interface for slide readers (OpenSlide, ImageSlide, or custom)."""

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]) -> Image.Image:
        ...

    def close(self) -> None:
        ...


class WholeSlidePatchH5(Dataset):
    """
    Dataset reading patch coordinates from an HDF5 produced by tiling,
    and extracting RGB tiles from the original WSI via OpenSlide.

    - Lazily opens the HDF5 and WSI files only when needed.
    - Safe for multiprocessing: each worker reopens its own handles.
    """

    def __init__(
        self,
        coords_h5_path: Path | str,
        wsi_path: Path | str,
        img_transforms: Optional[T.Compose] = None,
    ):
        self.coords_h5_path = str(coords_h5_path)
        self.wsi_path = str(wsi_path)
        self.transform = img_transforms

        self._h5: Optional[h5py.File] = None
        self._wsi: Optional[SlideLike] = None
        self._pid: Optional[int] = None  # used to detect forked workers

        # read minimal metadata once
        with h5py.File(self.coords_h5_path, "r") as f:
            dset = get_dataset(f, "coords")
            attrs = read_attrs(f, "coords")
            self.length = len(dset)
            self.patch_level = int(attrs["patch_level"])
            self.patch_size = int(attrs["patch_size"])

    # ------------------------------------------------------------------
    # Lazy handles (HDF5 + OpenSlide)
    # ------------------------------------------------------------------

    def _reopen_if_needed(self) -> None:
        """Close and reopen handles if the process has changed (fork-safe)."""
        cur_pid = os.getpid()
        if self._pid != cur_pid:
            print("I am reopening if needed, cause current pid is", os.getpid())
            print("and my stored pid is", self._pid)
            self._close_handles()
            self._pid = cur_pid

    def _close_handles(self) -> None:
        """Gracefully close any open handles."""
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
        try:
            if self._wsi is not None:
                self._wsi.close()
        except Exception:
            pass
        self._h5 = None
        self._wsi = None

    @property
    def h5(self) -> h5py.File:
        """Lazily open and return the HDF5 file handle."""
        self._reopen_if_needed()
        if self._h5 is None:
            self._h5 = h5py.File(self.coords_h5_path, "r", swmr=True)
        return self._h5

    @property
    def wsi(self) -> SlideLike:
        """Lazily open and return the OpenSlide handle."""
        self._reopen_if_needed()
        if self._wsi is None:
            self._wsi = openslide.open_slide(self.wsi_path)
        return self._wsi

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dset = get_dataset(self.h5, "coords")
        coords = dset[idx]

        # read_region expects (x, y) in level coordinates
        img: Image.Image = self.wsi.read_region(
            (int(coords[0]), int(coords[1])),
            self.patch_level,
            (self.patch_size, self.patch_size),
        ).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return {"img": img, "coord": coords}

    def __del__(self) -> None:
        """Close files when object is garbage-collected."""
        self._close_handles()
