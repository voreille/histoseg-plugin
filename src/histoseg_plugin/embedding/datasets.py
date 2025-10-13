from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import openslide
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class SlideLike(Protocol):
    """Minimal interface for slide readers (OpenSlide, ImageSlide, or custom)."""

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]) -> Image.Image:
        ...

    def close(self) -> None:
        ...


class WholeSlidePatch(Dataset):
    """
    Dataset reading patch coordinates from an HDF5 produced by tiling,
    and extracting RGB tiles from the original WSI via OpenSlide.

    - Lazily opens the HDF5 and WSI files only when needed.
    - Safe for multiprocessing: each worker reopens its own handles.
    """

    def __init__(
        self,
        coords: np.ndarray,
        tile_level: int,
        tile_size: int,
        wsi_path: Path | str,
        transform: Optional[T.Compose] = None,
    ):
        self.coords = coords
        self.length = len(coords)
        self.tile_level = tile_level
        self.tile_size = tile_size

        self.wsi_path = str(wsi_path)
        self.transform = transform

        self._wsi: Optional[SlideLike] = None
        self._pid: Optional[int] = None  # used to detect forked workers

    def _reopen_if_needed(self) -> None:
        """Close and reopen handles if the process has changed (fork-safe)."""
        cur_pid = os.getpid()
        if self._pid != cur_pid:
            self._close_handles()
            self._pid = cur_pid

    def _close_handles(self) -> None:
        """Gracefully close any open handles."""
        try:
            if self._wsi is not None:
                self._wsi.close()
        except Exception:
            pass
        self._wsi = None

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
        tile_coords = self.coords[idx]
        # read_region expects (x, y) in level coordinates
        img: Image.Image = self.wsi.read_region(
            (int(tile_coords[0]), int(tile_coords[1])),
            self.tile_level,
            (self.tile_size, self.tile_size),
        ).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return {"img": img, "coord": tile_coords}

    def __del__(self) -> None:
        """Close files when object is garbage-collected."""
        self._close_handles()
