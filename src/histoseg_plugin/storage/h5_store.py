from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional
import os

import h5py
import numpy as np

from .interfaces import CoordsStore, FeatureStore


class H5CoordsStore(CoordsStore):
    """Tiling output reader: expects one H5 per slide with a 'coords' dataset and attrs."""

    def __init__(self, patches_dir: str | Path):
        self.root = Path(patches_dir)

    def _path(self, slide_id: str) -> Path:
        return self.root / f"{slide_id}.h5"

    def list_slides(self) -> Iterable[str]:
        for p in sorted(self.root.glob("*.h5")):
            yield p.stem

    def has_slide(self, slide_id: str) -> bool:
        return self._path(slide_id).exists()

    def coords_len(self, slide_id: str) -> int:
        with h5py.File(self._path(slide_id), "r") as f:
            return len(f["coords"])

    def get_coords(self, slide_id: str) -> np.ndarray:
        with h5py.File(self._path(slide_id), "r") as f:
            d = f["coords"]  # type: ignore[index]
            return d[:]  # (N,2) int

    def get_attrs(self, slide_id: str) -> Dict[str, Any]:
        with h5py.File(self._path(slide_id), "r") as f:
            d = f["coords"]  # type: ignore[index]
            return {k: v for k, v in d.attrs.items()}

    def get_coord_at(self, slide_id: str, idx: int) -> Tuple[int, int]:
        with h5py.File(self._path(slide_id), "r") as f:
            d = f["coords"]  # type: ignore[index]
            c = d[idx]
            return int(c[0]), int(c[1])


class H5FeatureStore(FeatureStore):
    """Features H5: datasets 'features' (N,D), 'coords' (N,2) and root attrs."""

    def __init__(self,
                 features_dir: str | Path,
                 compression: Optional[str] = None):
        self.root = Path(features_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.compression = compression  # e.g. "lzf" or "gzip"

    def _path(self, slide_id: str) -> Path:
        return self.root / f"{slide_id}.h5"

    def has_slide(self, slide_id: str) -> bool:
        return self._path(slide_id).exists()

    def create(self,
               slide_id: str,
               feature_dim: int,
               *,
               dtype: str = "float32",
               attrs: Optional[Dict[str, Any]] = None) -> None:
        p = self._path(slide_id)
        if p.exists():
            p.unlink()  # start fresh
        with h5py.File(p, "w") as f:
            maxshape = (None, feature_dim)
            f.create_dataset("features",
                             shape=(0, feature_dim),
                             maxshape=maxshape,
                             dtype=dtype,
                             chunks=True,
                             compression=self.compression)
            f.create_dataset("coords",
                             shape=(0, 2),
                             maxshape=(None, 2),
                             dtype="i4",
                             chunks=True,
                             compression=self.compression)
            if attrs:
                for k, v in attrs.items():
                    f.attrs[k] = v

    def append(self, slide_id: str, feats: np.ndarray,
               coords: np.ndarray) -> None:
        assert feats.ndim == 2 and coords.ndim == 2 and coords.shape[1] == 2
        p = self._path(slide_id)
        with h5py.File(p, "a") as f:
            df = f["features"]
            dc = f["coords"]  # type: ignore[index]
            n0 = df.shape[0]
            n = feats.shape[0]
            df.resize(n0 + n, axis=0)
            df[n0:n0 + n] = feats
            dc.resize(n0 + n, axis=0)
            dc[n0:n0 + n] = coords

    def read_features(self, slide_id: str) -> np.ndarray:
        with h5py.File(self._path(slide_id), "r") as f:
            return f["features"][:]  # type: ignore[index]

    def read_coords(self, slide_id: str) -> np.ndarray:
        with h5py.File(self._path(slide_id), "r") as f:
            return f["coords"][:]  # type: ignore[index]

    def read_attrs(self, slide_id: str) -> Dict[str, Any]:
        with h5py.File(self._path(slide_id), "r") as f:
            return {k: v for k, v in f.attrs.items()}

    def set_attrs(self, slide_id: str, attrs: Dict[str, Any]) -> None:
        with h5py.File(self._path(slide_id), "a") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
