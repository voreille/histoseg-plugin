# storage/embedding_h5_store.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from .interfaces import EmbeddingStore


class MultiEmbeddingStore(EmbeddingStore):

    def __init__(self, primary: EmbeddingStore,
                 extras: List[EmbeddingStore]) -> None:
        self.primary = primary
        self.extras = extras

    def begin_slide(self, slide_id: str, *, dim: int,
                    attrs: Dict[str, Any]) -> None:
        self.primary.begin_slide(slide_id, dim=dim, attrs=attrs)
        for s in self.extras:
            # sinks like PT don't need this, but harmless if implemented
            if hasattr(s, "begin_slide"):
                s.begin_slide(slide_id, dim=dim, attrs=attrs)

    def append_batch(self, slide_id: str, features: np.ndarray,
                     coords: np.ndarray) -> None:
        self.primary.append_batch(slide_id, features, coords)

    def finalize_slide(self, slide_id: str) -> Path:
        p = self.primary.finalize_slide(slide_id)
        feats, coords, attrs = self.primary.load(slide_id)
        for s in self.extras:
            s.finalize_slide(slide_id)
        return p


class H5EmbeddingStore(EmbeddingStore):

    def __init__(self,
                 dir: Path,
                 *,
                 compression: Optional[str] = None) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.comp = compression

    def _path(self, slide_id: str) -> Path:
        return self.dir / f"{slide_id}.h5"

    def begin_slide(self, slide_id: str, *, dim: int,
                    attrs: Dict[str, Any]) -> None:
        p = self._path(slide_id)
        mode = "w"
        with h5py.File(p, mode) as f:
            f.attrs["version"] = int(attrs.get("version", 1))
            for k, v in attrs.items():
                f.attrs[k] = v
            f.create_dataset("features",
                             shape=(0, dim),
                             maxshape=(None, dim),
                             chunks=True,
                             dtype=np.float32,
                             compression=self.comp)
            f.create_dataset("coords",
                             shape=(0, 2),
                             maxshape=(None, 2),
                             chunks=True,
                             dtype=np.int32,
                             compression=self.comp)

    def append_batch(self, slide_id: str, features: np.ndarray,
                     coords: np.ndarray) -> None:
        p = self._path(slide_id)
        feats = np.asarray(features,
                           dtype=np.float32).reshape(-1, features.shape[-1])
        crds = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        if feats.shape[0] != crds.shape[0]:
            raise ValueError("features and coords must have same N")
        with h5py.File(p, "a") as f:
            df = f["features"]
            dc = f["coords"]
            n0 = df.shape[0]
            n1 = n0 + feats.shape[0]
            df.resize(n1, axis=0)
            dc.resize(n1, axis=0)
            df[n0:n1] = feats
            dc[n0:n1] = crds

    def finalize_slide(self, slide_id: str) -> Path:
        return self._path(slide_id)

    def load(self,
             slide_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        p = self._path(slide_id)
        with h5py.File(p, "r") as f:
            feats = f["features"][...].astype(np.float32)
            coords = f["coords"][...].astype(np.int32)
            attrs = {k: f.attrs[k] for k in f.attrs.keys()}
        return feats, coords, attrs


class PtEmbeddingSink(EmbeddingStore):
    """
    Not appendable. Implemented as a finalize-only sink that expects
    an upstream H5EmbeddingStore to have produced an H5 file already.
    """

    def __init__(self, dir: Path, *, source_h5_dir: Path) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.source_h5_dir = Path(source_h5_dir)

    # no-op for streaming
    def begin_slide(self, slide_id: str, *, dim: int,
                    attrs: Dict[str, Any]) -> None:
        ...

    def append_batch(self, slide_id: str, features, coords) -> None:
        ...

    def finalize_slide(self, slide_id: str) -> Path:
        src = self.source_h5_dir / f"{slide_id}.h5"
        out = self.dir / f"{slide_id}.pt"
        with h5py.File(src, "r") as f:
            data = {
                "features": torch.from_numpy(f["features"][...]),
                "coords": torch.from_numpy(f["coords"][...]),
                "meta": {
                    k: (v.decode() if isinstance(v, bytes) else v)
                    for k, v in f.attrs.items()
                },
            }
        torch.save(data, out)
        return out
