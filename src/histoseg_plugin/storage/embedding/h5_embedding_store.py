# storage/embedding_h5_store.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from ..interfaces import EmbeddingStore


class H5EmbeddingStore(EmbeddingStore):
    """
    Minimal HDF5-backed embedding store:
      - features: (N, D) float32
      - coords:   (N, 2) int32
      - attrs:    slide-level metadata (f.attrs)
    """

    def __init__(self,
                 *,
                 root_dir: Path,
                 slides_root: Path,
                 features_dir: Path,
                 compression: Optional[str] = None,
                 pt_dir: Optional[Path] = None) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not slides_root.exists():
            raise ValueError(f"slides_root does not exist: {slides_root}")
        self.slides_root = Path(slides_root)
        self.features_dir = self.root_dir / features_dir
        self.features_dir.mkdir(parents=True, exist_ok=True)

        self.comp = compression
        self.pt_dir = Path(pt_dir) if pt_dir else None

    def _path(self, slide_id: str) -> Path:
        return self.root_dir / f"{slide_id}.h5"

    # ---------- write path ----------
    def begin_slide(self, slide_id: str, *, dim: int,
                    attrs: Dict[str, Any]) -> None:
        p = self._path(slide_id)
        with h5py.File(p, "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            f.create_dataset(
                "features",
                shape=(0, dim),
                maxshape=(None, dim),
                chunks=True,
                dtype=np.float32,
                compression=self.comp,
            )
            f.create_dataset(
                "coords",
                shape=(0, 2),
                maxshape=(None, 2),
                chunks=True,
                dtype=np.int32,
                compression=self.comp,
            )

    def append_batch(self, slide_id: str, features: np.ndarray,
                     coords: np.ndarray) -> None:
        p = self._path(slide_id)
        feats = np.asarray(features, dtype=np.float32)
        crds = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        if feats.ndim != 2:
            raise ValueError(f"features must be (N, D); got {feats.shape}")
        if crds.shape[0] != feats.shape[0]:
            raise ValueError("features and coords must have the same N")
        with h5py.File(p, "a") as f:
            df, dc = f["features"], f["coords"]
            n0 = df.shape[0]
            n1 = n0 + feats.shape[0]
            df.resize(n1, axis=0)
            dc.resize(n1, axis=0)
            df[n0:n1] = feats
            dc[n0:n1] = crds

    def finalize_slide(self, slide_id: str) -> Path:
        return self._path(slide_id)

    # ---------- read path ----------
    def load(self,
             slide_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        p = self._path(slide_id)
        with h5py.File(p, "r") as f:
            feats = f["features"][...].astype(np.float32, copy=False)
            coords = f["coords"][...].astype(np.int32, copy=False)
            attrs = {k: f.attrs[k] for k in f.attrs.keys()}
        return feats, coords, attrs

    # ---------- export helpers ----------
    def export_to_pt(self, slide_id: str, pt_dir: Path) -> Path:
        """
        Read H5 for a slide and save a torch .pt bundle:
          {"features": torch.FloatTensor (N,D),
           "coords":   torch.IntTensor   (N,2),
           "meta":     dict}
        """

        feats, coords, attrs = self.load(slide_id)
        import torch  # local import so torch isn't required unless used
        out = pt_dir / f"{slide_id}.pt"
        torch.save(
            {
                "features": torch.from_numpy(feats),
                "coords": torch.from_numpy(coords),
                "meta": attrs,
            },
            out,
        )
        return out
