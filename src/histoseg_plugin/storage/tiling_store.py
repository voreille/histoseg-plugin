from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import h5py
import numpy as np
from PIL import Image

from .interfaces import TilingStore

# -------------------------
# Helpers
# -------------------------

_EXT_TO_FORMAT = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
    ".webp": "WEBP",
}


def _pil_format_for_ext(ext: str) -> str:
    fmt = _EXT_TO_FORMAT.get(ext.lower())
    if not fmt:
        raise ValueError(f"Unsupported image extension: {ext}")
    return fmt


def _ensure_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
    return coords


def _ensure_cont_idx(cont_idx: Optional[np.ndarray], n: int) -> np.ndarray:
    if cont_idx is None:
        return np.full((n, ), -1, dtype=np.int32)
    cont_idx = np.asarray(cont_idx, dtype=np.int32).reshape(-1)
    if cont_idx.shape[0] != n:
        raise ValueError("cont_idx length must match coords rows.")
    return cont_idx


# ============================================================
# JSON store (coords as JSONL + meta.json; masks/stitches as images)
# ============================================================


class JSONTilingStore(TilingStore):
    """
    Writes coords as JSON Lines (NDJSON) for easy appends + a separate meta file.
    Files:
      <dir>/<slide_id>.coords.jsonl   # one coord per line: {"x":..,"y":..,"cont_idx":..}
      <dir>/<slide_id>.meta.json      # attrs dict (contract)
      masks/ and stitches/ are standard images (atomic write with .part).
    """

    def __init__(
        self,
        coords_dir: Path,
        masks_dir: Path,
        stitches_dir: Path,
        *,
        mask_ext: str = ".png",
        stitch_ext: str = ".png",
    ) -> None:
        self.coords_dir = Path(coords_dir)
        self.coords_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir = Path(masks_dir)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.stitches_dir = Path(stitches_dir)
        self.stitches_dir.mkdir(parents=True, exist_ok=True)
        self.mask_ext = mask_ext
        self.stitch_ext = stitch_ext

    # ---- paths ----
    def _paths(self, slide_id: str) -> Tuple[Path, Path]:
        base = self.coords_dir / slide_id
        return base.with_suffix(".coords.jsonl"), base.with_suffix(
            ".meta.json")

    def coords_path(self, slide_id: str) -> Path:
        base = self.coords_dir / slide_id
        return base.with_suffix(".coords.jsonl")

    # ---- coords I/O ----
    def save_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        attrs: Dict[str, Any],
        cont_idx: Optional[np.ndarray] = None,
        *,
        overwrite: bool = True,
    ) -> Path:
        coords = _ensure_coords(coords)
        cont_idx = _ensure_cont_idx(cont_idx, coords.shape[0])

        coords_path, meta_path = self._paths(slide_id)
        if coords_path.exists() and not overwrite:
            raise FileExistsError(
                f"coords file exists and overwrite=False: {coords_path}")

        # meta (atomic)
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".part")
        tmp_meta.write_text(json.dumps(attrs, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        tmp_meta.replace(meta_path)

        # coords (atomic create)
        tmp_coords = coords_path.with_suffix(coords_path.suffix + ".part")
        with tmp_coords.open("w", encoding="utf-8") as f:
            for (x, y), ci in zip(coords.tolist(), cont_idx.tolist()):
                f.write(
                    json.dumps({
                        "x": int(x),
                        "y": int(y),
                        "cont_idx": int(ci)
                    }) + "\n")
        tmp_coords.replace(coords_path)
        return coords_path

    def append_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        cont_idx: Optional[np.ndarray] = None,
    ) -> None:
        coords_path, _ = self._paths(slide_id)
        coords = _ensure_coords(coords)
        if coords.shape[0] == 0:
            return
        cont_idx = _ensure_cont_idx(cont_idx, coords.shape[0])

        coords_path.parent.mkdir(parents=True, exist_ok=True)
        with coords_path.open("a", encoding="utf-8") as f:
            for (x, y), ci in zip(coords.tolist(), cont_idx.tolist()):
                f.write(
                    json.dumps({
                        "x": int(x),
                        "y": int(y),
                        "cont_idx": int(ci)
                    }) + "\n")

    def load_coords(
            self,
            slide_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        coords_path, meta_path = self._paths(slide_id)
        if not coords_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"coords/meta missing for slide_id={slide_id} in {self.coords_dir}"
            )
        attrs = json.loads(meta_path.read_text(encoding="utf-8"))
        xs, ys, cis = [], [], []
        with coords_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                xs.append(int(obj["x"]))
                ys.append(int(obj["y"]))
                cis.append(int(obj.get("cont_idx", -1)))
        coords = np.asanyarray(list(zip(xs, ys)), dtype=np.int32)
        cont_idx = np.asanyarray(cis, dtype=np.int32)
        return coords, cont_idx, attrs

    # ---- images ----
    def save_mask(self, slide_id: str, image: Image.Image) -> Path:
        out = self.masks_dir / f"{slide_id}{self.mask_ext}"
        tmp = out.with_suffix(out.suffix + ".part")
        fmt = _pil_format_for_ext(self.mask_ext)
        image.save(tmp, format=fmt)
        tmp.replace(out)
        return out

    def save_stitch(self, slide_id: str, image: Image.Image) -> Path:
        out = self.stitches_dir / f"{slide_id}{self.stitch_ext}"
        tmp = out.with_suffix(out.suffix + ".part")
        fmt = _pil_format_for_ext(self.stitch_ext)
        image.save(tmp, format=fmt)
        tmp.replace(out)
        return out


# ============================================================
# HDF5 store (coords/cont_idx in HDF5; masks/stitches as images)
# ============================================================


class H5TilingStore(TilingStore):
    """
    HDF5-backed tiling store.
    - /coords (Nx2) int32
    - /cont_idx (N,) int32  (filled with -1 if unknown)
    - attrs: stored on the /coords dataset
    """

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

    def coords_path(self, slide_id: str) -> Path:
        return self.coords_dir / f"{slide_id}.h5"

    # ---- coords I/O ----
    def save_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        attrs: Dict[str, Any],
        cont_idx: Optional[np.ndarray] = None,
        *,
        overwrite: bool = True,
    ) -> Path:
        out = self.coords_path(slide_id)
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"coords file exists and overwrite=False: {out}")

        tmp = out.with_suffix(out.suffix + ".part")

        coords = _ensure_coords(coords)
        cont_idx = _ensure_cont_idx(cont_idx, coords.shape[0])

        with h5py.File(tmp, "w") as f:
            d = f.create_dataset(
                "coords",
                data=coords,
                maxshape=(None, 2),
                chunks=True,
                compression=self.comp,
            )
            f.create_dataset(
                "cont_idx",
                data=cont_idx,
                maxshape=(None, ),
                chunks=True,
                compression=self.comp,
            )
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
        out = self.coords_path(slide_id)

        coords = _ensure_coords(coords)
        if coords.shape[0] == 0:
            return
        cont_idx = _ensure_cont_idx(cont_idx, coords.shape[0])

        if not out.exists():
            # Create with empty attrs if appending first
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

    def load_coords(
            self,
            slide_id: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        path = self.coords_path(slide_id)
        if not path.exists():
            raise FileNotFoundError(f"coords file not found: {path}")

        with h5py.File(path, "r") as f:
            d = f["coords"]

            # Force concrete ndarrays (avoid Dataset | AsTypeView unions)
            coords = np.asarray(d[...],
                                dtype=np.int32)  # type: ignore[no-any-return]
            coords = cast(np.ndarray, coords)

            # attrs as plain dict[str, Any]
            attrs: Dict[str, Any] = {}
            for k in d.attrs.keys():
                v = d.attrs[k]
                # normalize h5py/numpy scalars to Python types (optional, but nice)
                if isinstance(v, np.generic):
                    attrs[k] = v.item()
                elif isinstance(v, np.ndarray):
                    attrs[k] = v.tolist()
                else:
                    attrs[k] = v

            if "cont_idx" in f:
                cont = f["cont_idx"][...]
                cont_idx = np.asarray(
                    cont, dtype=np.int32)  # type: ignore[no-any-return]
            else:
                cont_idx = np.full((coords.shape[0], ), -1, dtype=np.int32)

            cont_idx = cast(np.ndarray, cont_idx)

        return coords, cont_idx, attrs

    # ---- images ----
    def save_mask(self, slide_id: str, image: Image.Image) -> Path:
        out = self.masks_dir / f"{slide_id}{self.mask_ext}"
        tmp = out.with_suffix(out.suffix + ".part")
        fmt = _pil_format_for_ext(self.mask_ext)
        image.save(tmp, format=fmt)
        tmp.replace(out)
        return out

    def save_stitch(self, slide_id: str, image: Image.Image) -> Path:
        out = self.stitches_dir / f"{slide_id}{self.stitch_ext}"
        tmp = out.with_suffix(out.suffix + ".part")
        fmt = _pil_format_for_ext(self.stitch_ext)
        image.save(tmp, format=fmt)
        tmp.replace(out)
        return out
