# storage/interfaces.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple
import numpy as np
from PIL import Image


class TilingStore(Protocol):
    def save_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        attrs: Dict[str, Any],
        cont_idx: Optional[np.ndarray] = None,
        *,
        overwrite: bool = True,
    ) -> Path:
        ...

    def append_coords(
        self,
        slide_id: str,
        coords: np.ndarray,
        cont_idx: Optional[np.ndarray] = None,
    ) -> None:
        ...

    def load_coords(
        self,
        slide_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        ...

    def save_mask(self, slide_id: str, image: Image.Image) -> Path:
        ...

    def save_stitch(self, slide_id: str, image: Image.Image) -> Path:
        ...

    def coords_path(self, slide_id: str) -> Path:
        ...
