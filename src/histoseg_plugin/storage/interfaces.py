# storage/interfaces.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import numpy as np
from PIL import Image


class TilingWriter(Protocol):

    def save_coords(self, slide_id: str, coords: np.ndarray,
                    attrs: Dict[str, Any], cont_idx: Optional[np.ndarray]) -> Path:
        ...

    def append_coords(self,
                      slide_id: str,
                      coords: np.ndarray,
                      cont_idx: Optional[np.ndarray] = None) -> None:
        ...

    def save_mask(self, slide_id: str, image: Image.Image) -> Path:
        ...

    def save_stitch(self, slide_id: str, image: Image.Image) -> Path:
        ...

    def load_coords(self, slide_id: str) -> np.ndarray:
        ...
