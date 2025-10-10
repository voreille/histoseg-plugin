# storage/specs.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import StorageConfig


@dataclass(frozen=True)
class TilingStoresSpec:
    coords_dir: Path
    masks_dir: Path
    stitches_dir: Path
    compression: Optional[str] = None
    mask_ext: str = ".png"
    stitch_ext: str = ".png"

    @classmethod
    def from_config(cls, cfg: StorageConfig) -> "TilingStoresSpec":
        return cls(
            coords_dir=cfg.tiling.coords.dir,
            masks_dir=cfg.tiling.masks.dir,
            stitches_dir=cfg.tiling.stitches.dir,
            compression=cfg.tiling.coords.compression,
            mask_ext=cfg.tiling.masks.extension or ".png",
            stitch_ext=cfg.tiling.stitches.extension or ".png",
        )

