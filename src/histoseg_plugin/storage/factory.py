from pathlib import Path

from .tiling_writer import FSTilingWriter, JSONTilingWriter, TilingWriter
from .specs import TilingStoresSpec


def build_tiling_writer(spec: TilingStoresSpec) -> TilingWriter:
    coords_dir = Path(spec.coords_dir)
    masks_dir = Path(spec.masks_dir)
    stitches_dir = Path(spec.stitches_dir)
    kind = spec.coords_kind.lower()

    if kind == "h5":
        return FSTilingWriter(coords_dir=coords_dir,
                              masks_dir=masks_dir,
                              stitches_dir=stitches_dir,
                              compression=getattr(spec.compression,
                                                  "compression", None))
    elif kind == "json":
        return JSONTilingWriter(coords_dir=coords_dir,
                                masks_dir=masks_dir,
                                stitches_dir=stitches_dir)
    else:
        raise ValueError(f"Unknown coords kind: {kind}")
