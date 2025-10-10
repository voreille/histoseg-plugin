from pathlib import Path

from .tiling_store import H5TilingStore, JSONTilingStore, TilingStore
from .specs import TilingStoresSpec


def build_tiling_store(spec: TilingStoresSpec) -> TilingStore:
    coords_dir = Path(spec.coords_dir)
    masks_dir = Path(spec.masks_dir)
    stitches_dir = Path(spec.stitches_dir)
    kind = spec.coords_kind.lower()

    if kind == "h5":
        return H5TilingStore(coords_dir=coords_dir,
                             masks_dir=masks_dir,
                             stitches_dir=stitches_dir,
                             compression=getattr(spec.compression,
                                                 "compression", None),
                             mask_ext=spec.mask_ext,
                             stitch_ext=spec.stitch_ext)
    elif kind == "json":
        return JSONTilingStore(coords_dir=coords_dir,
                               masks_dir=masks_dir,
                               stitches_dir=stitches_dir,
                               mask_ext=spec.mask_ext,
                               stitch_ext=spec.stitch_ext)
    else:
        raise ValueError(f"Unknown coords kind: {kind}")
