from .fs_writer import FSTilingWriter
from .specs import TilingStoresSpec


def build_tiling_writer(spec: TilingStoresSpec) -> FSTilingWriter:
    return FSTilingWriter(
        coords_dir=spec.coords_dir,
        masks_dir=spec.masks_dir,
        stitches_dir=spec.stitches_dir,
        compression=spec.compression,
        mask_ext=spec.mask_ext,
        stitch_ext=spec.stitch_ext,
    )
