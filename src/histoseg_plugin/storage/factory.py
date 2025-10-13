from pathlib import Path

from .interfaces import TilingStore
from .specs import TilingStoresSpec, EmbeddingStoresSpec
from .interfaces import EmbeddingStore


def build_tiling_store(
    *,
    spec: TilingStoresSpec,
    root_dir: Path,
    slides_root: Path,
) -> TilingStore:
    root_dir = Path(root_dir)
    coords_dir = Path(spec.coords_dir)
    masks_dir = Path(spec.masks_dir)
    stitches_dir = Path(spec.stitches_dir)
    kind = spec.coords_kind.lower()

    if kind == "h5":
        from .tiling.h5_tiling_store import H5TilingStore
        return H5TilingStore(root_dir=root_dir,
                             slides_root=slides_root,
                             coords_dir=coords_dir,
                             masks_dir=masks_dir,
                             stitches_dir=stitches_dir,
                             compression=getattr(spec.compression,
                                                 "compression", None),
                             mask_ext=spec.mask_ext,
                             stitch_ext=spec.stitch_ext)
    elif kind == "json":
        from .tiling.json_tiling_store import JSONTilingStore
        return JSONTilingStore(root_dir=root_dir,
                               slides_root=slides_root,
                               coords_dir=coords_dir,
                               masks_dir=masks_dir,
                               stitches_dir=stitches_dir,
                               mask_ext=spec.mask_ext,
                               stitch_ext=spec.stitch_ext)
    else:
        raise ValueError(f"Unknown coords kind: {kind}")


def build_tiling_store_from_dir(
    *,
    root_dir: Path,
    slides_root: Path,
) -> TilingStore:
    spec = TilingStoresSpec.from_json(root_dir / ".tiling_store.json")
    return build_tiling_store(spec=spec,
                              root_dir=root_dir,
                              slides_root=slides_root)


def build_embedding_store(
    *,
    spec: EmbeddingStoresSpec,
    slides_root: Path,
    root_dir: Path,
) -> EmbeddingStore:
    if spec.kind.lower() == "h5":
        from .embedding.h5_embedding_store import H5EmbeddingStore
        return H5EmbeddingStore(
            root_dir=root_dir,
            features_dir=spec.features_dir,
            slides_root=slides_root,
            compression=getattr(spec.compression, "compression", None),
            pt_dir=spec.pt_dir,
        )

    else:
        raise ValueError(f"Unknown embedding kind: {spec.kind}")
