from pathlib import Path

from .interfaces import TilingStore
from .specs import TilingStoresSpec
from .interfaces import EmbeddingStore
from .embedding_store import H5EmbeddingStore, PtEmbeddingSink, MultiEmbeddingStore


def build_tiling_store(spec: TilingStoresSpec) -> TilingStore:
    coords_dir = Path(spec.coords_dir)
    masks_dir = Path(spec.masks_dir)
    stitches_dir = Path(spec.stitches_dir)
    kind = spec.coords_kind.lower()

    if kind == "h5":
        from .tiling.h5_tiling_store import H5TilingStore
        return H5TilingStore(coords_dir=coords_dir,
                             masks_dir=masks_dir,
                             stitches_dir=stitches_dir,
                             compression=getattr(spec.compression,
                                                 "compression", None),
                             mask_ext=spec.mask_ext,
                             stitch_ext=spec.stitch_ext)
    elif kind == "json":
        from .tiling.json_tiling_store import JSONTilingStore
        return JSONTilingStore(coords_dir=coords_dir,
                               masks_dir=masks_dir,
                               stitches_dir=stitches_dir,
                               mask_ext=spec.mask_ext,
                               stitch_ext=spec.stitch_ext)
    else:
        raise ValueError(f"Unknown coords kind: {kind}")


# def build_embedding_store(spec) -> EmbeddingStore:
#     sinks_cfg = spec.embedding.features.sinks
#     primary = None
#     extras = []
#     for s in sinks_cfg:
#         kind = s.kind.lower()
#         if kind == "h5":
#             store = H5EmbeddingStore(dir=s.dir,
#                                      compression=getattr(
#                                          s, "compression", None))
#         elif kind == "pt":
#             store = PtEmbeddingSink(dir=s.dir, source_h5_dir=s.source_h5_dir)
#         else:
#             raise ValueError(f"Unknown embedding sink kind: {kind}")

#         if getattr(s, "primary", False) or primary is None:
#             primary = store
#         else:
#             extras.append(store)
#     assert primary is not None, "At least one embedding sink (primary) is required"
#     return MultiEmbeddingStore(primary, extras)
