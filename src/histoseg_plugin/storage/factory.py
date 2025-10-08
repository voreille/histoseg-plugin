from .h5_store import H5CoordsStore, H5FeatureStore
from .config import CoordsStoreConfig, FeatureStoreConfig

def build_coords_store(cfg: CoordsStoreConfig):
    if cfg.kind == "h5":
        return H5CoordsStore(cfg.path)
    raise ValueError(f"Unknown coords store kind: {cfg.kind}")

def build_feature_store(cfg: FeatureStoreConfig):
    if cfg.kind == "h5":
        return H5FeatureStore(cfg.path, compression=cfg.compression)
    raise ValueError(f"Unknown feature store kind: {cfg.kind}")
