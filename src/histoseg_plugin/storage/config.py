from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import Optional, Literal, Union
import yaml


class StoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["h5", "files", "json"] = "files"
    dir: Path
    compression: Optional[str] = None  # h5-only
    extension: Optional[str] = None  # files-only

    def resolve(self, base: Path) -> "StoreConfig":
        d = self.dir if self.dir.is_absolute() else base / self.dir
        return self.model_copy(update={"dir": d})


class TilingStores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    coords: StoreConfig
    masks: StoreConfig
    stitches: StoreConfig


class EmbeddingStores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: StoreConfig


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tiling: TilingStores
    embedding: EmbeddingStores

    def resolve_paths(self, base: Path) -> "StorageConfig":
        return StorageConfig(
            tiling=TilingStores(
                coords=self.tiling.coords.resolve(base),
                masks=self.tiling.masks.resolve(base),
                stitches=self.tiling.stitches.resolve(base),
            ),
            embedding=EmbeddingStores(
                features=self.embedding.features.resolve(base), ),
        )

    @classmethod
    def from_yaml(cls,
                  path: Union[str, Path],
                  root_key: Union[str, None] = None) -> "StorageConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        if root_key:
            data = data.get(root_key, {}) or {}
        return cls.model_validate(data)
