# storage/specs.py
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Union

from .config import EmbeddingStoreConfig, TilingStoreConfig


@dataclass(frozen=True)
class TilingStoresSpec:
    coords_dir: Path
    masks_dir: Path
    stitches_dir: Path
    coords_kind: str = "h5"
    compression: Optional[str] = None
    mask_ext: str = ".png"
    stitch_ext: str = ".png"

    @classmethod
    def from_yaml(cls,
                  *,
                  path: Union[str, Path],
                  root_key: Union[str, None] = "tiling") -> "TilingStoresSpec":
        return cls.from_config(TilingStoreConfig.from_yaml(path, root_key))

    @classmethod
    def from_config(cls, cfg: TilingStoreConfig) -> "TilingStoresSpec":
        return cls(
            coords_dir=cfg.coords.dir,
            coords_kind=cfg.coords.kind,
            masks_dir=cfg.masks.dir,
            stitches_dir=cfg.stitches.dir,
            compression=cfg.coords.compression,
            mask_ext=cfg.masks.extension or ".png",
            stitch_ext=cfg.stitches.extension or ".png",
        )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        data = asdict(self)
        for k, v in data.items():
            if isinstance(v, Path):
                data[k] = str(v)
        return data

    def to_json(self, path: Path) -> None:
        """Dump spec to a JSON manifest (e.g., .tiling_store.json)."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "TilingStoresSpec":
        """Rebuild a spec from JSON manifest."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # Convert string paths back to Path
        for f_ in fields(cls):
            if f_.type is Path or f_.type == Optional[Path]:
                if data.get(f_.name) is not None:
                    data[f_.name] = Path(data[f_.name])
        return cls(**data)


@dataclass(frozen=True)
class EmbeddingStoresSpec:
    features_dir: Path
    kind: str = "h5"
    compression: Optional[str] = None
    pt_dir: Optional[Path] = None
    enable_pt_export: bool = False

    @classmethod
    def from_config(cls, cfg: EmbeddingStoreConfig) -> "EmbeddingStoresSpec":
        return cls(
            features_dir=cfg.features.dir,
            kind=cfg.features.kind,
            compression=cfg.features.compression,
            pt_dir=cfg.export_pt.dir,
            enable_pt_export=cfg.export_pt.enabled,
        )

    @classmethod
    def from_yaml(cls,
                  *,
                  path: Union[str, Path],
                  root_key: Union[str,
                                  None] = "tiling") -> "EmbeddingStoresSpec":
        return cls.from_config(EmbeddingStoreConfig.from_yaml(path, root_key))
