from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml
from pydantic import ValidationError

from .parameter_models import TilingConfig

# -------------------------
# Small helpers
# -------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str,
                                                     Any]) -> Dict[str, Any]:
    """Recursively merge dict 'override' into dict 'base' (non-mutating)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path | str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _extract_root(data: Mapping[str, Any],
                  root_key: str | None) -> Dict[str, Any]:
    """Return the subdict at `root_key` if provided, else the whole dict."""
    if root_key is None:
        return dict(data)
    sub = data.get(root_key, {})
    if not isinstance(sub, dict):
        raise ValueError(
            f"Root key '{root_key}' is not a mapping in the YAML.")
    return dict(sub)


def _undot(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert dotted keys into nested dicts.
    Example: {"resolution.level_mode": "fixed", "grid.tile_size": 512}
             -> {"resolution": {"level_mode": "fixed"}, "grid": {"tile_size": 512}}
    """
    root: Dict[str, Any] = {}
    for k, v in overrides.items():
        if "." not in k:
            root[k] = v
            continue
        cur = root
        *parts, last = k.split(".")
        for p in parts:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise ValueError(
                    f"Conflict while building overrides at '{k}'.")
        if isinstance(cur.get(last), dict) and isinstance(v, dict):
            cur[last] = _deep_merge(cur[last], v)  # type: ignore[arg-type]
        else:
            cur[last] = v
    return root


# -------------------------
# Public API
# -------------------------


def load_config_with_presets(
    default_path: Path | str,
    presets: Iterable[Path | str] | None = None,
    *,
    root_key: str | None = "preprocessing",
    overrides: Mapping[str, Any] | None = None,
) -> TilingConfig:
    """
    Load tiling config from YAML(s), optionally scoped under a root key
    (e.g. 'preprocessing'), then apply in-memory overrides.

    Precedence:
      default.yaml  <  preset1.yaml  <  presetN.yaml  <  overrides (dict)

    `overrides` can be nested dicts or dotted keys, e.g.:
      {"resolution.level_mode": "fixed", "grid.tile_size": 512}
    """
    # 1) base
    merged = _extract_root(_load_yaml(default_path), root_key)

    # 2) presets (later wins)
    for p in (presets or []):
        merged = _deep_merge(merged, _extract_root(_load_yaml(p), root_key))

    # 3) in-memory overrides
    if overrides:
        merged = _deep_merge(merged, _undot(overrides))

    # 4) validate
    try:
        return TilingConfig.model_validate(merged)
    except ValidationError as e:
        raise ValueError(
            f"Invalid tiling configuration after merge/override:\n{e}") from e


def apply_overrides(
    cfg: TilingConfig,
    overrides: Mapping[str, Any],
) -> TilingConfig:
    """
    Pure function: return a new TilingConfig with overrides applied.
    Accepts dotted or nested keys.
    """
    nested = _undot(overrides)
    # pydantic will validate the nested update
    return cfg.model_copy(update=nested)
