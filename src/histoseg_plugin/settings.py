import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str

    allowed_roots: list[Path] = Field(default_factory=lambda: [])

    debug: bool = False

    results_root: Path = Path("./results")
    models_root: Path = Path("./models")
    logs_root: Path = Path("./logs")

    worker_poll_interval_seconds: float = 1.0
    worker_heartbeat_seconds: float = 5.0
    gpu_idle_unload_seconds: float = 300.0
    stale_task_timeout_seconds: int = 60

    default_model_id: str = "default"
    preferred_device: str = "cuda"

    model_config = SettingsConfigDict(
        env_prefix="HISTOSEG_",
        extra="ignore",
    )


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping at top level: {path}")

    return data


def build_settings(config_path: str | None = None) -> Settings:
    path_str = (
        config_path or os.environ.get("HISTOSEG_CONFIG") or "config/settings.yaml"
    )
    yaml_data = _load_yaml_config(Path(path_str).resolve())
    return Settings(**yaml_data)


def ensure_settings_dirs(settings: Settings) -> Settings:
    settings.results_root.mkdir(parents=True, exist_ok=True)
    settings.models_root.mkdir(parents=True, exist_ok=True)
    settings.logs_root.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    return ensure_settings_dirs(build_settings(config_path))
