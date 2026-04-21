from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    queue_db_url: str = "sqlite:///./histoseg_queue.db"

    debug: bool = False

    results_root: Path = Field(default=Path("./results"))
    models_root: Path = Field(default=Path("./models"))
    logs_root: Path = Field(default=Path("./logs"))

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


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    path_str = config_path or os.environ.get("HISTOSEG_CONFIG") or "config/settings.yaml"
    config_file = Path(path_str).resolve()

    yaml_data = _load_yaml_config(config_file)

    settings = Settings(**yaml_data)

    settings.results_root.mkdir(parents=True, exist_ok=True)
    settings.models_root.mkdir(parents=True, exist_ok=True)
    settings.logs_root.mkdir(parents=True, exist_ok=True)

    return settings