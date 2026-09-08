import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls):
        super().__init__(settings_cls)
        self.config_env_var = "HISTOSEG_CONFIG"
        self.default_path = "./config/settings.yaml"

    def get_field_value(self, field, field_name):
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        env_var = os.environ.get(self.config_env_var)
        yaml_path = Path(env_var) if env_var else Path(self.default_path)
        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            pass

        return {}


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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Setup precedence for settings sources:
            1) settings.yaml file
            2) env variables

        Standard BaseSettings method. Arguments should not be changed"""
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
        )


def ensure_settings_dirs(settings: Settings) -> Settings:
    settings.results_root.mkdir(parents=True, exist_ok=True)
    settings.models_root.mkdir(parents=True, exist_ok=True)
    settings.logs_root.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    return ensure_settings_dirs(Settings())
