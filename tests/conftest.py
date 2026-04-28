# tests/conftest.py
import pytest
import yaml
from fastapi.testclient import TestClient

from histoseg_plugin.api.main import create_app
from histoseg_plugin.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings(tmp_path):
    return Settings(
        queue_db_url=f"sqlite:///{tmp_path / 'test.db'}",
        allowed_roots=[tmp_path],
        debug=False,
        results_root=tmp_path / "results",
        models_root=tmp_path / "models",
        logs_root=tmp_path / "logs",
        preferred_device="cpu",
    )


@pytest.fixture
def test_config_file(tmp_path, test_settings):
    config_path = tmp_path / "settings-test.yaml"

    config = {
        "queue_db_url": test_settings.queue_db_url,
        "allowed_roots": [str(p) for p in test_settings.allowed_roots],
        "debug": test_settings.debug,
        "results_root": str(test_settings.results_root),
        "models_root": str(test_settings.models_root),
        "logs_root": str(test_settings.logs_root),
        "preferred_device": test_settings.preferred_device,
        "default_model_id": test_settings.default_model_id,
        "worker_poll_interval_seconds": test_settings.worker_poll_interval_seconds,
        "worker_heartbeat_seconds": test_settings.worker_heartbeat_seconds,
        "gpu_idle_unload_seconds": test_settings.gpu_idle_unload_seconds,
        "stale_task_timeout_seconds": test_settings.stale_task_timeout_seconds,
    }

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.fixture
def app(monkeypatch, test_config_file, test_settings):
    monkeypatch.setenv("HISTOSEG_CONFIG", str(test_config_file))

    app = create_app()

    # Still useful for routes/dependencies using Depends(get_settings)
    app.dependency_overrides[get_settings] = lambda: test_settings

    return app


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_slide(tmp_path):
    slide_path = tmp_path / "slide.svs"
    slide_path.write_bytes(b"fake slide")
    return slide_path