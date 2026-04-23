import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from histoseg_plugin.api.main import create_app, get_settings
from histoseg_plugin.settings import Settings
from histoseg_plugin.jobs.queue_models import Base


@pytest.fixture
def test_settings(tmp_path):
    return Settings(
        queue_db_url=f"sqlite:///{tmp_path / 'test.db'}",
        allowed_roots=[tmp_path],
        debug=True,
        results_root=tmp_path / "results",
        models_root=tmp_path / "models",
        logs_root=tmp_path / "logs",
        preferred_device="cpu",
    )


@pytest.fixture
def engine(test_settings):
    engine = create_engine(test_settings.queue_db_url)
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def app(test_settings):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings
    return app


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client
