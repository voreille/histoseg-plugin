from fastapi.testclient import TestClient

from histoseg_plugin.api.main import create_app, get_settings


def test_health(test_settings):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
