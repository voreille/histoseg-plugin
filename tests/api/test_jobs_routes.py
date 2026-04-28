def test_create_job(client, test_slide):
    payload = {
        "items": [
            {
                "slide_uri": str(test_slide),
                "model_id": "default",
                "params": {},
            }
        ]
    }

    response = client.post("/jobs", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] in {"pending", "completed"}


def test_read_job(client, test_slide):
    payload = {
        "items": [
            {
                "slide_uri": str(test_slide),
                "model_id": "default",
                "params": {},
            }
        ]
    }

    create_response = client.post("/jobs", json=payload)
    assert create_response.status_code == 200

    job_id = create_response.json()["job_id"]

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    data = response.json()

    assert data["job_id"] == job_id
    assert data["status"] in {"pending", "completed"}
    assert len(data["tasks"]) == 1

    task = data["tasks"][0]
    assert task["model_id"] == "default"
    assert task["status"] in {"pending", "cached", "completed"}


def test_read_missing_job_returns_404(client):
    response = client.get("/jobs/999999")

    assert response.status_code == 404
