def test_get_queue_state(client):
    response = client.get("/queue")

    assert response.status_code == 200
    assert response.json() == {"paused": False}


def test_pause_queue(client):
    response = client.post("/queue/pause")

    assert response.status_code == 200
    assert response.json() == {"paused": True}

    response = client.get("/queue")
    assert response.status_code == 200
    assert response.json() == {"paused": True}


def test_resume_queue(client):
    client.post("/queue/pause")

    response = client.post("/queue/resume")

    assert response.status_code == 200
    assert response.json() == {"paused": False}

    response = client.get("/queue")
    assert response.status_code == 200
    assert response.json() == {"paused": False}
