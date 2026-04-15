from fastapi.testclient import TestClient

from app.main import app



def test_status_endpoint_returns_device_status():
    client = TestClient(app)
    response = client.get("/api/status")

    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "device" in data
    assert "database" in data
