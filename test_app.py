from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_prediction():
    r = client.post(
        "/predict",
        json={"text": "This is a wonderful product"}
    )
    assert r.status_code == 200
    assert "sentiment" in r.json()

def test_empty_text():
    r = client.post(
        "/predict",
        json={"text": ""}
    )
    assert r.status_code == 200
    assert "error" in r.json()
