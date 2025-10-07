from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_score_repeatable():
    r1 = client.get("/score", params={"ticker": "AAPL", "seed": 42}).json()
    r2 = client.get("/score", params={"ticker": "AAPL", "seed": 42}).json()
    assert r1["hash"] == r2["hash"]
