"""
API endpoint tests for QuantraCore Apex server.
"""
from fastapi.testclient import TestClient
from src.quantracore_apex.server.app import app

client = TestClient(app)


def test_health_ok():
    """Test health endpoint returns healthy status."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"


def test_root_endpoint():
    """Test root endpoint returns system info."""
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "name" in data
    assert data["name"] == "QuantraCore Apex"
    assert data["status"] == "operational"


def test_stats_endpoint():
    """Test stats endpoint returns system statistics."""
    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert "protocols" in data
    protocols = data["protocols"]
    assert protocols["tier"] == 80
    assert protocols["learning"] == 25
    assert protocols["monster_runner"] == 5
    assert protocols["omega"] == 5


def test_scan_symbol_success():
    """Test that symbol scanning returns valid results."""
    r = client.post("/scan_symbol", json={"symbol": "AAPL", "seed": 42})
    assert r.status_code == 200
    data = r.json()
    assert "quantrascore" in data
    assert "symbol" in data
    assert "window_hash" in data
    assert data["symbol"] == "AAPL"
    assert 0 <= data["quantrascore"] <= 100
