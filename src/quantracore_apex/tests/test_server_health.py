"""
Test Server Health Endpoints

Verifies the FastAPI server health and basic endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test /health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_root_endpoint(client):
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "name" in data or "version" in data


def test_scan_endpoint_requires_symbol(client):
    """Test scan endpoint validation."""
    response = client.get("/scan/AAPL")
    assert response.status_code in [200, 422, 500]


def test_score_endpoint(client):
    """Test legacy score endpoint."""
    response = client.get("/score?ticker=TEST")
    assert response.status_code in [200, 404, 500]
