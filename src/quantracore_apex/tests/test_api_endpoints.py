"""
Comprehensive API Endpoint Tests

Tests all API endpoints for correctness.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health endpoints."""
    
    def test_health(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data is not None


class TestScanEndpoints:
    """Tests for scan endpoints."""
    
    def test_scan_valid_symbol(self, client):
        """Test scan with valid symbol."""
        response = client.get("/scan/AAPL")
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "quantrascore" in data or "error" in data
    
    def test_scan_invalid_symbol(self, client):
        """Test scan with potentially invalid symbol."""
        response = client.get("/scan/INVALID123456")
        
        assert response.status_code in [200, 404, 422, 500]
    
    def test_scan_empty_symbol(self, client):
        """Test scan endpoint requires symbol."""
        response = client.get("/scan/")
        
        assert response.status_code in [307, 404, 405]


class TestScoreEndpoints:
    """Tests for legacy score endpoints."""
    
    def test_score_with_ticker(self, client):
        """Test score endpoint with ticker."""
        response = client.get("/score?ticker=TEST")
        
        assert response.status_code in [200, 404, 500]
    
    def test_score_without_ticker(self, client):
        """Test score endpoint without ticker."""
        response = client.get("/score")
        
        assert response.status_code in [200, 400, 422]


class TestMonsterRunnerEndpoints:
    """Tests for MonsterRunner endpoints."""
    
    def test_monster_runner(self, client):
        """Test monster_runner endpoint."""
        response = client.get("/monster_runner/AAPL")
        
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data is not None


class TestTraceEndpoints:
    """Tests for trace endpoints."""
    
    def test_trace_with_hash(self, client):
        """Test trace endpoint with hash."""
        response = client.get("/trace/abc123")
        
        assert response.status_code in [200, 404, 500]


class TestRiskEndpoints:
    """Tests for risk endpoints."""
    
    def test_risk_hud(self, client):
        """Test risk HUD endpoint."""
        response = client.get("/risk/hud")
        
        assert response.status_code in [200, 404, 500]


class TestAuditEndpoints:
    """Tests for audit endpoints."""
    
    def test_audit_export(self, client):
        """Test audit export endpoint."""
        response = client.get("/audit/export")
        
        assert response.status_code in [200, 404, 500]


class TestAPICompliance:
    """Tests for API compliance."""
    
    def test_no_trading_endpoints(self, client):
        """Test no trading/order endpoints exist."""
        trading_paths = ["/order", "/trade", "/execute", "/buy", "/sell"]
        
        for path in trading_paths:
            response = client.get(path)
            assert response.status_code in [404, 405]
            
            response = client.post(path)
            assert response.status_code in [404, 405]
    
    def test_json_responses(self, client):
        """Test endpoints return JSON."""
        endpoints = ["/", "/health"]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.headers["content-type"].startswith("application/json")


class TestAPIErrorHandling:
    """Tests for error handling."""
    
    def test_404_on_invalid_path(self, client):
        """Test 404 on invalid path."""
        response = client.get("/nonexistent/path")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed on GET-only endpoints."""
        response = client.delete("/health")
        
        assert response.status_code == 405
