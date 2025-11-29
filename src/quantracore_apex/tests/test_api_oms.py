"""
Tests for OMS API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.quantracore_apex.server.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestOMSEndpoints:
    """Tests for Order Management System API endpoints."""
    
    def test_place_market_order(self, client):
        response = client.post("/oms/place", json={
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "order_type": "market"
        })
        assert response.status_code == 200
        data = response.json()
        assert "order" in data
        assert data["order"]["symbol"] == "AAPL"
        assert data["order"]["status"] == "pending"
        assert data["message"] == "Order placed (simulation mode)"
    
    def test_place_limit_order(self, client):
        response = client.post("/oms/place", json={
            "symbol": "TSLA",
            "side": "sell",
            "quantity": 50,
            "order_type": "limit",
            "limit_price": 250.00
        })
        assert response.status_code == 200
        data = response.json()
        assert data["order"]["order_type"] == "limit"
        assert data["order"]["limit_price"] == 250.00
    
    def test_submit_order(self, client):
        place_res = client.post("/oms/place", json={
            "symbol": "NVDA",
            "side": "buy",
            "quantity": 25
        })
        order_id = place_res.json()["order"]["order_id"]
        
        submit_res = client.post(f"/oms/submit/{order_id}")
        assert submit_res.status_code == 200
        assert submit_res.json()["order"]["status"] == "submitted"
    
    def test_fill_order_updates_portfolio(self, client):
        client.post("/oms/reset")
        
        place_res = client.post("/oms/place", json={
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100
        })
        order_id = place_res.json()["order"]["order_id"]
        
        client.post(f"/oms/submit/{order_id}")
        
        fill_res = client.post("/oms/fill", json={
            "order_id": order_id,
            "fill_price": 150.00,
            "commission": 1.00
        })
        
        assert fill_res.status_code == 200
        data = fill_res.json()
        assert data["order_status"] == "filled"
        assert data["fill"]["quantity"] == 100
        assert data["fill"]["price"] == 150.00
        
        portfolio_res = client.get("/portfolio/status")
        portfolio_data = portfolio_res.json()
        assert portfolio_data["snapshot"]["num_positions"] == 1
    
    def test_cancel_order(self, client):
        place_res = client.post("/oms/place", json={
            "symbol": "AMD",
            "side": "buy",
            "quantity": 200
        })
        order_id = place_res.json()["order"]["order_id"]
        
        cancel_res = client.post(f"/oms/cancel/{order_id}")
        assert cancel_res.status_code == 200
        assert cancel_res.json()["order"]["status"] == "cancelled"
    
    def test_get_orders(self, client):
        client.post("/oms/reset")
        
        client.post("/oms/place", json={"symbol": "AAPL", "side": "buy", "quantity": 100})
        client.post("/oms/place", json={"symbol": "TSLA", "side": "sell", "quantity": 50})
        
        response = client.get("/oms/orders")
        assert response.status_code == 200
        assert response.json()["count"] >= 2
    
    def test_get_positions(self, client):
        response = client.get("/oms/positions")
        assert response.status_code == 200
        data = response.json()
        assert "positions" in data
        assert data["simulation_mode"]
    
    def test_reset_oms(self, client):
        client.post("/oms/place", json={"symbol": "AAPL", "side": "buy", "quantity": 100})
        
        reset_res = client.post("/oms/reset")
        assert reset_res.status_code == 200
        assert "reset" in reset_res.json()["message"].lower()
        
        orders_res = client.get("/oms/orders")
        assert orders_res.json()["count"] == 0
    
    def test_submit_nonexistent_order(self, client):
        response = client.post("/oms/submit/nonexistent-id")
        assert response.status_code == 404
    
    def test_cancel_filled_order_fails(self, client):
        client.post("/oms/reset")
        
        place_res = client.post("/oms/place", json={
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10
        })
        order_id = place_res.json()["order"]["order_id"]
        
        client.post(f"/oms/submit/{order_id}")
        client.post("/oms/fill", json={"order_id": order_id, "fill_price": 150.0})
        
        cancel_res = client.post(f"/oms/cancel/{order_id}")
        assert cancel_res.status_code == 400


class TestRiskSignalEndpoints:
    """Tests for Risk and Signal API endpoints."""
    
    def test_risk_assess_endpoint(self, client):
        response = client.post("/risk/assess/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert "risk_assessment" in data
        assert "composite_risk" in data["risk_assessment"]
        assert "permission" in data["risk_assessment"]
    
    def test_signal_generate_endpoint(self, client):
        response = client.post("/signal/generate/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert "signal" in data
        assert "direction" in data["signal"]
        assert "strength" in data["signal"]
        assert "confidence" in data["signal"]


class TestPortfolioEndpoints:
    """Tests for Portfolio API endpoints."""
    
    def test_portfolio_status(self, client):
        response = client.get("/portfolio/status")
        assert response.status_code == 200
        data = response.json()
        assert "snapshot" in data
        assert "total_equity" in data
    
    def test_portfolio_heat_map(self, client):
        response = client.get("/portfolio/heat_map")
        assert response.status_code == 200
        data = response.json()
        assert "heat_map" in data
        assert "sector_exposure" in data


class TestDashboardEndpoint:
    """Tests for ApexDesk dashboard endpoint."""
    
    def test_desk_returns_html(self, client):
        response = client.get("/desk")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "ApexDesk" in response.text
        assert "scanSymbol" in response.text
        assert "loadPortfolio" in response.text
    
    def test_api_stats(self, client):
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "9.0-A"
        assert data["protocols"]["tier"] == 80
        assert data["protocols"]["learning"] == 25
        assert data["simulation_mode"]
        assert "v9_hardening" in data
        assert data["v9_hardening"]["redundant_scoring"]
