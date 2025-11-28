#!/usr/bin/env python3
"""
QuantraCore Apex v8.2 - Live API End-to-End Test

Tests all API endpoints against the running server.
"""

import httpx
import time
import json
from datetime import datetime

print("=" * 80)
print("QUANTRACORE APEX v8.2 - LIVE API END-TO-END TEST")
print("=" * 80)
print(f"Started: {datetime.now().isoformat()}")
print()

base_url = "http://127.0.0.1:5000"
client = httpx.Client(timeout=60.0)

results = {"passed": 0, "failed": 0, "tests": []}

def test(name: str, passed: bool, details: str = "", duration_ms: float = 0):
    icon = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    print(f"  {icon} {name}: {status} ({duration_ms:.1f}ms)")
    if details and not passed:
        print(f"      → {details[:100]}")
    results["tests"].append({"name": name, "passed": passed, "details": details})
    if passed:
        results["passed"] += 1
    else:
        results["failed"] += 1

def section(name: str):
    print()
    print(f"{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")


section("1. CORE ENDPOINTS")

t0 = time.time()
try:
    r = client.get(f"{base_url}/")
    data = r.json()
    test("GET / (root)", r.status_code == 200 and "QuantraCore" in data.get("name", ""), 
         json.dumps(data)[:100], (time.time() - t0) * 1000)
except Exception as e:
    test("GET / (root)", False, str(e), 0)

t0 = time.time()
try:
    r = client.get(f"{base_url}/health")
    data = r.json()
    test("GET /health", r.status_code == 200 and data.get("status") == "healthy",
         json.dumps(data)[:100], (time.time() - t0) * 1000)
except Exception as e:
    test("GET /health", False, str(e), 0)

t0 = time.time()
try:
    r = client.get(f"{base_url}/desk")
    test("GET /desk (dashboard)", r.status_code == 200 and "ApexDesk" in r.text,
         f"content_length={len(r.text)}", (time.time() - t0) * 1000)
except Exception as e:
    test("GET /desk (dashboard)", False, str(e), 0)


section("2. SCAN ENDPOINTS")

t0 = time.time()
try:
    r = client.post(f"{base_url}/scan_symbol", json={
        "symbol": "AAPL",
        "timeframe": "1d",
        "lookback_days": 150
    })
    data = r.json()
    test("POST /scan_symbol", r.status_code == 200 and "quantrascore" in data,
         f"score={data.get('quantrascore', 'N/A')}", (time.time() - t0) * 1000)
except Exception as e:
    test("POST /scan_symbol", False, str(e), 0)

t0 = time.time()
try:
    r = client.post(f"{base_url}/scan_universe", json={
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "timeframe": "1d",
        "lookback_days": 150
    })
    data = r.json()
    test("POST /scan_universe (3 symbols)", r.status_code == 200 and "results" in data,
         f"results={len(data.get('results', []))}", (time.time() - t0) * 1000)
except Exception as e:
    test("POST /scan_universe (3 symbols)", False, str(e), 0)


section("3. RISK ASSESSMENT")

t0 = time.time()
try:
    r = client.post(f"{base_url}/risk/assess/AAPL", params={"lookback_days": 150})
    data = r.json()
    test("POST /risk/assess/{symbol}", r.status_code == 200 and "risk_assessment" in data,
         f"tier={data.get('risk_assessment', {}).get('risk_tier', 'N/A')}", (time.time() - t0) * 1000)
except Exception as e:
    test("POST /risk/assess/{symbol}", False, str(e), 0)


section("4. SIGNAL GENERATION")

t0 = time.time()
try:
    r = client.post(f"{base_url}/signal/generate/AAPL", params={"lookback_days": 150})
    data = r.json()
    test("POST /signal/generate/{symbol}", r.status_code == 200 and "signal" in data,
         f"direction={data.get('signal', {}).get('direction', 'N/A')}", (time.time() - t0) * 1000)
except Exception as e:
    test("POST /signal/generate/{symbol}", False, str(e), 0)


section("5. MONSTER RUNNER")

t0 = time.time()
try:
    r = client.post(f"{base_url}/monster_runner/AAPL", params={"lookback_days": 150})
    data = r.json()
    test("POST /monster_runner/{symbol}", r.status_code == 200 and "runner_probability" in data,
         f"prob={data.get('runner_probability', 'N/A')}", (time.time() - t0) * 1000)
except Exception as e:
    test("POST /monster_runner/{symbol}", False, str(e), 0)


section("6. PORTFOLIO MANAGEMENT")

t0 = time.time()
try:
    r = client.get(f"{base_url}/portfolio/status")
    data = r.json()
    test("GET /portfolio/status", r.status_code == 200 and "total_equity" in data,
         f"equity={data.get('total_equity', 'N/A')}", (time.time() - t0) * 1000)
except Exception as e:
    test("GET /portfolio/status", False, str(e), 0)

t0 = time.time()
try:
    r = client.get(f"{base_url}/portfolio/heat_map")
    data = r.json()
    test("GET /portfolio/heat_map", r.status_code == 200 and "heat_map" in data,
         f"sectors={len(data.get('heat_map', {}))}", (time.time() - t0) * 1000)
except Exception as e:
    test("GET /portfolio/heat_map", False, str(e), 0)


section("7. ORDER MANAGEMENT SYSTEM")

t0 = time.time()
try:
    r = client.post(f"{base_url}/oms/reset")
    test("POST /oms/reset", r.status_code == 200,
         "", (time.time() - t0) * 1000)
except Exception as e:
    test("POST /oms/reset", False, str(e), 0)

t0 = time.time()
try:
    r = client.post(f"{base_url}/oms/place", json={
        "symbol": "TEST",
        "side": "buy",
        "quantity": 100,
        "order_type": "market"
    })
    data = r.json()
    order_id = data.get("order", {}).get("order_id", "")
    test("POST /oms/place (market order)", r.status_code == 200 and order_id,
         f"order_id={order_id[:8]}..." if order_id else "no order_id", (time.time() - t0) * 1000)
except Exception as e:
    test("POST /oms/place (market order)", False, str(e), 0)
    order_id = ""

if order_id:
    t0 = time.time()
    try:
        r = client.post(f"{base_url}/oms/submit/{order_id}")
        test("POST /oms/submit/{order_id}", r.status_code == 200,
             "", (time.time() - t0) * 1000)
    except Exception as e:
        test("POST /oms/submit/{order_id}", False, str(e), 0)

    t0 = time.time()
    try:
        r = client.post(f"{base_url}/oms/fill", json={
            "order_id": order_id,
            "fill_price": 150.0,
            "fill_quantity": 100,
            "commission": 1.0
        })
        test("POST /oms/fill", r.status_code == 200,
             "", (time.time() - t0) * 1000)
    except Exception as e:
        test("POST /oms/fill", False, str(e), 0)

t0 = time.time()
try:
    r = client.post(f"{base_url}/oms/place", json={
        "symbol": "TEST2",
        "side": "sell",
        "quantity": 50,
        "order_type": "limit",
        "limit_price": 155.0
    })
    data = r.json()
    order_id2 = data.get("order", {}).get("order_id", "")
    test("POST /oms/place (limit order)", r.status_code == 200 and order_id2,
         f"order_id={order_id2[:8]}..." if order_id2 else "no order_id", (time.time() - t0) * 1000)
except Exception as e:
    test("POST /oms/place (limit order)", False, str(e), 0)
    order_id2 = ""

if order_id2:
    t0 = time.time()
    try:
        r = client.post(f"{base_url}/oms/cancel/{order_id2}")
        test("POST /oms/cancel/{order_id}", r.status_code == 200,
             "", (time.time() - t0) * 1000)
    except Exception as e:
        test("POST /oms/cancel/{order_id}", False, str(e), 0)

t0 = time.time()
try:
    r = client.get(f"{base_url}/oms/orders")
    data = r.json()
    test("GET /oms/orders", r.status_code == 200 and "orders" in data,
         f"count={len(data.get('orders', []))}", (time.time() - t0) * 1000)
except Exception as e:
    test("GET /oms/orders", False, str(e), 0)

t0 = time.time()
try:
    r = client.get(f"{base_url}/oms/positions")
    data = r.json()
    test("GET /oms/positions", r.status_code == 200 and "positions" in data,
         f"count={len(data.get('positions', []))}", (time.time() - t0) * 1000)
except Exception as e:
    test("GET /oms/positions", False, str(e), 0)


section("8. API STATISTICS")

t0 = time.time()
try:
    r = client.get(f"{base_url}/api/stats")
    data = r.json()
    test("GET /api/stats", r.status_code == 200 and "version" in data,
         f"version={data.get('version', 'N/A')}", (time.time() - t0) * 1000)
except Exception as e:
    test("GET /api/stats", False, str(e), 0)


section("9. END-TO-END FLOW: Scan → Risk → Signal → Order")

t0 = time.time()
flow_passed = True
flow_details = []

try:
    r = client.post(f"{base_url}/scan_symbol", json={"symbol": "FLOW_TEST", "lookback_days": 150})
    scan_data = r.json()
    score = scan_data.get("quantrascore", 0)
    flow_details.append(f"scan:score={score:.1f}")
    
    r = client.post(f"{base_url}/risk/assess/FLOW_TEST", params={"lookback_days": 150})
    risk_data = r.json()
    risk_tier = risk_data.get("risk_assessment", {}).get("risk_tier", "unknown")
    flow_details.append(f"risk:{risk_tier}")
    
    r = client.post(f"{base_url}/signal/generate/FLOW_TEST", params={"lookback_days": 150})
    signal_data = r.json()
    direction = signal_data.get("signal", {}).get("direction", "none")
    flow_details.append(f"signal:{direction}")
    
    r = client.post(f"{base_url}/oms/place", json={
        "symbol": "FLOW_TEST",
        "side": "buy" if direction in ["long", "bullish"] else "sell",
        "quantity": 10,
        "order_type": "market"
    })
    order_data = r.json()
    order_id = order_data.get("order", {}).get("order_id", "")
    flow_details.append(f"order:{order_id[:8]}..." if order_id else "order:failed")
    
    flow_passed = all([score > 0, risk_tier != "unknown", order_id])
except Exception as e:
    flow_passed = False
    flow_details.append(f"error:{str(e)[:30]}")

test("Full E2E Flow", flow_passed, " → ".join(flow_details), (time.time() - t0) * 1000)


section("10. STRESS TEST: Rapid Sequential Requests")

t0 = time.time()
stress_results = []
for i in range(10):
    try:
        r = client.get(f"{base_url}/health")
        stress_results.append(r.status_code == 200)
    except:
        stress_results.append(False)

test(f"Rapid health checks (10x)", all(stress_results),
     f"success={sum(stress_results)}/10", (time.time() - t0) * 1000)

t0 = time.time()
scan_results = []
for i in range(5):
    try:
        r = client.post(f"{base_url}/scan_symbol", json={
            "symbol": f"STRESS{i}",
            "lookback_days": 150
        })
        scan_results.append(r.status_code == 200)
    except:
        scan_results.append(False)

test(f"Rapid scans (5x)", all(scan_results),
     f"success={sum(scan_results)}/5", (time.time() - t0) * 1000)


section("11. MULTI-SYMBOL UNIVERSE SCAN")

t0 = time.time()
try:
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "WMT"]
    r = client.post(f"{base_url}/scan_universe", json={
        "symbols": symbols,
        "lookback_days": 150
    })
    data = r.json()
    results_count = len(data.get("results", []))
    test(f"Universe scan ({len(symbols)} symbols)", 
         r.status_code == 200 and results_count == len(symbols),
         f"results={results_count}/{len(symbols)}", (time.time() - t0) * 1000)
except Exception as e:
    test(f"Universe scan ({len(symbols)} symbols)", False, str(e), 0)


client.close()

print()
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"  Total:   {results['passed'] + results['failed']}")
print(f"  Passed:  {results['passed']}")
print(f"  Failed:  {results['failed']}")
print(f"  Success: {results['passed'] / max(1, results['passed'] + results['failed']) * 100:.1f}%")
print()
print(f"Completed: {datetime.now().isoformat()}")
print()

if results['failed'] > 0:
    print("FAILED TESTS:")
    for t in results['tests']:
        if not t['passed']:
            print(f"  ✗ {t['name']}: {t['details']}")
    print()

import sys
sys.exit(0 if results['failed'] == 0 else 1)
