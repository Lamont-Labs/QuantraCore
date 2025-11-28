"""
QuantraCore Apex Dashboard — Web UI Application

A professional web dashboard for viewing scan results,
running analysis, and exploring backtest data.

All outputs are structural probabilities for research only.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from polygon import RESTClient
except ImportError:
    RESTClient = None

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvBar

app = FastAPI(
    title="QuantraCore Apex Dashboard",
    description="Research dashboard for structural probability analysis",
    version="8.0.0",
)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

engine = ApexEngine(auto_load_protocols=True)

PROOF_LOG = Path("proof_logs/real_world_proof.jsonl")
SUMMARY_LOG = Path("proof_logs/backtest_500_summary.json")


def load_recent_scans(limit: int = 10) -> List[Dict[str, Any]]:
    """Load recent scans from proof log."""
    scans = []
    if PROOF_LOG.exists():
        with open(PROOF_LOG) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    scans.append({
                        "symbol": rec.get("symbol", "?"),
                        "score": rec.get("quantra_score", 0),
                        "regime": rec.get("regime", "unknown"),
                        "risk": rec.get("risk_tier", "medium"),
                        "protocol_count": rec.get("protocol_count", 0),
                    })
                except:
                    continue
    
    return sorted(scans, key=lambda x: x["score"], reverse=True)[:limit]


def load_backtest_summary() -> Optional[Dict[str, Any]]:
    """Load backtest summary if available."""
    for path in [SUMMARY_LOG, Path("proof_logs/backtest_summary.json")]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def get_dashboard_stats() -> Dict[str, Any]:
    """Calculate dashboard statistics."""
    scans = load_recent_scans(100)
    
    return {
        "symbols_scanned": len(scans),
        "avg_score": sum(s["score"] for s in scans) / len(scans) if scans else 0,
        "high_conviction": sum(1 for s in scans if s["score"] >= 60),
        "protocols_loaded": 105,
    }


def get_regime_distribution(scans: List[Dict]) -> Dict[str, int]:
    """Calculate regime distribution."""
    dist = {}
    for s in scans:
        regime = s.get("regime", "unknown")
        dist[regime] = dist.get(regime, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    scans = load_recent_scans(50)
    stats = get_dashboard_stats()
    regime_dist = get_regime_distribution(scans)
    top_signals = [s for s in scans if s["score"] >= 60][:10]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "active": "dashboard",
        "stats": stats,
        "recent_scans": scans[:10],
        "regime_dist": regime_dist,
        "top_signals": top_signals,
    })


@app.get("/scan", response_class=HTMLResponse)
async def scan_page(request: Request):
    """Scan input page."""
    return templates.TemplateResponse("scan.html", {
        "request": request,
        "active": "scan",
        "symbol": None,
        "result": None,
        "error": None,
    })


@app.post("/scan", response_class=HTMLResponse)
async def run_scan(request: Request, symbol: str = Form(...), period: str = Form("1y")):
    """Run a scan on a symbol."""
    result = None
    error = None
    
    try:
        client = RESTClient(api_key=os.environ.get("POLYGON_API_KEY", "")) if RESTClient else None
        
        if client:
            import time
            from datetime import datetime, timedelta
            
            end = datetime.now()
            start = end - timedelta(days=365 if period == "1y" else 180 if period == "6m" else 90)
            
            aggs = list(client.get_aggs(
                symbol.upper(), 1, "day",
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
                adjusted=True, limit=1000
            ))
            
            if len(aggs) < 50:
                error = f"Insufficient data for {symbol} ({len(aggs)} bars)"
            else:
                bars = []
                for a in aggs:
                    bars.append(OhlcvBar(
                        timestamp=datetime.utcfromtimestamp(a.timestamp / 1000),
                        open=float(a.open),
                        high=float(a.high),
                        low=float(a.low),
                        close=float(a.close),
                        volume=float(a.volume),
                    ))
                
                scan = engine.run_scan(bars, symbol.upper(), seed=42)
                
                fired = [p.protocol_id for p in scan.protocol_results if p.fired]
                
                result = {
                    "symbol": symbol.upper(),
                    "score": scan.quantrascore,
                    "regime": scan.regime.value if hasattr(scan.regime, 'value') else str(scan.regime),
                    "risk": scan.risk_tier.value if hasattr(scan.risk_tier, 'value') else str(scan.risk_tier),
                    "entropy": str(scan.entropy_metrics.entropy_state),
                    "protocols": fired,
                    "num_candles": len(bars),
                    "window_hash": scan.window_hash,
                }
        else:
            error = "Polygon API not configured. Set POLYGON_API_KEY environment variable."
    
    except Exception as e:
        error = str(e)
    
    return templates.TemplateResponse("scan.html", {
        "request": request,
        "active": "scan",
        "symbol": symbol,
        "result": result,
        "error": error,
    })


@app.get("/scan/{symbol}", response_class=HTMLResponse)
async def scan_symbol(request: Request, symbol: str):
    """Direct scan URL for a symbol."""
    return await run_scan(request, symbol=symbol, period="1y")


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    """Backtest results page."""
    summary = load_backtest_summary()
    
    return templates.TemplateResponse("backtest.html", {
        "request": request,
        "active": "backtest",
        "summary": summary,
    })


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "8.0.0",
        "engine": "ApexEngine",
        "protocols_loaded": 105,
        "compliance_note": "Research purposes only - not trading advice",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
