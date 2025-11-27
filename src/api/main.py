"""
QuantraCore Apex API v8.0

FastAPI endpoints for the Apex engine.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import ApexContext
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.apexlab.windows import WindowBuilder
from src.quantracore_apex.prediction.monster_runner import MonsterRunnerEngine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="QuantraCore Apex API",
    version="8.0.0",
    description="Institutional-Grade Deterministic AI Trading Intelligence Engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ApexEngine(enable_logging=False)
data_adapter = SyntheticAdapter(seed=42)
window_builder = WindowBuilder(window_size=100)
monster_runner = MonsterRunnerEngine()

scan_cache = {}


@app.get("/")
def root():
    return {
        "name": "QuantraCore Apex",
        "version": "8.0.0",
        "status": "operational",
        "compliance_note": "Structural analysis only - not trading advice"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "8.0.0",
        "time": datetime.utcnow().isoformat(),
        "engine": "operational"
    }


@app.get("/score")
def score(ticker: str = "AAPL", seed: int = 42):
    """Get QuantraScore for a symbol (legacy endpoint)."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)
        
        bars = data_adapter.fetch_ohlcv(ticker, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        window = window_builder.build_single(normalized_bars, ticker)
        
        if window is None:
            return {"ticker": ticker, "score": 50, "error": "insufficient_data"}
        
        context = ApexContext(seed=seed, compliance_mode=True)
        result = engine.run(window, context)
        
        return {
            "ticker": ticker,
            "score": result.quantrascore,
            "regime": result.regime.value,
            "risk_tier": result.risk_tier.value,
            "seed": seed,
            "window_hash": result.window_hash
        }
    except Exception as e:
        logger.error(f"Error computing score: {e}")
        return {"ticker": ticker, "score": 50, "error": str(e)}


@app.get("/scan/{symbol}")
def scan_symbol(symbol: str, seed: int = 42, lookback: int = 150):
    """Full Apex scan for a symbol."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback)
        
        bars = data_adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        
        if len(bars) < 100:
            raise HTTPException(status_code=400, detail=f"Insufficient data: {len(bars)} bars")
        
        normalized_bars, _ = normalize_ohlcv(bars)
        window = window_builder.build_single(normalized_bars, symbol)
        
        if window is None:
            raise HTTPException(status_code=400, detail="Failed to build analysis window")
        
        context = ApexContext(seed=seed, compliance_mode=True)
        result = engine.run(window, context)
        
        scan_cache[result.window_hash] = result
        
        protocol_fired = sum(1 for p in result.protocol_results if p.fired)
        
        return {
            "symbol": result.symbol,
            "quantrascore": result.quantrascore,
            "score_bucket": result.score_bucket.value,
            "regime": result.regime.value,
            "risk_tier": result.risk_tier.value,
            "entropy_state": result.entropy_state.value,
            "suppression_state": result.suppression_state.value,
            "drift_state": result.drift_state.value,
            "verdict": {
                "action": result.verdict.action,
                "confidence": result.verdict.confidence,
                "compliance_note": result.verdict.compliance_note
            },
            "protocol_fired_count": protocol_fired,
            "protocol_results": [
                {
                    "protocol_id": p.protocol_id,
                    "fired": bool(p.fired),
                    "confidence": float(p.confidence),
                    "signal_type": p.signal_type
                }
                for p in result.protocol_results if p.fired
            ],
            "window_hash": result.window_hash,
            "timestamp": result.timestamp.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def sanitize_dict(d):
    """Convert numpy types to native Python types."""
    import numpy as np
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(v) for v in d]
    elif isinstance(d, np.bool_):
        return bool(d)
    elif isinstance(d, np.integer):
        return int(d)
    elif isinstance(d, np.floating):
        return float(d)
    elif isinstance(d, np.ndarray):
        return d.tolist()
    return d

@app.get("/trace/{window_hash}")
def get_trace(window_hash: str):
    """Get detailed protocol trace for a scan."""
    if window_hash not in scan_cache:
        raise HTTPException(status_code=404, detail="Trace not found - run a scan first")
    
    result = scan_cache[window_hash]
    
    return sanitize_dict({
        "window_hash": window_hash,
        "symbol": result.symbol,
        "microtraits": result.microtraits.model_dump(),
        "entropy_metrics": result.entropy_metrics.model_dump(),
        "suppression_metrics": result.suppression_metrics.model_dump(),
        "drift_metrics": result.drift_metrics.model_dump(),
        "continuation_metrics": result.continuation_metrics.model_dump(),
        "volume_metrics": result.volume_metrics.model_dump(),
        "protocol_results": [p.model_dump() for p in result.protocol_results[:20]],
        "verdict": result.verdict.model_dump()
    })


@app.get("/monster_runner/{symbol}")
def check_monster_runner(symbol: str, lookback: int = 150):
    """Check for MonsterRunner rare event precursors."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback)
        
        bars = data_adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        window = window_builder.build_single(normalized_bars, symbol)
        
        if window is None:
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        output = monster_runner.analyze(window)
        
        return {
            "symbol": symbol,
            "runner_probability": output.runner_probability,
            "runner_state": output.runner_state.value,
            "rare_event_class": output.rare_event_class.value,
            "metrics": {
                "compression_trace": output.compression_trace,
                "entropy_floor": output.entropy_floor,
                "volume_pulse": output.volume_pulse,
                "range_contraction": output.range_contraction,
                "primed_confidence": output.primed_confidence
            },
            "compliance_note": "Structural detection only - not a trade signal"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MonsterRunner error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/hud")
def risk_hud(ticker: str = "AAPL", seed: int = 42):
    """Risk HUD (legacy endpoint)."""
    result = score(ticker, seed)
    return {
        "ticker": ticker,
        "score": result["score"],
        "risk_tier": result.get("risk_tier", "unknown"),
        "filters": {
            "extreme_risk": result.get("score", 50) < 20 or result.get("score", 50) > 80
        }
    }


@app.get("/audit/export")
def audit_export():
    """Export audit data (legacy endpoint)."""
    Path("dist/golden_demo_outputs").mkdir(parents=True, exist_ok=True)
    doc = {
        "status": "demo_export",
        "version": "8.0.0",
        "time": datetime.utcnow().isoformat(),
        "scan_cache_size": len(scan_cache)
    }
    Path("dist/golden_demo_outputs/audit_export.json").write_text(json.dumps(doc, indent=2))
    return doc
