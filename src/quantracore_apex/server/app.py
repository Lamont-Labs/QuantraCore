"""
FastAPI Application for QuantraCore Apex.

Provides REST API for Apex engine functionality.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import ApexContext, OhlcvWindow
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.apexlab.windows import WindowBuilder
from src.quantracore_apex.prediction.monster_runner import MonsterRunnerEngine
from src.quantracore_apex.protocols.omega.omega import OmegaDirectives


logger = logging.getLogger(__name__)


class ScanRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    lookback_days: int = 150
    seed: Optional[int] = 42


class UniverseScanRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1d"
    lookback_days: int = 150


class ScanResult(BaseModel):
    symbol: str
    quantrascore: float
    score_bucket: str
    regime: str
    risk_tier: str
    entropy_state: str
    suppression_state: str
    drift_state: str
    verdict_action: str
    verdict_confidence: float
    omega_alerts: List[str]
    protocol_fired_count: int
    window_hash: str
    timestamp: str


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="QuantraCore Apex API",
        description="Institutional-Grade Deterministic AI Trading Intelligence Engine",
        version="8.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    engine = ApexEngine(enable_logging=True)
    data_adapter = SyntheticAdapter(seed=42)
    window_builder = WindowBuilder(window_size=100)
    monster_runner = MonsterRunnerEngine()
    omega_directives = OmegaDirectives()
    
    scan_cache: Dict[str, Any] = {}
    
    @app.get("/")
    async def root():
        return {
            "name": "QuantraCore Apex",
            "version": "8.0.0",
            "status": "operational",
            "compliance_note": "Structural analysis only - not trading advice"
        }
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "engine": "operational",
            "data_layer": "operational"
        }
    
    @app.post("/scan_symbol", response_model=ScanResult)
    async def scan_symbol(request: ScanRequest):
        """Scan a single symbol and return Apex analysis."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=request.lookback_days)
            
            bars = data_adapter.fetch_ohlcv(
                request.symbol, start_date, end_date, request.timeframe
            )
            
            if len(bars) < 100:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient data: got {len(bars)} bars, need at least 100"
                )
            
            normalized_bars, _ = normalize_ohlcv(bars)
            
            window = window_builder.build_single(
                normalized_bars, request.symbol, request.timeframe
            )
            
            if window is None:
                raise HTTPException(status_code=400, detail="Failed to build analysis window")
            
            context = ApexContext(seed=request.seed or 42, compliance_mode=True)
            result = engine.run(window, context)
            
            omega_statuses = omega_directives.apply_all(result)
            omega_alerts = [
                name for name, status in omega_statuses.items()
                if status.active
            ]
            
            protocol_fired = sum(1 for p in result.protocol_results if p.fired)
            
            scan_cache[result.window_hash] = result
            
            return ScanResult(
                symbol=result.symbol,
                quantrascore=result.quantrascore,
                score_bucket=result.score_bucket.value,
                regime=result.regime.value,
                risk_tier=result.risk_tier.value,
                entropy_state=result.entropy_state.value,
                suppression_state=result.suppression_state.value,
                drift_state=result.drift_state.value,
                verdict_action=result.verdict.action,
                verdict_confidence=result.verdict.confidence,
                omega_alerts=omega_alerts,
                protocol_fired_count=protocol_fired,
                window_hash=result.window_hash,
                timestamp=result.timestamp.isoformat()
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error scanning {request.symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/scan_universe")
    async def scan_universe(request: UniverseScanRequest):
        """Scan multiple symbols and return summarized results."""
        results = []
        errors = []
        
        for symbol in request.symbols:
            try:
                scan_request = ScanRequest(
                    symbol=symbol,
                    timeframe=request.timeframe,
                    lookback_days=request.lookback_days
                )
                result = await scan_symbol(scan_request)
                results.append(result.model_dump())
            except Exception as e:
                errors.append({"symbol": symbol, "error": str(e)})
        
        results.sort(key=lambda x: x["quantrascore"], reverse=True)
        
        return {
            "scan_count": len(request.symbols),
            "success_count": len(results),
            "error_count": len(errors),
            "results": results,
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/trace/{window_hash}")
    async def get_trace(window_hash: str):
        """Get detailed protocol trace for a scan."""
        if window_hash not in scan_cache:
            raise HTTPException(status_code=404, detail="Trace not found")
        
        result = scan_cache[window_hash]
        
        return {
            "window_hash": window_hash,
            "symbol": result.symbol,
            "microtraits": result.microtraits.model_dump(),
            "entropy_metrics": result.entropy_metrics.model_dump(),
            "suppression_metrics": result.suppression_metrics.model_dump(),
            "drift_metrics": result.drift_metrics.model_dump(),
            "continuation_metrics": result.continuation_metrics.model_dump(),
            "volume_metrics": result.volume_metrics.model_dump(),
            "protocol_results": [p.model_dump() for p in result.protocol_results],
            "verdict": result.verdict.model_dump(),
            "omega_overrides": result.omega_overrides
        }
    
    @app.post("/monster_runner/{symbol}")
    async def check_monster_runner(symbol: str, lookback_days: int = 150):
        """Check for MonsterRunner rare event precursors."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
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
            
        except Exception as e:
            logger.error(f"MonsterRunner error for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


app = create_app()
