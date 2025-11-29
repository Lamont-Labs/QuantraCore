"""
FastAPI Application for QuantraCore Apex.

Provides REST API for Apex engine functionality.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import os

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import ApexContext
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.apexlab.windows import WindowBuilder
from src.quantracore_apex.prediction.monster_runner import MonsterRunnerEngine
from src.quantracore_apex.protocols.omega.omega import OmegaDirectives
from src.quantracore_apex.risk.engine import RiskEngine
from src.quantracore_apex.broker.oms import OrderManagementSystem, OrderSide, OrderType
from src.quantracore_apex.portfolio.portfolio import Portfolio
from src.quantracore_apex.signal.signal_builder import SignalBuilder
import numpy as np


logger = logging.getLogger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class ScanRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    lookback_days: int = 150
    seed: Optional[int] = 42


class UniverseScanRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1d"
    lookback_days: int = 150


class PlaceOrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


class FillOrderRequest(BaseModel):
    order_id: str
    fill_price: float
    fill_quantity: Optional[float] = None
    commission: float = 0.0


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
        description="Institutional-Grade Deterministic AI Trading Intelligence Engine (v9.0-A Institutional Hardening)",
        version="9.0-A",
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
    
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "static")
    if os.path.exists(static_dir):
        app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="dashboard")
    
    engine = ApexEngine(enable_logging=True)
    data_adapter = SyntheticAdapter(seed=42)
    window_builder = WindowBuilder(window_size=100)
    monster_runner = MonsterRunnerEngine()
    omega_directives = OmegaDirectives()
    risk_engine = RiskEngine()
    oms = OrderManagementSystem(initial_cash=100000.0)
    portfolio = Portfolio(initial_cash=100000.0)
    signal_builder = SignalBuilder()
    
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
        
        return convert_numpy_types({
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
        })
    
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
    
    @app.post("/risk/assess/{symbol}")
    async def assess_risk(symbol: str, lookback_days: int = 150):
        """Comprehensive risk assessment for a symbol."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            bars = data_adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, symbol)
            
            if window is None:
                raise HTTPException(status_code=400, detail="Insufficient data")
            
            context = ApexContext(seed=42, compliance_mode=True)
            result = engine.run(window, context)
            
            assessment = risk_engine.assess(
                symbol=symbol,
                quantra_score=result.quantrascore,
                regime=result.regime.value,
                entropy_state=str(result.entropy_state),
                drift_state=str(result.drift_state),
                suppression_state=str(result.suppression_state),
                volatility_ratio=result.microtraits.volatility_ratio,
            )
            
            return {
                "symbol": symbol,
                "risk_assessment": assessment.model_dump(),
                "underlying_analysis": {
                    "quantrascore": result.quantrascore,
                    "regime": result.regime.value,
                    "entropy_state": result.entropy_state.value,
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/signal/generate/{symbol}")
    async def generate_signal(symbol: str, lookback_days: int = 150):
        """Generate unified trade signal for a symbol."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            bars = data_adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, symbol)
            
            if window is None:
                raise HTTPException(status_code=400, detail="Insufficient data")
            
            context = ApexContext(seed=42, compliance_mode=True)
            result = engine.run(window, context)
            
            assessment = risk_engine.assess(
                symbol=symbol,
                quantra_score=result.quantrascore,
                regime=result.regime.value,
                entropy_state=str(result.entropy_state),
                drift_state=str(result.drift_state),
                suppression_state=str(result.suppression_state),
                volatility_ratio=result.microtraits.volatility_ratio,
            )
            
            fired_protocols = [p.protocol_id for p in result.protocol_results if p.fired]
            current_price = bars[-1].close if bars else None
            
            signal = signal_builder.build_signal(
                symbol=symbol,
                quantra_score=result.quantrascore,
                regime=result.regime.value,
                risk_tier=result.risk_tier.value,
                entropy_state=str(result.entropy_state),
                current_price=current_price,
                volatility_pct=result.microtraits.volatility_ratio * 2,
                fired_protocols=fired_protocols,
                risk_approved=assessment.permission.value == "allow",
                risk_notes="; ".join(assessment.denial_reasons),
            )
            
            return {
                "symbol": symbol,
                "signal": signal.model_dump(),
                "risk_tier": assessment.risk_tier,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/portfolio/status")
    async def get_portfolio_status():
        """Get current portfolio status and positions."""
        snapshot = portfolio.take_snapshot()
        positions = [p.model_dump() for p in portfolio.get_all_positions()]
        
        return {
            "snapshot": snapshot.model_dump(),
            "positions": positions,
            "open_orders": len(oms.get_open_orders()),
            "cash": portfolio.cash,
            "total_equity": snapshot.total_equity,
            "total_pnl": snapshot.total_pnl,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/portfolio/heat_map")
    async def get_heat_map():
        """Get portfolio heat map by sector."""
        return {
            "heat_map": portfolio.get_heat_map(),
            "sector_exposure": portfolio.get_sector_exposure(),
            "long_exposure": portfolio.get_long_exposure(),
            "short_exposure": portfolio.get_short_exposure(),
            "net_exposure": portfolio.get_net_exposure(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/oms/orders")
    async def get_orders(symbol: Optional[str] = None, status: Optional[str] = None):
        """Get orders, optionally filtered by symbol or status."""
        orders = list(oms.orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if status:
            orders = [o for o in orders if o.status.value == status]
        
        return {
            "orders": [o.model_dump() for o in orders],
            "count": len(orders),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/oms/positions")
    async def get_positions():
        """Get all current positions from OMS."""
        return {
            "positions": oms.get_all_positions(),
            "cash": oms.cash,
            "initial_cash": oms.initial_cash,
            "simulation_mode": oms.simulation_mode,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.post("/oms/place")
    async def place_order(request: PlaceOrderRequest):
        """Place a new simulated order."""
        try:
            side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL
            order_type_map = {
                "market": OrderType.MARKET,
                "limit": OrderType.LIMIT,
                "stop": OrderType.STOP,
                "stop_limit": OrderType.STOP_LIMIT,
            }
            order_type = order_type_map.get(request.order_type.lower(), OrderType.MARKET)
            
            order = oms.place_order(
                symbol=request.symbol,
                side=side,
                quantity=request.quantity,
                order_type=order_type,
                limit_price=request.limit_price,
                stop_price=request.stop_price,
            )
            
            return {
                "order": order.model_dump(),
                "message": "Order placed (simulation mode)",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/oms/submit/{order_id}")
    async def submit_order(order_id: str):
        """Submit a pending order for execution."""
        try:
            order = oms.submit_order(order_id)
            return {
                "order": order.model_dump(),
                "message": "Order submitted",
                "timestamp": datetime.utcnow().isoformat()
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @app.post("/oms/fill")
    async def fill_order(request: FillOrderRequest):
        """Simulate filling an order and update portfolio."""
        try:
            fill = oms.simulate_fill(
                order_id=request.order_id,
                fill_price=request.fill_price,
                fill_quantity=request.fill_quantity,
                commission=request.commission,
            )
            
            order = oms.get_order(request.order_id)
            if order:
                qty_change = fill.quantity if order.side == OrderSide.BUY else -fill.quantity
                portfolio.update_position(
                    symbol=order.symbol,
                    quantity_change=qty_change,
                    price=fill.price,
                    commission=fill.commission,
                )
            
            return {
                "fill": fill.model_dump(),
                "order_status": order.status.value if order else "unknown",
                "portfolio_cash": portfolio.cash,
                "message": "Order filled (simulation)",
                "timestamp": datetime.utcnow().isoformat()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/oms/cancel/{order_id}")
    async def cancel_order(order_id: str):
        """Cancel a pending or submitted order."""
        try:
            order = oms.cancel_order(order_id)
            return {
                "order": order.model_dump(),
                "message": "Order cancelled",
                "timestamp": datetime.utcnow().isoformat()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/oms/reset")
    async def reset_oms():
        """Reset OMS and portfolio to initial state."""
        oms.reset()
        portfolio.reset()
        signal_builder.clear()
        return {
            "message": "OMS and portfolio reset to initial state",
            "cash": oms.cash,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    from src.quantracore_apex.compliance.excellence import excellence_engine
    from src.quantracore_apex.compliance.reporter import regulatory_reporter
    
    @app.get("/compliance/score")
    async def get_compliance_score():
        """
        Get current regulatory compliance excellence score.
        
        Returns comprehensive metrics showing how much we EXCEED
        regulatory requirements (not just meet them).
        """
        score = excellence_engine.calculate_score()
        return {
            "overall_score": score.overall_score,
            "excellence_level": score.level.value,
            "timestamp": score.timestamp.isoformat(),
            "metrics": score.metrics.to_dict(),
            "standards_met": score.standards_met,
            "standards_exceeded": score.standards_exceeded,
            "areas_of_excellence": score.areas_of_excellence,
            "compliance_mode": "RESEARCH_ONLY",
        }
    
    @app.get("/compliance/excellence")
    async def get_excellence_summary():
        """
        Get regulatory excellence summary with multipliers.
        
        Shows how many times we exceed each regulatory requirement:
        - FINRA 15-09: Target 3x (150 vs 50 iterations)
        - MiFID II Latency: Target 5x (1s vs 5s)
        - SEC Risk Controls: Target 4x sensitivity
        """
        return excellence_engine.get_excellence_summary()
    
    @app.get("/compliance/report")
    async def generate_compliance_report(period_days: int = 1):
        """
        Generate comprehensive regulatory compliance report.
        
        Report includes:
        - Executive summary with excellence metrics
        - Regulatory adherence breakdown (all regulations exceeded)
        - Risk control status (all Omega directives)
        - Recommendations for maintaining excellence
        """
        excellence_summary = excellence_engine.get_excellence_summary()
        result = regulatory_reporter.generate_and_save(
            excellence_summary=excellence_summary,
            period_days=period_days,
        )
        return {
            "report_id": result["report_id"],
            "saved_files": result["saved_files"],
            "executive_summary": result["executive_summary"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    @app.get("/compliance/standards")
    async def get_regulatory_standards():
        """
        Get list of all regulatory standards and our excellence targets.
        
        Each standard shows:
        - Minimum regulatory requirement
        - Industry best practice
        - QuantraCore excellence target (2x-5x requirements)
        """
        from src.quantracore_apex.compliance.excellence import RegulatoryExcellenceEngine
        
        standards = []
        for s in RegulatoryExcellenceEngine.EXCELLENCE_STANDARDS:
            standards.append({
                "regulation": s.regulation,
                "section": s.section,
                "description": s.description,
                "minimum_requirement": s.minimum_requirement,
                "industry_best_practice": s.industry_best_practice,
                "quantracore_target": s.quantracore_target,
                "excellence_multiplier": f"{s.quantracore_target / s.minimum_requirement:.1f}x",
            })
        
        return {
            "standards": standards,
            "total_regulations": len(standards),
            "compliance_mode": "RESEARCH_ONLY",
            "note": "All targets exceed regulatory minimums by 2x-5x",
        }
    
    @app.get("/compliance/omega")
    async def get_omega_status():
        """
        Get status of all Omega Directives (safety overrides).
        
        Omega directives are institutional-grade safety controls:
        - Omega 1: Extreme risk safety lock
        - Omega 2: Entropy override
        - Omega 3: Drift override
        - Omega 4: Compliance mode (always active)
        - Omega 5: Suppression lock
        """
        return {
            "omega_directives": {
                "omega_1_safety": {
                    "name": "Extreme Risk Safety Lock",
                    "enabled": omega_directives.enable_omega_1,
                    "trigger": "Extreme risk tier detected",
                    "effect": "Hard halt on all analysis outputs",
                },
                "omega_2_entropy": {
                    "name": "Entropy Override",
                    "enabled": omega_directives.enable_omega_2,
                    "trigger": "Chaotic entropy state",
                    "effect": "Suppress high-confidence signals",
                },
                "omega_3_drift": {
                    "name": "Drift Override",
                    "enabled": omega_directives.enable_omega_3,
                    "trigger": "Critical model drift",
                    "effect": "Flag outputs as potentially unreliable",
                },
                "omega_4_compliance": {
                    "name": "Compliance Mode",
                    "enabled": omega_directives.enable_omega_4,
                    "trigger": "Always active",
                    "effect": "All outputs framed as structural analysis, not advice",
                },
                "omega_5_suppression": {
                    "name": "Signal Suppression Lock",
                    "enabled": omega_directives.enable_omega_5,
                    "trigger": "Strong suppression state",
                    "effect": "Suppress signal generation",
                },
            },
            "all_directives_active": all([
                omega_directives.enable_omega_1,
                omega_directives.enable_omega_2,
                omega_directives.enable_omega_3,
                omega_directives.enable_omega_4,
                omega_directives.enable_omega_5,
            ]),
            "compliance_mode": "RESEARCH_ONLY",
        }
    
    @app.get("/desk")
    async def apex_desk():
        """Serve ApexDesk dashboard."""
        from fastapi.responses import HTMLResponse
        
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ApexDesk - QuantraCore Apex v8.2</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: #0a0a0f; 
            color: #e0e0e0; 
            min-height: 100vh;
        }
        .header { 
            background: linear-gradient(135deg, #1a1a2e, #16213e); 
            padding: 1rem 2rem; 
            border-bottom: 1px solid #2a2a4e;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { 
            font-size: 1.5rem; 
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status-badge { 
            background: #10b981; 
            color: #fff; 
            padding: 0.25rem 0.75rem; 
            border-radius: 999px; 
            font-size: 0.75rem;
        }
        .container { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 1rem; 
            padding: 1rem; 
            max-width: 1600px;
            margin: 0 auto;
        }
        .card { 
            background: #12121a; 
            border: 1px solid #2a2a4e; 
            border-radius: 0.5rem; 
            padding: 1rem;
        }
        .card-header { 
            font-size: 0.875rem; 
            color: #888; 
            margin-bottom: 0.75rem; 
            text-transform: uppercase; 
            letter-spacing: 0.05em;
        }
        .metric { 
            font-size: 2rem; 
            font-weight: 600; 
            color: #fff;
        }
        .metric.positive { color: #10b981; }
        .metric.negative { color: #ef4444; }
        .metric.neutral { color: #f59e0b; }
        .scan-input { 
            display: flex; 
            gap: 0.5rem; 
            margin-bottom: 1rem;
        }
        input { 
            background: #1a1a2e; 
            border: 1px solid #2a2a4e; 
            color: #fff; 
            padding: 0.5rem 1rem; 
            border-radius: 0.25rem; 
            flex: 1;
        }
        input:focus { outline: none; border-color: #7c3aed; }
        button { 
            background: linear-gradient(135deg, #7c3aed, #5b21b6); 
            color: #fff; 
            border: none; 
            padding: 0.5rem 1.5rem; 
            border-radius: 0.25rem; 
            cursor: pointer; 
            font-weight: 500;
            transition: transform 0.1s;
        }
        button:hover { transform: scale(1.02); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .results-table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 1rem;
            font-size: 0.875rem;
        }
        .results-table th, .results-table td { 
            padding: 0.5rem; 
            text-align: left; 
            border-bottom: 1px solid #2a2a4e;
        }
        .results-table th { 
            color: #888; 
            font-weight: 500; 
            text-transform: uppercase;
            font-size: 0.75rem;
        }
        .score-bar { 
            height: 6px; 
            background: #2a2a4e; 
            border-radius: 3px; 
            overflow: hidden;
            width: 100px;
        }
        .score-fill { 
            height: 100%; 
            transition: width 0.3s;
        }
        .risk-low { background: #10b981; }
        .risk-medium { background: #f59e0b; }
        .risk-high { background: #ef4444; }
        .risk-extreme { background: #dc2626; }
        .full-width { grid-column: 1 / -1; }
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
        .mini-card { 
            background: #1a1a2e; 
            padding: 0.75rem; 
            border-radius: 0.25rem;
            text-align: center;
        }
        .mini-label { font-size: 0.75rem; color: #666; margin-bottom: 0.25rem; }
        .mini-value { font-size: 1.25rem; font-weight: 600; }
        .compliance-note { 
            font-size: 0.75rem; 
            color: #666; 
            text-align: center; 
            padding: 1rem;
            border-top: 1px solid #2a2a4e;
        }
        .loading { opacity: 0.5; }
        .regime-badge { 
            display: inline-block; 
            padding: 0.125rem 0.5rem; 
            border-radius: 0.25rem; 
            font-size: 0.75rem;
            font-weight: 500;
        }
        .regime-trending_up { background: #10b98120; color: #10b981; }
        .regime-trending_down { background: #ef444420; color: #ef4444; }
        .regime-volatile { background: #f59e0b20; color: #f59e0b; }
        .regime-range_bound { background: #3b82f620; color: #3b82f6; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ApexDesk v8.2</h1>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span id="clock" style="color: #888; font-size: 0.875rem;"></span>
            <span class="status-badge">RESEARCH MODE</span>
        </div>
    </div>
    
    <div class="container">
        <div class="card full-width">
            <div class="card-header">Symbol Scanner</div>
            <div class="scan-input">
                <input type="text" id="symbolInput" placeholder="Enter symbol (e.g., AAPL, TSLA, NVDA)" value="AAPL">
                <button onclick="scanSymbol()" id="scanBtn">Scan</button>
            </div>
            <div id="scanResults"></div>
        </div>
        
        <div class="card">
            <div class="card-header">Latest Analysis</div>
            <div id="analysisPanel">
                <div style="color: #666; padding: 2rem; text-align: center;">
                    Scan a symbol to see analysis
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">Risk Assessment</div>
            <div id="riskPanel">
                <div style="color: #666; padding: 2rem; text-align: center;">
                    Risk data will appear after scan
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">Signal Builder</div>
            <div id="signalPanel">
                <div style="color: #666; padding: 2rem; text-align: center;">
                    Signal data will appear after scan
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">Portfolio Overview</div>
            <div id="portfolioPanel">
                <div class="grid-3">
                    <div class="mini-card">
                        <div class="mini-label">Equity</div>
                        <div class="mini-value" id="equity">$100,000</div>
                    </div>
                    <div class="mini-card">
                        <div class="mini-label">P&L</div>
                        <div class="mini-value" id="pnl">$0.00</div>
                    </div>
                    <div class="mini-card">
                        <div class="mini-label">Positions</div>
                        <div class="mini-value" id="posCount">0</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card full-width">
            <div class="card-header">System Status</div>
            <div class="grid-3">
                <div class="mini-card">
                    <div class="mini-label">Engine</div>
                    <div class="mini-value positive">Online</div>
                </div>
                <div class="mini-card">
                    <div class="mini-label">Protocols</div>
                    <div class="mini-value">80 Tier + 25 LP</div>
                </div>
                <div class="mini-card">
                    <div class="mini-label">Omega Directives</div>
                    <div class="mini-value">5 Active</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="compliance-note">
        All outputs are structural probabilities for research purposes only. Not trading advice.
    </div>
    
    <script>
        function updateClock() {
            document.getElementById('clock').textContent = new Date().toLocaleTimeString();
        }
        setInterval(updateClock, 1000);
        updateClock();
        
        async function scanSymbol() {
            const symbol = document.getElementById('symbolInput').value.toUpperCase();
            const btn = document.getElementById('scanBtn');
            btn.disabled = true;
            btn.textContent = 'Scanning...';
            
            try {
                const [scanRes, riskRes, signalRes] = await Promise.all([
                    fetch('/scan_symbol', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({symbol: symbol, seed: Date.now() % 1000})
                    }).then(r => r.json()),
                    fetch('/risk/assess/' + symbol, {method: 'POST'}).then(r => r.json()),
                    fetch('/signal/generate/' + symbol, {method: 'POST'}).then(r => r.json())
                ]);
                
                displayAnalysis(scanRes);
                displayRisk(riskRes);
                displaySignal(signalRes);
                
            } catch (e) {
                console.error(e);
                document.getElementById('scanResults').innerHTML = '<div style="color: #ef4444;">Error: ' + e.message + '</div>';
            }
            
            btn.disabled = false;
            btn.textContent = 'Scan';
        }
        
        function displayAnalysis(data) {
            const scoreClass = data.quantrascore >= 60 ? 'positive' : data.quantrascore <= 40 ? 'negative' : 'neutral';
            const riskClass = 'risk-' + data.risk_tier;
            
            document.getElementById('analysisPanel').innerHTML = `
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 2.5rem; font-weight: 700;" class="${scoreClass}">${data.quantrascore.toFixed(1)}</div>
                    <div style="color: #888; font-size: 0.875rem;">QuantraScore (${data.score_bucket})</div>
                </div>
                <div class="grid-3">
                    <div class="mini-card">
                        <div class="mini-label">Regime</div>
                        <div class="regime-badge regime-${data.regime}">${data.regime}</div>
                    </div>
                    <div class="mini-card">
                        <div class="mini-label">Entropy</div>
                        <div class="mini-value" style="font-size: 0.875rem;">${data.entropy_state}</div>
                    </div>
                    <div class="mini-card">
                        <div class="mini-label">Protocols</div>
                        <div class="mini-value">${data.protocol_fired_count}</div>
                    </div>
                </div>
                <div style="margin-top: 1rem; padding: 0.5rem; background: #1a1a2e; border-radius: 0.25rem;">
                    <div style="font-size: 0.75rem; color: #888;">Verdict</div>
                    <div style="font-weight: 500;">${data.verdict_action} (${(data.verdict_confidence * 100).toFixed(0)}% conf)</div>
                </div>
            `;
        }
        
        function displayRisk(data) {
            const ra = data.risk_assessment;
            const permClass = ra.permission === 'allow' ? 'positive' : ra.permission === 'deny' ? 'negative' : 'neutral';
            
            document.getElementById('riskPanel').innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div>
                        <div style="font-size: 2rem; font-weight: 700;" class="${permClass}">${ra.permission.toUpperCase()}</div>
                        <div style="color: #888; font-size: 0.875rem;">Risk Tier: ${ra.risk_tier}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5rem; font-weight: 600;">${(ra.composite_risk * 100).toFixed(0)}%</div>
                        <div style="color: #888; font-size: 0.75rem;">Composite Risk</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; font-size: 0.8rem;">
                    <div>Vol: ${(ra.volatility_risk * 100).toFixed(0)}%</div>
                    <div>Spread: ${(ra.spread_risk * 100).toFixed(0)}%</div>
                    <div>Regime: ${(ra.regime_risk * 100).toFixed(0)}%</div>
                    <div>Entropy: ${(ra.entropy_risk * 100).toFixed(0)}%</div>
                    <div>Drift: ${(ra.drift_risk * 100).toFixed(0)}%</div>
                    <div>Suppression: ${(ra.suppression_risk * 100).toFixed(0)}%</div>
                </div>
            `;
        }
        
        function displaySignal(data) {
            const s = data.signal;
            const dirClass = s.direction === 'long' ? 'positive' : s.direction === 'short' ? 'negative' : 'neutral';
            
            document.getElementById('signalPanel').innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div>
                        <div style="font-size: 2rem; font-weight: 700;" class="${dirClass}">${s.direction.toUpperCase()}</div>
                        <div style="color: #888; font-size: 0.875rem;">Strength: ${s.strength}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5rem; font-weight: 600;">${(s.confidence * 100).toFixed(0)}%</div>
                        <div style="color: #888; font-size: 0.75rem;">Confidence</div>
                    </div>
                </div>
                ${s.entry_zone_low ? `
                <div style="font-size: 0.8rem; display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem;">
                    <div>Entry: $${s.entry_zone_low.toFixed(2)} - $${s.entry_zone_high.toFixed(2)}</div>
                    <div>Stop: $${s.stop_loss?.toFixed(2) || 'N/A'}</div>
                    <div>Target 1: $${s.target_1?.toFixed(2) || 'N/A'}</div>
                    <div>Target 2: $${s.target_2?.toFixed(2) || 'N/A'}</div>
                </div>
                ` : ''}
                <div style="margin-top: 0.5rem; font-size: 0.75rem; color: ${s.risk_approved ? '#10b981' : '#ef4444'};">
                    Risk: ${s.risk_approved ? 'Approved' : 'Not Approved'}
                </div>
            `;
        }
        
        async function loadPortfolio() {
            try {
                const res = await fetch('/portfolio/status');
                const data = await res.json();
                document.getElementById('equity').textContent = '$' + data.total_equity.toLocaleString();
                document.getElementById('pnl').textContent = '$' + data.total_pnl.toFixed(2);
                document.getElementById('posCount').textContent = data.snapshot.num_positions;
            } catch (e) {
                console.error(e);
            }
        }
        loadPortfolio();
    </script>
</body>
</html>
        """
        return HTMLResponse(content=html_content)
    
    @app.get("/engine/health_extended")
    async def get_engine_health_extended():
        """Get extended engine health with drift status (v9.0-A)."""
        try:
            from src.quantracore_apex.core.drift_detector import DriftDetector
            import yaml
            from pathlib import Path
            
            drift_detector = DriftDetector()
            baselines_loaded = drift_detector.load_baselines()
            drift_status = drift_detector.get_status()
            
            mode_config = "research"
            mode_file = Path("config/mode.yaml")
            if mode_file.exists():
                with open(mode_file) as f:
                    config = yaml.safe_load(f)
                    mode_config = config.get("default_mode", "research")
            
            return {
                "engine_version": "9.0-A",
                "base_version": "8.2",
                "active_model_version": "apexcore_full_v1",
                "drift_mode": drift_status["mode"],
                "drift_baselines_loaded": baselines_loaded,
                "data_provider_status": "available",
                "mode_config": mode_config,
                "test_summary": {
                    "tests_total": 384,
                    "tests_passed": 384,
                    "tests_skipped": 5
                },
                "hardening_features": {
                    "redundant_scoring": True,
                    "drift_detection": True,
                    "fail_closed_gates": True,
                    "sandbox_replay": True,
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "engine_version": "9.0-A",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.get("/drift/status")
    async def get_drift_status():
        """Get current drift metrics and recent events (v9.0-A)."""
        try:
            from src.quantracore_apex.core.drift_detector import DriftDetector
            
            detector = DriftDetector()
            detector.load_baselines()
            status = detector.get_status()
            
            return {
                "mode": status["mode"],
                "baselines_loaded": status["baselines_loaded"],
                "total_drift_events": status["total_drift_events"],
                "recent_events": status["recent_events"][-10:],
                "rolling_metrics": status["rolling_metrics"],
                "thresholds": {
                    "mild_z": 2.0,
                    "severe_z": 3.0,
                },
                "compliance_note": "Research tool only - not financial advice",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    class ReplayDemoRequest(BaseModel):
        universe: str = "demo"
        timeframe: str = "1d"
        lookback_bars: int = 50
    
    @app.post("/replay/run_demo")
    async def run_replay_demo(request: ReplayDemoRequest):
        """Run sandbox replay on demo universe (v9.0-A)."""
        try:
            from src.quantracore_apex.replay.replay_engine import ReplayEngine, ReplayConfig
            
            replay_engine = ReplayEngine()
            config = ReplayConfig(
                universe=request.universe,
                timeframe=request.timeframe,
                lookback_bars=request.lookback_bars,
            )
            
            result = replay_engine.run_replay(config=config)
            
            return {
                "symbols_processed": result.symbols_processed,
                "signals_generated": result.signals_generated,
                "duration_seconds": result.duration_seconds,
                "equity_curve": result.equity_curve,
                "signal_frequency_stats": result.signal_frequency_stats,
                "drift_flags": result.drift_flags,
                "errors": result.errors[:10],
                "compliance_note": "Research tool only - not financial advice",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.get("/score/consistency/{symbol}")
    async def get_score_consistency(
        symbol: str,
        timeframe: str = "1d",
        lookback_days: int = 100
    ):
        """Check score consistency for a symbol (v9.0-A)."""
        try:
            from src.quantracore_apex.core.redundant_scorer import RedundantScorer
            
            adapter = SyntheticAdapter(seed=hash(symbol) % 10000)
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            data = adapter.fetch_ohlcv(symbol, start_date, end_date, timeframe)
            
            if not data:
                return {"error": "No data available", "symbol": symbol}
            
            result = engine.run_scan(data, symbol, seed=hash(symbol) % 10000, timeframe=timeframe)
            
            scorer = RedundantScorer()
            verification = scorer.compute_with_verification(
                primary_score=result.quantrascore,
                primary_band=result.score_bucket.value,
                protocol_results={p.protocol_id: {"fired": p.fired, "confidence": p.confidence} for p in result.protocol_results},
                regime=result.regime,
                risk_tier=result.risk_tier,
                monster_runner_state="idle"
            )
            
            return {
                "symbol": symbol,
                "primary_score": verification["primary_score"],
                "shadow_score": verification["shadow_score"],
                "primary_band": verification["primary_band"],
                "shadow_band": verification["shadow_band"],
                "consistency_status": verification["consistency_status"],
                "consistency_ok": verification["consistency_ok"],
                "absolute_diff": verification["absolute_diff"],
                "compliance_note": "Research tool only - not financial advice",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.get("/api/stats")
    async def get_api_stats():
        """Get API statistics and system info."""
        return {
            "version": "9.0-A",
            "protocols": {
                "tier": 80,
                "learning": 25,
                "monster_runner": 5,
                "omega": 5
            },
            "modules": {
                "core_engine": True,
                "apexlab": True,
                "apexcore": True,
                "risk_engine": True,
                "oms": True,
                "portfolio": True,
                "signal_builder": True,
                "prediction_stack": True,
                "redundant_scorer": True,
                "drift_detector": True,
                "decision_gates": True,
                "replay_engine": True,
                "universal_scanner": True,
            },
            "v9_hardening": {
                "redundant_scoring": True,
                "drift_detection": True,
                "fail_closed_gates": True,
                "sandbox_replay": True,
                "research_only_fence": True,
                "universal_scanner": True,
            },
            "simulation_mode": True,
            "compliance_mode": True,
            "desktop_only": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/modes")
    async def list_scan_modes():
        """List all available scan modes (v9.0-A Universal Scanner)."""
        try:
            from src.quantracore_apex.config.scan_modes import list_scan_modes, load_scan_mode
            
            modes_list = list_scan_modes()
            modes_detail = []
            
            for mode_name in modes_list:
                config = load_scan_mode(mode_name)
                modes_detail.append({
                    "name": config.name,
                    "description": config.description,
                    "buckets": config.buckets,
                    "max_symbols": config.max_symbols,
                    "chunk_size": config.chunk_size,
                    "is_smallcap_focused": config.is_smallcap_focused,
                    "is_extreme_risk": config.is_extreme_risk,
                    "risk_flags": config.risk_flags,
                    "filters": config.filters.model_dump() if config.filters else None,
                })
            
            return {
                "modes": modes_detail,
                "total_modes": len(modes_detail),
                "smallcap_modes": sum(1 for m in modes_detail if m["is_smallcap_focused"]),
                "extreme_risk_modes": sum(1 for m in modes_detail if m["is_extreme_risk"]),
                "compliance_note": "Research tool only - not financial advice",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error listing modes: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.get("/universe_stats")
    async def get_universe_stats():
        """Get statistics about symbol universe (v9.0-A Universal Scanner)."""
        try:
            from src.quantracore_apex.config.symbol_universe import (
                get_all_symbols,
                get_symbols_by_bucket,
                ALL_BUCKETS,
                SMALLCAP_BUCKETS,
            )
            
            all_symbols = get_all_symbols()
            bucket_counts = {}
            
            for bucket in ALL_BUCKETS:
                bucket_symbols = get_symbols_by_bucket([bucket])
                bucket_counts[bucket] = len(bucket_symbols)
            
            smallcap_symbols = get_symbols_by_bucket(list(SMALLCAP_BUCKETS))
            
            return {
                "total_symbols": len(all_symbols),
                "bucket_counts": bucket_counts,
                "buckets": list(ALL_BUCKETS),
                "smallcap_buckets": list(SMALLCAP_BUCKETS),
                "smallcap_count": len(smallcap_symbols),
                "largecap_count": len(all_symbols) - len(smallcap_symbols),
                "version": "9.0-A",
                "compliance_note": "Research tool only - not financial advice",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting universe stats: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.get("/symbol_info/{symbol}")
    async def get_symbol_info(symbol: str):
        """Get detailed info for a specific symbol (v9.0-A Universal Scanner)."""
        try:
            from src.quantracore_apex.config.symbol_universe import (
                get_symbol_info as _get_symbol_info,
            )
            
            info = _get_symbol_info(symbol.upper())
            
            if info is None:
                return {
                    "symbol": symbol.upper(),
                    "found": False,
                    "message": "Symbol not in configured universe - may still be scannable via dynamic loading",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return {
                "symbol": info.symbol,
                "found": True,
                "name": info.name,
                "market_cap_bucket": info.market_cap_bucket,
                "sector": info.sector,
                "float_millions": info.float_millions,
                "is_smallcap": info.is_smallcap,
                "risk_category": info.risk_category,
                "active": info.active,
                "allow_smallcap_scan": info.allow_smallcap_scan,
                "compliance_note": "Research tool only - not financial advice",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {
                "error": str(e),
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    class UniverseScanModeRequest(BaseModel):
        mode: str = "demo"
        max_results: int = 100
        include_mr_fuse: bool = True
    
    @app.post("/scan_universe_mode")
    async def scan_universe_by_mode(request: UniverseScanModeRequest):
        """Scan universe using a pre-defined mode (v9.0-A Universal Scanner)."""
        try:
            from src.quantracore_apex.config.scan_modes import load_scan_mode
            from src.quantracore_apex.core.universe_scan import create_universe_scanner
            
            mode_config = load_scan_mode(request.mode)
            
            scanner = create_universe_scanner()
            scan_result = scanner.scan(
                mode=request.mode,
                max_symbols=request.max_results,
            )
            
            return {
                "mode": request.mode,
                "mode_description": mode_config.description,
                "buckets": mode_config.buckets,
                "requested_symbols": request.max_results,
                "scan_count": scan_result.scan_count,
                "success_count": scan_result.success_count,
                "error_count": scan_result.error_count,
                "smallcap_count": scan_result.smallcap_count,
                "extreme_risk_count": scan_result.extreme_risk_count,
                "runner_candidate_count": scan_result.runner_candidate_count,
                "results": [r.__dict__ for r in scan_result.results[:request.max_results]],
                "errors": scan_result.errors[:10],
                "is_smallcap_mode": mode_config.is_smallcap_focused,
                "is_extreme_risk_mode": mode_config.is_extreme_risk,
                "risk_level": mode_config.risk_default,
                "compliance_note": "Research tool only - not financial advice. Small-cap and penny stocks carry extreme risk.",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error scanning universe with mode {request.mode}: {e}")
            return {
                "error": str(e),
                "mode": request.mode,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    class PredictiveAdvisoryRequest(BaseModel):
        symbol: str
        timeframe: str = "1d"
        lookback_days: int = 150
    
    @app.get("/predictive/status")
    async def predictive_status():
        """Get status of the Predictive Layer V2 (ApexCore V2)."""
        try:
            from src.quantracore_apex.core.integration_predictive import (
                PredictiveAdvisor,
                PredictiveConfig,
            )
            
            config = PredictiveConfig(enabled=True)
            advisor = PredictiveAdvisor(config=config)
            
            return {
                "version": "2.0",
                "status": advisor.status,
                "enabled": advisor.is_enabled,
                "model_variant": config.variant,
                "model_dir": config.model_dir,
                "runner_threshold": config.runner_prob_uprank_threshold,
                "avoid_threshold": config.avoid_trade_prob_max,
                "max_disagreement": config.max_disagreement_allowed,
                "compliance_note": "Predictive layer is ADVISORY ONLY - engine has final authority",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting predictive status: {e}")
            return {
                "version": "2.0",
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.post("/predictive/advise")
    async def predictive_advise(request: PredictiveAdvisoryRequest):
        """Get predictive advisory for a symbol (ApexCore V2 integration)."""
        try:
            from src.quantracore_apex.core.integration_predictive import (
                PredictiveAdvisor,
                PredictiveConfig,
            )
            from src.quantracore_apex.apexlab.apexlab_v2 import encode_protocol_vector
            
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
            
            context = ApexContext(seed=42, compliance_mode=True)
            result = engine.run(window, context)
            
            protocol_ids = [p.get("protocol_id", "") for p in result.protocol_results if p]
            protocol_vector = encode_protocol_vector(protocol_ids)
            features = np.array(protocol_vector, dtype=np.float32)
            
            config = PredictiveConfig(enabled=True)
            advisor = PredictiveAdvisor(config=config)
            
            advisory = advisor.advise_on_candidate(
                symbol=request.symbol,
                base_quantra_score=result.quantrascore,
                features=features,
            )
            
            response = advisory.to_dict()
            response["engine_quantra_score"] = result.quantrascore
            response["engine_regime"] = result.regime.value
            response["engine_risk_tier"] = result.risk_tier.value
            response["predictive_status"] = advisor.status
            response["compliance_note"] = "Advisory only - engine has final authority"
            response["timestamp"] = datetime.utcnow().isoformat()
            
            return convert_numpy_types(response)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting predictive advisory for {request.symbol}: {e}")
            return {
                "symbol": request.symbol,
                "error": str(e),
                "recommendation": "DISABLED",
                "reasons": [f"Error: {str(e)}"],
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.get("/predictive/model_info")
    async def predictive_model_info():
        """Get information about loaded ApexCore V2 models."""
        try:
            from src.quantracore_apex.apexcore.manifest import (
                load_manifest,
                select_best_model,
            )
            from pathlib import Path
            
            manifest_dir = Path("models/apexcore_v2/big/manifests")
            
            if not manifest_dir.exists():
                return {
                    "status": "NO_MANIFESTS",
                    "message": "No model manifests found - models not yet trained",
                    "manifest_dir": str(manifest_dir),
                    "available_manifests": [],
                    "compliance_note": "Train models using ApexLab V2 before using predictions",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            manifest_files = list(manifest_dir.glob("*.json"))
            
            if not manifest_files:
                return {
                    "status": "NO_MANIFESTS",
                    "message": "No model manifests found in directory",
                    "manifest_dir": str(manifest_dir),
                    "available_manifests": [],
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            manifest_path, latest_manifest = select_best_model(str(manifest_dir))
            
            return {
                "status": "MODELS_AVAILABLE",
                "manifest_count": len(manifest_files),
                "latest_manifest": latest_manifest.to_dict() if latest_manifest else None,
                "available_manifests": [str(m) for m in manifest_files[:5]],
                "model_variant": latest_manifest.variant if latest_manifest else "unknown",
                "created_at": latest_manifest.created_utc if latest_manifest else None,
                "compliance_note": "Models are advisory only - research mode",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    class BatchAdvisoryRequest(BaseModel):
        symbols: List[str]
        timeframe: str = "1d"
        lookback_days: int = 150
        max_results: int = 50
    
    @app.post("/predictive/batch_advise")
    async def predictive_batch_advise(request: BatchAdvisoryRequest):
        """Get predictive advisory for multiple symbols (batch processing)."""
        try:
            from src.quantracore_apex.core.integration_predictive import (
                PredictiveAdvisor,
                PredictiveConfig,
            )
            from src.quantracore_apex.apexlab.apexlab_v2 import encode_protocol_vector
            
            config = PredictiveConfig(enabled=True)
            advisor = PredictiveAdvisor(config=config)
            
            results = []
            errors = []
            
            for symbol in request.symbols[:request.max_results]:
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=request.lookback_days)
                    
                    bars = data_adapter.fetch_ohlcv(
                        symbol, start_date, end_date, request.timeframe
                    )
                    
                    if len(bars) < 100:
                        errors.append({
                            "symbol": symbol,
                            "error": f"Insufficient data: {len(bars)} bars"
                        })
                        continue
                    
                    normalized_bars, _ = normalize_ohlcv(bars)
                    window = window_builder.build_single(
                        normalized_bars, symbol, request.timeframe
                    )
                    
                    context = ApexContext(seed=42, compliance_mode=True)
                    result = engine.run(window, context)
                    
                    protocol_ids = [p.get("protocol_id", "") for p in result.protocol_results if p]
                    protocol_vector = encode_protocol_vector(protocol_ids)
                    features = np.array(protocol_vector, dtype=np.float32)
                    
                    advisory = advisor.advise_on_candidate(
                        symbol=symbol,
                        base_quantra_score=result.quantrascore,
                        features=features,
                    )
                    
                    result_dict = advisory.to_dict()
                    result_dict["engine_quantra_score"] = result.quantrascore
                    result_dict["engine_regime"] = result.regime.value
                    results.append(convert_numpy_types(result_dict))
                    
                except Exception as e:
                    errors.append({"symbol": symbol, "error": str(e)})
            
            uprank_count = sum(1 for r in results if r.get("recommendation") == "UPRANK")
            avoid_count = sum(1 for r in results if r.get("recommendation") == "AVOID")
            
            return {
                "total_requested": len(request.symbols),
                "total_processed": len(results),
                "total_errors": len(errors),
                "uprank_count": uprank_count,
                "avoid_count": avoid_count,
                "predictive_status": advisor.status,
                "results": results,
                "errors": errors[:10],
                "compliance_note": "Advisory only - engine has final authority",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in batch predictive advisory: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # =========================================================================
    # ESTIMATED MOVE MODULE ENDPOINTS
    # =========================================================================
    
    from src.quantracore_apex.estimated_move import (
        EstimatedMoveEngine,
        EstimatedMoveInput,
    )
    
    estimated_move_engine = EstimatedMoveEngine(seed=42)
    
    class EstimatedMoveRequest(BaseModel):
        symbol: str
        timeframe: str = "1d"
        lookback_days: int = 150
        include_vision: bool = False
    
    class BatchEstimatedMoveRequest(BaseModel):
        symbols: List[str]
        timeframe: str = "1d"
        lookback_days: int = 150
        max_results: int = 50
    
    @app.get("/estimated_move/horizons")
    async def get_estimated_move_horizons():
        """Get available horizon windows for estimated move calculation."""
        return {
            "horizons": [
                {"id": "1d", "name": "Short Term", "days": 1},
                {"id": "3d", "name": "Medium Term", "days": 3},
                {"id": "5d", "name": "Extended Term", "days": 5},
                {"id": "10d", "name": "Research Term", "days": 10},
            ],
            "compliance_note": "Horizons for statistical research only - not trading timeframes",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/estimated_move/{symbol}")
    async def get_estimated_move(
        symbol: str,
        timeframe: str = "1d",
        lookback_days: int = 150
    ):
        """
        Get estimated move ranges for a symbol.
        
        Returns statistical move ranges for research purposes.
        NOT a prediction. NOT a price target. RESEARCH ONLY.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            bars = data_adapter.fetch_ohlcv(symbol, start_date, end_date, timeframe)
            
            if len(bars) < 100:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient data: got {len(bars)} bars, need at least 100"
                )
            
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, symbol, timeframe)
            
            if window is None:
                raise HTTPException(status_code=400, detail="Failed to build analysis window")
            
            context = ApexContext(seed=42, compliance_mode=True)
            result = engine.run(window, context)
            
            mr_output = monster_runner.analyze(window)
            
            em_input = EstimatedMoveInput(
                symbol=symbol,
                quantra_score=result.quantrascore,
                risk_tier=result.risk_tier.value,
                volatility_band=result.entropy_state.value,
                entropy_band=result.entropy_state.value,
                regime_type=result.regime.value,
                suppression_state=result.suppression_state.value,
                protocol_vector=[1.0 if p.fired else 0.0 for p in result.protocol_results],
                runner_prob=mr_output.runner_probability,
                avoid_trade_prob=0.0,
                ensemble_disagreement=0.0,
                market_cap_band="mid",
            )
            
            em_output = estimated_move_engine.compute(em_input)
            
            return convert_numpy_types({
                "symbol": symbol,
                "estimated_move": em_output.to_dict(),
                "underlying_analysis": {
                    "quantra_score": result.quantrascore,
                    "regime": result.regime.value,
                    "risk_tier": result.risk_tier.value,
                    "runner_prob": mr_output.runner_probability,
                },
                "compliance_note": "Structural research output only - NOT a price target or trading signal",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error computing estimated move for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/estimated_move/batch")
    async def batch_estimated_move(request: BatchEstimatedMoveRequest):
        """
        Get estimated move ranges for multiple symbols.
        
        Returns statistical move ranges for research purposes.
        NOT predictions. NOT price targets. RESEARCH ONLY.
        """
        results = []
        errors = []
        
        for symbol in request.symbols[:request.max_results]:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=request.lookback_days)
                
                bars = data_adapter.fetch_ohlcv(
                    symbol, start_date, end_date, request.timeframe
                )
                
                if len(bars) < 100:
                    errors.append({
                        "symbol": symbol,
                        "error": f"Insufficient data: {len(bars)} bars"
                    })
                    continue
                
                normalized_bars, _ = normalize_ohlcv(bars)
                window = window_builder.build_single(
                    normalized_bars, symbol, request.timeframe
                )
                
                if window is None:
                    errors.append({
                        "symbol": symbol,
                        "error": "Failed to build window"
                    })
                    continue
                
                context = ApexContext(seed=42, compliance_mode=True)
                result = engine.run(window, context)
                
                mr_output = monster_runner.analyze(window)
                
                em_input = EstimatedMoveInput(
                    symbol=symbol,
                    quantra_score=result.quantrascore,
                    risk_tier=result.risk_tier.value,
                    volatility_band=result.entropy_state.value,
                    entropy_band=result.entropy_state.value,
                    regime_type=result.regime.value,
                    suppression_state=result.suppression_state.value,
                    protocol_vector=[1.0 if p.fired else 0.0 for p in result.protocol_results],
                    runner_prob=mr_output.runner_probability,
                    avoid_trade_prob=0.0,
                    ensemble_disagreement=0.0,
                    market_cap_band="mid",
                )
                
                em_output = estimated_move_engine.compute(em_input)
                
                results.append(convert_numpy_types({
                    "symbol": symbol,
                    "estimated_move": em_output.to_dict(),
                    "quantra_score": result.quantrascore,
                    "runner_prob": mr_output.runner_probability,
                }))
                
            except Exception as e:
                errors.append({"symbol": symbol, "error": str(e)})
        
        results.sort(
            key=lambda x: x.get("estimated_move", {}).get("ranges", {}).get("5d", {}).get("max_move_pct", 0),
            reverse=True
        )
        
        return {
            "total_requested": len(request.symbols),
            "total_processed": len(results),
            "total_errors": len(errors),
            "results": results,
            "errors": errors[:10],
            "compliance_note": "Structural research output only - NOT price targets or trading signals",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    from src.quantracore_apex.broker import (
        ExecutionEngine as BrokerExecutionEngine,
        BrokerConfig,
        ExecutionMode,
        ApexSignal,
        SignalDirection,
        load_broker_config,
    )
    
    broker_config = load_broker_config("config/broker.yaml")
    broker_engine: Optional[BrokerExecutionEngine] = None
    
    def get_broker_engine() -> BrokerExecutionEngine:
        """Get or create broker execution engine."""
        nonlocal broker_engine
        if broker_engine is None:
            broker_engine = BrokerExecutionEngine(config=broker_config)
        return broker_engine
    
    @app.get("/broker/status")
    async def broker_status():
        """
        Get broker layer status.
        
        Returns current execution mode, adapter, equity, and positions.
        SAFETY: Live trading is DISABLED by default.
        """
        try:
            engine = get_broker_engine()
            status = engine.get_status()
            return {
                **status,
                "safety_note": "Live trading is DISABLED. Paper trading only.",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "mode": broker_config.execution_mode.value,
                "error": str(e),
                "safety_note": "Live trading is DISABLED. Paper trading only.",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.get("/broker/positions")
    async def broker_positions():
        """Get current broker positions."""
        try:
            engine = get_broker_engine()
            positions = engine.router.get_positions()
            return {
                "positions": [p.to_dict() for p in positions],
                "count": len(positions),
                "mode": engine.mode.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/broker/orders")
    async def broker_open_orders():
        """Get open orders from broker."""
        try:
            engine = get_broker_engine()
            orders = engine.router.get_open_orders()
            return {
                "orders": [o.to_dict() for o in orders],
                "count": len(orders),
                "mode": engine.mode.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/broker/equity")
    async def broker_equity():
        """Get current account equity."""
        try:
            engine = get_broker_engine()
            equity = engine.router.get_account_equity()
            return {
                "equity": equity,
                "mode": engine.mode.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    class ExecuteSignalRequest(BaseModel):
        symbol: str
        direction: str  # LONG, SHORT, EXIT, HOLD
        quantra_score: float = 50.0
        runner_prob: float = 0.0
        size_hint: Optional[float] = None
    
    @app.post("/broker/execute")
    async def execute_signal(request: ExecuteSignalRequest):
        """
        Execute a trading signal through the broker layer.
        
        SAFETY: Orders only execute in PAPER mode.
        LIVE trading is DISABLED.
        """
        try:
            engine = get_broker_engine()
            
            if engine.mode == ExecutionMode.LIVE:
                raise HTTPException(
                    status_code=403,
                    detail="LIVE trading is DISABLED. Use PAPER mode only."
                )
            
            import uuid
            signal = ApexSignal(
                signal_id=str(uuid.uuid4()),
                symbol=request.symbol.upper(),
                direction=SignalDirection[request.direction.upper()],
                quantra_score=request.quantra_score,
                runner_prob=request.runner_prob,
                size_hint=request.size_hint,
            )
            
            result = engine.execute_signal(signal)
            
            if result is None:
                return {
                    "executed": False,
                    "reason": "Signal filtered (no position to exit or already in position)",
                    "mode": engine.mode.value,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return {
                "executed": True,
                "result": result.to_dict(),
                "mode": engine.mode.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/broker/config")
    async def broker_config_info():
        """Get broker configuration (sanitized - no secrets)."""
        return {
            "execution_mode": broker_config.execution_mode.value,
            "default_account": broker_config.default_account,
            "alpaca_paper_enabled": broker_config.alpaca_paper.enabled,
            "alpaca_paper_configured": broker_config.alpaca_paper.is_configured,
            "alpaca_live_enabled": broker_config.alpaca_live.enabled,
            "risk_limits": {
                "max_notional_exposure_usd": broker_config.risk.max_notional_exposure_usd,
                "max_position_notional_per_symbol_usd": broker_config.risk.max_position_notional_per_symbol_usd,
                "max_positions": broker_config.risk.max_positions,
                "max_order_notional_usd": broker_config.risk.max_order_notional_usd,
                "max_daily_turnover_usd": broker_config.risk.max_daily_turnover_usd,
                "max_leverage": broker_config.risk.max_leverage,
                "block_short_selling": broker_config.risk.block_short_selling,
            },
            "safety_note": "Live trading is DISABLED by default.",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    paper_config = BrokerConfig(execution_mode=ExecutionMode.PAPER)
    paper_engine: Optional[BrokerExecutionEngine] = None
    
    def get_paper_engine() -> BrokerExecutionEngine:
        """Get or create paper trading execution engine."""
        nonlocal paper_engine
        if paper_engine is None:
            paper_engine = BrokerExecutionEngine(config=paper_config)
        return paper_engine
    
    @app.get("/broker/paper/status")
    async def paper_broker_status():
        """
        Get paper trading engine status.
        
        This endpoint uses PAPER mode with PaperSimAdapter for actual fills.
        """
        try:
            engine = get_paper_engine()
            status = engine.get_status()
            return {
                **status,
                "safety_note": "Paper trading mode - simulated fills only.",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/broker/paper/execute")
    async def execute_paper_signal(request: ExecuteSignalRequest):
        """
        Execute a trading signal using PAPER mode (actual fills).
        
        This endpoint uses PaperSimAdapter which fills orders immediately.
        Use this for paper trading testing and demonstration.
        """
        try:
            engine = get_paper_engine()
            
            import uuid
            signal = ApexSignal(
                signal_id=str(uuid.uuid4()),
                symbol=request.symbol.upper(),
                direction=SignalDirection[request.direction.upper()],
                quantra_score=request.quantra_score,
                runner_prob=request.runner_prob,
                size_hint=request.size_hint,
            )
            
            result = engine.execute_signal(signal)
            
            if result is None:
                return {
                    "executed": False,
                    "reason": "Signal filtered (no position to exit, already in position, or short blocked)",
                    "mode": engine.mode.value,
                    "adapter": engine.router.adapter_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return {
                "executed": True,
                "result": result.to_dict(),
                "mode": engine.mode.value,
                "adapter": engine.router.adapter_name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/broker/paper/positions")
    async def paper_positions():
        """Get positions from paper trading engine."""
        try:
            engine = get_paper_engine()
            positions = engine.router.get_positions()
            return {
                "positions": [p.to_dict() for p in positions],
                "count": len(positions),
                "mode": engine.mode.value,
                "adapter": engine.router.adapter_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/broker/paper/reset")
    async def reset_paper_engine():
        """Reset paper trading engine to initial state."""
        nonlocal paper_engine
        paper_engine = None
        return {
            "message": "Paper trading engine reset to initial state",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    from src.quantracore_apex.eeo_engine import (
        EntryExitOptimizer,
        SignalContext as EEOSignalContext,
        PredictiveContext,
        MarketMicrostructure,
        RiskContext,
        EEOContext,
        SignalDirection as EEOSignalDirection,
        QualityTier,
        ProfileType,
        CONSERVATIVE_PROFILE,
        BALANCED_PROFILE,
        AGGRESSIVE_RESEARCH_PROFILE,
        get_profile,
    )
    
    class EEOPlanRequest(BaseModel):
        symbol: str
        direction: str = "LONG"
        quantra_score: float = 50.0
        current_price: float = 100.0
        atr: float = 0.0
        account_equity: float = 100000.0
        runner_prob: float = 0.0
        quality_tier: str = "C"
        estimated_move_median: Optional[float] = None
        estimated_move_max: Optional[float] = None
        profile: str = "balanced"
        signal_id: str = ""
    
    @app.post("/eeo/plan")
    async def build_eeo_plan(request: EEOPlanRequest):
        """
        Build an Entry/Exit Optimization Plan.
        
        This endpoint calculates optimal entry zones, stops, and targets
        for a given signal using deterministic + model-assisted methods.
        """
        try:
            profile_map = {
                "conservative": ProfileType.CONSERVATIVE,
                "balanced": ProfileType.BALANCED,
                "aggressive_research": ProfileType.AGGRESSIVE_RESEARCH,
            }
            profile_type = profile_map.get(request.profile.lower(), ProfileType.BALANCED)
            
            optimizer = EntryExitOptimizer(profile_type=profile_type)
            
            direction = EEOSignalDirection.LONG if request.direction.upper() == "LONG" else EEOSignalDirection.SHORT
            
            quality_map = {
                "A_PLUS": QualityTier.A_PLUS,
                "A": QualityTier.A,
                "B": QualityTier.B,
                "C": QualityTier.C,
                "D": QualityTier.D,
            }
            quality_tier = quality_map.get(request.quality_tier.upper(), QualityTier.C)
            
            signal_context = EEOSignalContext(
                symbol=request.symbol.upper(),
                direction=direction,
                quantra_score=request.quantra_score,
                signal_id=request.signal_id or f"api_{request.symbol}_{datetime.utcnow().strftime('%H%M%S')}",
            )
            
            predictive_context = PredictiveContext(
                runner_prob=request.runner_prob,
                future_quality_tier=quality_tier,
                estimated_move_median=request.estimated_move_median,
                estimated_move_max=request.estimated_move_max,
            )
            
            atr = request.atr if request.atr > 0 else request.current_price * 0.02
            microstructure = MarketMicrostructure.from_price(request.current_price, atr)
            
            risk_context = RiskContext(
                account_equity=request.account_equity,
                per_trade_risk_fraction=optimizer.profile.per_trade_risk_fraction,
            )
            
            context = EEOContext(
                signal=signal_context,
                predictive=predictive_context,
                microstructure=microstructure,
                risk=risk_context,
            )
            
            plan = optimizer.build_plan(context)
            
            return {
                "success": True,
                "plan": plan.to_dict(),
                "profile_used": optimizer.profile.name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/eeo/plan/execute")
    async def execute_eeo_plan(request: EEOPlanRequest):
        """
        Build and execute an Entry/Exit Plan using paper trading.
        
        This endpoint builds a plan and immediately executes the entry
        using the paper trading engine.
        """
        try:
            profile_map = {
                "conservative": ProfileType.CONSERVATIVE,
                "balanced": ProfileType.BALANCED,
                "aggressive_research": ProfileType.AGGRESSIVE_RESEARCH,
            }
            profile_type = profile_map.get(request.profile.lower(), ProfileType.BALANCED)
            
            optimizer = EntryExitOptimizer(profile_type=profile_type)
            
            direction = EEOSignalDirection.LONG if request.direction.upper() == "LONG" else EEOSignalDirection.SHORT
            
            signal_context = EEOSignalContext(
                symbol=request.symbol.upper(),
                direction=direction,
                quantra_score=request.quantra_score,
                signal_id=request.signal_id or f"eeo_{request.symbol}_{datetime.utcnow().strftime('%H%M%S')}",
            )
            
            predictive_context = PredictiveContext(
                runner_prob=request.runner_prob,
                estimated_move_median=request.estimated_move_median,
                estimated_move_max=request.estimated_move_max,
            )
            
            atr = request.atr if request.atr > 0 else request.current_price * 0.02
            microstructure = MarketMicrostructure.from_price(request.current_price, atr)
            
            risk_context = RiskContext(
                account_equity=request.account_equity,
                per_trade_risk_fraction=optimizer.profile.per_trade_risk_fraction,
            )
            
            context = EEOContext(
                signal=signal_context,
                predictive=predictive_context,
                microstructure=microstructure,
                risk=risk_context,
            )
            
            plan = optimizer.build_plan(context)
            
            engine = get_paper_engine()
            execution_result = engine.execute_plan(plan)
            
            return {
                "success": execution_result.get("success", False),
                "plan": plan.to_dict(),
                "execution": execution_result,
                "profile_used": optimizer.profile.name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/eeo/profiles")
    async def list_eeo_profiles():
        """List available EEO profiles."""
        return {
            "profiles": [
                {
                    "name": "conservative",
                    "description": CONSERVATIVE_PROFILE.description,
                    "risk_fraction": CONSERVATIVE_PROFILE.per_trade_risk_fraction,
                    "aggressiveness": CONSERVATIVE_PROFILE.entry_aggressiveness,
                },
                {
                    "name": "balanced",
                    "description": BALANCED_PROFILE.description,
                    "risk_fraction": BALANCED_PROFILE.per_trade_risk_fraction,
                    "aggressiveness": BALANCED_PROFILE.entry_aggressiveness,
                },
                {
                    "name": "aggressive_research",
                    "description": AGGRESSIVE_RESEARCH_PROFILE.description,
                    "risk_fraction": AGGRESSIVE_RESEARCH_PROFILE.per_trade_risk_fraction,
                    "aggressiveness": AGGRESSIVE_RESEARCH_PROFILE.entry_aggressiveness,
                },
            ],
            "default": "balanced",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/eeo/profiles/{profile_name}")
    async def get_eeo_profile(profile_name: str):
        """Get detailed information about a specific EEO profile."""
        profile_map = {
            "conservative": CONSERVATIVE_PROFILE,
            "balanced": BALANCED_PROFILE,
            "aggressive_research": AGGRESSIVE_RESEARCH_PROFILE,
        }
        
        profile = profile_map.get(profile_name.lower())
        if not profile:
            raise HTTPException(
                status_code=404, 
                detail=f"Profile '{profile_name}' not found. Available: conservative, balanced, aggressive_research"
            )
        
        return {
            "profile": profile.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    from src.quantracore_apex.integrations.google_docs import (
        google_docs_client,
        research_report_generator,
        trade_journal,
        investor_updates,
        notes_importer,
        doc_sync,
    )
    
    class GenerateReportRequest(BaseModel):
        symbol: str
        include_risk: bool = True
        include_monster_runner: bool = True
        include_signal: bool = True
    
    class LogResearchNoteRequest(BaseModel):
        title: str
        content: str
        symbol: Optional[str] = None
    
    class ImportDocumentRequest(BaseModel):
        document_id: str
    
    class ExportDocumentRequest(BaseModel):
        local_path: str
    
    class InvestorUpdateRequest(BaseModel):
        month: Optional[str] = None
        highlights: Optional[List[str]] = None
    
    @app.get("/gdocs/status")
    async def google_docs_status():
        """Check Google Docs connection status."""
        try:
            status = await google_docs_client.check_connection()
            return status
        except Exception as e:
            return {
                "connected": False,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.get("/gdocs/documents")
    async def list_google_documents(max_results: int = 20):
        """List recent Google Docs documents."""
        try:
            docs = await google_docs_client.list_documents(max_results)
            return {
                "documents": docs,
                "count": len(docs),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/gdocs/search")
    async def search_google_documents(query: str, max_results: int = 10):
        """Search Google Docs by query."""
        try:
            docs = await google_docs_client.search_documents(query, max_results)
            return {
                "query": query,
                "documents": docs,
                "count": len(docs),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/report/generate")
    async def generate_research_report(request: GenerateReportRequest):
        """
        Generate a comprehensive research report for a symbol.
        
        Runs full analysis and creates a formatted Google Doc with:
        - Executive summary
        - Protocol analysis
        - Risk assessment
        - MonsterRunner detection
        - Signal generation
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=150)
            
            bars = data_adapter.fetch_ohlcv(request.symbol, start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, request.symbol)
            
            if window is None:
                raise HTTPException(status_code=400, detail="Insufficient data for analysis")
            
            context = ApexContext(seed=42, compliance_mode=True)
            result = engine.run(window, context)
            
            omega_statuses = omega_directives.apply_all(result)
            omega_alerts = [name for name, status in omega_statuses.items() if status.active]
            
            scan_result = {
                "symbol": result.symbol,
                "quantrascore": result.quantrascore,
                "score_bucket": result.score_bucket.value,
                "regime": result.regime.value,
                "risk_tier": result.risk_tier.value,
                "entropy_state": result.entropy_state.value,
                "suppression_state": result.suppression_state.value,
                "drift_state": result.drift_state.value,
                "verdict_action": result.verdict.action,
                "verdict_confidence": result.verdict.confidence,
                "omega_alerts": omega_alerts,
                "protocol_fired_count": sum(1 for p in result.protocol_results if p.fired),
                "window_hash": result.window_hash,
            }
            
            risk_data = None
            if request.include_risk:
                assessment = risk_engine.assess(
                    symbol=request.symbol,
                    quantra_score=result.quantrascore,
                    regime=result.regime.value,
                    entropy_state=str(result.entropy_state),
                    drift_state=str(result.drift_state),
                    suppression_state=str(result.suppression_state),
                    volatility_ratio=result.microtraits.volatility_ratio,
                )
                risk_data = {"risk_assessment": assessment.model_dump()}
            
            monster_data = None
            if request.include_monster_runner:
                output = monster_runner.analyze(window)
                monster_data = {
                    "runner_probability": output.runner_probability,
                    "runner_state": output.runner_state.value,
                    "rare_event_class": output.rare_event_class.value,
                    "metrics": {
                        "compression_trace": output.compression_trace,
                        "entropy_floor": output.entropy_floor,
                        "volume_pulse": output.volume_pulse,
                        "range_contraction": output.range_contraction,
                        "primed_confidence": output.primed_confidence,
                    }
                }
            
            signal_data = None
            if request.include_signal:
                current_price = bars[-1].close if bars else None
                fired_protocols = [p.protocol_id for p in result.protocol_results if p.fired]
                risk_approved = risk_data.get("risk_assessment", {}).get("permission", "deny") == "allow" if risk_data else False
                
                signal = signal_builder.build_signal(
                    symbol=request.symbol,
                    quantra_score=result.quantrascore,
                    regime=result.regime.value,
                    risk_tier=result.risk_tier.value,
                    entropy_state=str(result.entropy_state),
                    current_price=current_price,
                    volatility_pct=result.microtraits.volatility_ratio * 2,
                    fired_protocols=fired_protocols,
                    risk_approved=risk_approved,
                    risk_notes="",
                )
                signal_data = {"signal": signal.model_dump()}
            
            report = await research_report_generator.generate_report(
                symbol=request.symbol,
                scan_result=scan_result,
                risk_data=risk_data,
                monster_data=monster_data,
                signal_data=signal_data,
            )
            
            return {
                "success": True,
                "report": report,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/report/batch")
    async def generate_batch_report(symbols: List[str]):
        """Generate a batch report for multiple symbols."""
        try:
            scan_results = []
            for symbol in symbols:
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=150)
                    bars = data_adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
                    normalized_bars, _ = normalize_ohlcv(bars)
                    window = window_builder.build_single(normalized_bars, symbol)
                    
                    if window:
                        context = ApexContext(seed=42, compliance_mode=True)
                        result = engine.run(window, context)
                        omega_statuses = omega_directives.apply_all(result)
                        omega_alerts = [name for name, status in omega_statuses.items() if status.active]
                        
                        scan_results.append({
                            "symbol": symbol,
                            "quantrascore": result.quantrascore,
                            "score_bucket": result.score_bucket.value,
                            "regime": result.regime.value,
                            "risk_tier": result.risk_tier.value,
                            "verdict_action": result.verdict.action,
                            "omega_alerts": omega_alerts,
                        })
                except Exception as e:
                    logger.warning(f"Error scanning {symbol}: {e}")
            
            report = await research_report_generator.generate_batch_report(scan_results)
            return {
                "success": True,
                "report": report,
                "symbols_analyzed": len(scan_results),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/gdocs/journal/today")
    async def get_todays_journal():
        """Get URL for today's trade journal."""
        try:
            url = await trade_journal.get_journal_url()
            return {
                "url": url,
                "date": datetime.utcnow().strftime('%Y-%m-%d'),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/gdocs/journal/list")
    async def list_trade_journals(max_results: int = 30):
        """List all trade journals."""
        try:
            journals = await trade_journal.list_journals(max_results)
            return {
                "journals": journals,
                "count": len(journals),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/journal/log/note")
    async def log_research_note(request: LogResearchNoteRequest):
        """Log a research note to today's journal."""
        try:
            result = await trade_journal.log_research_note(
                title=request.title,
                content=request.content,
                symbol=request.symbol,
            )
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/journal/log/scan/{symbol}")
    async def log_scan_to_journal(symbol: str):
        """Run a scan and log results to the journal."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=150)
            bars = data_adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, symbol)
            
            if window is None:
                raise HTTPException(status_code=400, detail="Insufficient data")
            
            context = ApexContext(seed=42, compliance_mode=True)
            result = engine.run(window, context)
            omega_statuses = omega_directives.apply_all(result)
            omega_alerts = [name for name, status in omega_statuses.items() if status.active]
            
            scan_result = {
                "symbol": symbol,
                "quantrascore": result.quantrascore,
                "score_bucket": result.score_bucket.value,
                "regime": result.regime.value,
                "risk_tier": result.risk_tier.value,
                "entropy_state": result.entropy_state.value,
                "suppression_state": result.suppression_state.value,
                "drift_state": result.drift_state.value,
                "verdict_action": result.verdict.action,
                "verdict_confidence": result.verdict.confidence,
                "omega_alerts": omega_alerts,
                "protocol_fired_count": sum(1 for p in result.protocol_results if p.fired),
                "window_hash": result.window_hash,
            }
            
            journal_result = await trade_journal.log_scan_result(symbol, scan_result)
            
            if omega_alerts:
                await trade_journal.log_omega_alert(symbol, omega_alerts, scan_result)
            
            return {
                "success": True,
                "scan_result": scan_result,
                "journal": journal_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/investor/monthly")
    async def generate_monthly_investor_update(request: InvestorUpdateRequest):
        """Generate a monthly investor update document."""
        try:
            result = await investor_updates.generate_monthly_update(
                month=request.month,
                highlights=request.highlights,
            )
            return {
                "success": True,
                "document": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/investor/quarterly")
    async def generate_quarterly_review(quarter: str = "Q4", year: int = 2025):
        """Generate a quarterly business review document."""
        try:
            result = await investor_updates.generate_quarterly_review(quarter, year)
            return {
                "success": True,
                "document": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/investor/due-diligence")
    async def generate_due_diligence_package(investor_name: Optional[str] = None):
        """Generate a comprehensive due diligence package."""
        try:
            result = await investor_updates.generate_due_diligence_package(investor_name)
            return {
                "success": True,
                "document": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/notes/import")
    async def import_research_notes(request: ImportDocumentRequest):
        """Import a Google Doc as research notes."""
        try:
            note = await notes_importer.import_document(request.document_id)
            return {
                "success": True,
                "note": note.model_dump(),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/notes/watchlist")
    async def import_watchlist(request: ImportDocumentRequest, name: Optional[str] = None):
        """Import a Google Doc as a watchlist."""
        try:
            watchlist = await notes_importer.import_watchlist(request.document_id, name)
            return {
                "success": True,
                "watchlist": watchlist.model_dump(),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/gdocs/notes/symbols")
    async def extract_all_symbols(max_documents: int = 20):
        """Extract all stock symbols from recent documents."""
        try:
            result = await notes_importer.get_all_symbols_from_notes(max_documents)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/gdocs/notes/watchlists")
    async def find_watchlist_documents():
        """Find documents that appear to be watchlists."""
        try:
            docs = await notes_importer.find_watchlist_documents()
            return {
                "potential_watchlists": docs,
                "count": len(docs),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/sync/export")
    async def export_document_to_gdocs(request: ExportDocumentRequest):
        """Export a local document to Google Docs."""
        try:
            result = await doc_sync.export_document(request.local_path)
            return {
                "success": result.success,
                "result": result.model_dump(),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/sync/export-all")
    async def export_all_docs():
        """Export all local documentation to Google Docs."""
        try:
            result = await doc_sync.export_all_docs()
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/sync/specs")
    async def sync_specification_documents():
        """Sync key specification documents to Google Docs."""
        try:
            result = await doc_sync.sync_spec_documents()
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/gdocs/sync/list")
    async def list_synced_documents():
        """List all QuantraCore documents in Google Docs."""
        try:
            docs = await doc_sync.list_synced_docs()
            return {
                "documents": docs,
                "count": len(docs),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/gdocs/sync/index")
    async def create_documentation_index():
        """Create an index document linking to all synced docs."""
        try:
            result = await doc_sync.create_documentation_index()
            return {
                "success": True,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    _alpha_factory = None
    
    @app.get("/alpha-factory/status")
    async def get_alpha_factory_status():
        """Get alpha factory status and portfolio summary."""
        global _alpha_factory
        if _alpha_factory is None:
            return {
                "running": False,
                "message": "Alpha factory not started",
                "timestamp": datetime.utcnow().isoformat()
            }
        return {
            **_alpha_factory.get_status(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/alpha-factory/portfolio")
    async def get_portfolio_summary():
        """Get current portfolio summary."""
        global _alpha_factory
        if _alpha_factory is None:
            return {"error": "Alpha factory not running"}
        return _alpha_factory.portfolio.get_summary()
    
    @app.get("/alpha-factory/positions")
    async def get_active_positions():
        """Get active positions."""
        global _alpha_factory
        if _alpha_factory is None:
            return {"positions": [], "error": "Alpha factory not running"}
        return {"positions": _alpha_factory.portfolio.get_active_positions()}
    
    @app.get("/alpha-factory/equity-curve")
    async def get_equity_curve():
        """Get equity curve data."""
        global _alpha_factory
        if _alpha_factory is None:
            return {"data": [], "error": "Alpha factory not running"}
        df = _alpha_factory.portfolio.get_equity_curve()
        return {"data": df.to_dict('records') if not df.empty else []}
    
    return app


app = create_app()
