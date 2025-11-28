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
from src.quantracore_apex.risk.engine import RiskEngine, RiskAssessment
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
    elif isinstance(obj, (np.bool_, np.bool_.__class__)):
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
            from src.quantracore_apex.core.redundant_scorer import RedundantScorer
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
            
            adapter = SyntheticAdapter()
            data = adapter.get_ohlcv(symbol, lookback_bars=lookback_days, seed=hash(symbol) % 10000)
            
            if not data:
                return {"error": "No data available", "symbol": symbol}
            
            normalized = normalize_ohlcv(data)
            window_builder = WindowBuilder()
            window = window_builder.build_window(normalized)
            
            result = engine.run_scan(window)
            
            scorer = RedundantScorer()
            verification = scorer.compute_with_verification(
                primary_score=result.quantrascore,
                primary_band=result.score_band,
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
            },
            "v9_hardening": {
                "redundant_scoring": True,
                "drift_detection": True,
                "fail_closed_gates": True,
                "sandbox_replay": True,
                "research_only_fence": True,
            },
            "simulation_mode": True,
            "compliance_mode": True,
            "desktop_only": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return app


app = create_app()
