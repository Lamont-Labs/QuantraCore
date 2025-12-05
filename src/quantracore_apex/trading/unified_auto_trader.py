"""
Unified Autonomous Trading System.

LEGAL NOTICE AND COMPLIANCE:
- This module operates EXCLUSIVELY on Alpaca PAPER trading accounts
- NO REAL MONEY is ever at risk - all trades are simulated
- All outputs are for RESEARCH and EDUCATIONAL purposes only
- This is NOT financial advice and should not be treated as such

Combines EOD and intraday models for unified moonshot detection,
then executes bracket orders (entry + stop-loss + take-profit in one call).
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import json
from pathlib import Path

from ..broker.adapters.alpaca_adapter import AlpacaPaperAdapter
from ..broker.enums import OrderSide, OrderStatus, TimeInForce, OrderType
from ..broker.models import OrderTicket
from ..server.ml_scanner import (
    combined_moonshot_scan,
    scan_for_runners,
    MOONSHOT_UNIVERSE,
)

logger = logging.getLogger(__name__)

QUICK_SCAN_UNIVERSE = [
    "AMC", "GME", "BBBY", "MARA", "RIOT",
    "PLUG", "FCEL", "SOFI", "LCID", "RIVN",
    "CLOV", "WISH", "PLTR", "NIO", "XPEV",
    "LAZR", "VLDR", "GOEV", "WKHS", "FFIE",
]


@dataclass
class UnifiedCandidate:
    """A stock candidate identified by the merged model system."""
    symbol: str
    combined_score: float
    eod_confidence: float
    intraday_confidence: float
    high_conviction: bool
    signal_sources: List[str]
    current_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    shares: int = 0
    position_value: float = 0.0
    risk_reward: float = 0.0


@dataclass
class BracketOrderResult:
    """Result of a bracket order execution."""
    symbol: str
    order_id: str
    status: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    shares: int
    position_value: float
    error: Optional[str] = None
    timestamp: str = ""
    legs: List[str] = field(default_factory=list)


@dataclass
class PositionAnalysis:
    """Analysis of an existing position for adjustment decisions."""
    symbol: str
    qty: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    market_value: float
    action: str  # "hold", "close", "partial_close", "trail_stop"
    reason: str
    model_score: float = 0.0
    momentum_score: float = 0.0
    days_held: int = 0


class UnifiedAutoTrader:
    """
    Unified Autonomous Trading System.
    
    This system combines EOD and intraday models for maximum signal quality,
    then executes bracket orders that include entry, stop-loss, and take-profit
    all in a single atomic order.
    
    PAPER TRADING ONLY - No real money at risk.
    """
    
    STOP_LOSS_PCT = 0.08
    TAKE_PROFIT_PCT = 0.50
    MAX_POSITION_PCT = 0.10
    MIN_COMBINED_SCORE = 0.5
    
    def __init__(
        self,
        max_new_positions: int = 3,
        max_position_pct: float = 0.10,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.50,
        min_combined_score: float = 0.5,
        require_high_conviction: bool = False,
        log_dir: str = "investor_logs/unified_trades/",
    ):
        """
        Initialize unified auto trader.
        
        Args:
            max_new_positions: Max new positions to open per scan
            max_position_pct: Max % of equity per position
            stop_loss_pct: Stop-loss percentage (e.g., 0.08 = 8%)
            take_profit_pct: Take-profit percentage (e.g., 0.50 = 50%)
            min_combined_score: Minimum combined model score
            require_high_conviction: Only trade if BOTH models agree
            log_dir: Directory for trade logs
        """
        self.max_new_positions = max_new_positions
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_combined_score = min_combined_score
        self.require_high_conviction = require_high_conviction
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.alpaca = AlpacaPaperAdapter()
    
    def get_account_status(self) -> Dict[str, Any]:
        """Get current Alpaca account status."""
        if not self.alpaca.is_configured:
            return {"error": "Alpaca not configured", "configured": False}
        
        try:
            account = self.alpaca.get_account_info()
            positions = self.alpaca.get_positions()
            
            return {
                "configured": True,
                "equity": float(account.get("equity", 0)),
                "cash": float(account.get("cash", 0)),
                "buying_power": float(account.get("buying_power", 0)),
                "positions_count": len(positions),
                "current_symbols": [p.symbol for p in positions],
                "positions": [
                    {
                        "symbol": p.symbol,
                        "qty": p.qty,
                        "market_value": p.market_value,
                        "unrealized_pl": p.unrealized_pl,
                        "unrealized_plpc": p.unrealized_plpc,
                    }
                    for p in positions
                ],
            }
        except Exception as e:
            logger.error(f"Failed to get account status: {e}")
            return {"error": str(e), "configured": True}
    
    def scan_universe_merged(
        self,
        symbols: Optional[List[str]] = None,
        include_eod: bool = True,
        include_intraday: bool = True,
    ) -> List[UnifiedCandidate]:
        """
        Scan universe using merged EOD + intraday models.
        
        Returns candidates sorted by combined score.
        """
        if symbols is None:
            symbols = MOONSHOT_UNIVERSE
        
        try:
            result = combined_moonshot_scan(
                symbols=symbols,
                include_eod=include_eod,
                include_intraday=include_intraday,
            )
        except Exception as e:
            logger.error(f"Combined scan failed: {e}")
            return []
        
        candidates = []
        for c in result.get('combined_candidates', []):
            if c['combined_score'] < self.min_combined_score:
                continue
            
            if self.require_high_conviction and not c['high_conviction']:
                continue
            
            candidates.append(UnifiedCandidate(
                symbol=c['symbol'],
                combined_score=c['combined_score'],
                eod_confidence=c['eod_confidence'],
                intraday_confidence=c['intraday_confidence'],
                high_conviction=c['high_conviction'],
                signal_sources=c['signal_sources'],
            ))
        
        return candidates
    
    def enrich_candidates(
        self,
        candidates: List[UnifiedCandidate],
        equity: float,
        available_cash: float,
    ) -> List[UnifiedCandidate]:
        """Add pricing and position sizing to candidates with capital checks."""
        max_position_value = min(equity * self.max_position_pct, available_cash * 0.95)
        
        if max_position_value < 100:
            logger.warning(f"Insufficient capital for positions: ${available_cash:.2f} cash")
            return []
        
        enriched = []
        remaining_cash = available_cash
        
        for c in candidates:
            if remaining_cash < 100:
                logger.info(f"Stopping enrichment: insufficient remaining cash ${remaining_cash:.2f}")
                break
            
            try:
                price = self.alpaca.get_last_price(c.symbol)
                if price <= 0:
                    continue
                
                position_budget = min(max_position_value, remaining_cash * 0.95)
                shares = int(position_budget / price)
                if shares < 1:
                    continue
                
                position_cost = shares * price
                if position_cost > remaining_cash:
                    shares = int(remaining_cash * 0.95 / price)
                    if shares < 1:
                        continue
                    position_cost = shares * price
                
                stop_loss = round(price * (1 - self.stop_loss_pct), 2)
                take_profit = round(price * (1 + self.take_profit_pct), 2)
                
                c.current_price = price
                c.stop_loss_price = stop_loss
                c.take_profit_price = take_profit
                c.shares = shares
                c.position_value = position_cost
                c.risk_reward = self.take_profit_pct / self.stop_loss_pct
                
                enriched.append(c)
                remaining_cash -= position_cost
                
            except Exception as e:
                logger.warning(f"Failed to enrich {c.symbol}: {e}")
                continue
        
        return enriched
    
    def execute_bracket_order(
        self,
        candidate: UnifiedCandidate,
    ) -> BracketOrderResult:
        """Execute a bracket order for a candidate."""
        if not self.alpaca.is_configured:
            return BracketOrderResult(
                symbol=candidate.symbol,
                order_id="",
                status="rejected",
                entry_price=candidate.current_price,
                stop_loss_price=candidate.stop_loss_price,
                take_profit_price=candidate.take_profit_price,
                shares=candidate.shares,
                position_value=candidate.position_value,
                error="Alpaca not configured",
                timestamp=datetime.utcnow().isoformat(),
            )
        
        try:
            result = self.alpaca.place_bracket_order(
                symbol=candidate.symbol,
                qty=candidate.shares,
                side=OrderSide.BUY,
                stop_loss_price=candidate.stop_loss_price,
                take_profit_price=candidate.take_profit_price,
                time_in_force=TimeInForce.GTC,
            )
            
            legs = result.raw_broker_payload.get("legs", []) if result.raw_broker_payload else []
            leg_ids = [leg.get("id", "") for leg in legs]
            
            order_result = BracketOrderResult(
                symbol=candidate.symbol,
                order_id=result.order_id,
                status=result.status.value if result.status else "unknown",
                entry_price=result.avg_fill_price or candidate.current_price,
                stop_loss_price=candidate.stop_loss_price,
                take_profit_price=candidate.take_profit_price,
                shares=candidate.shares,
                position_value=candidate.position_value,
                error=result.error_message,
                timestamp=datetime.utcnow().isoformat(),
                legs=leg_ids,
            )
            
            self._log_trade(order_result, candidate)
            return order_result
            
        except Exception as e:
            logger.error(f"Bracket order failed for {candidate.symbol}: {e}")
            return BracketOrderResult(
                symbol=candidate.symbol,
                order_id="",
                status="error",
                entry_price=candidate.current_price,
                stop_loss_price=candidate.stop_loss_price,
                take_profit_price=candidate.take_profit_price,
                shares=candidate.shares,
                position_value=candidate.position_value,
                error=str(e),
                timestamp=datetime.utcnow().isoformat(),
            )
    
    def scan_analyze_trade(
        self,
        symbols: Optional[List[str]] = None,
        max_trades: Optional[int] = None,
        include_eod: bool = True,
        include_intraday: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        UNIFIED ENDPOINT: Scan all stocks, analyze with merged models, 
        and execute bracket orders in one call.
        
        Args:
            symbols: List of symbols to scan (default: MOONSHOT_UNIVERSE)
            max_trades: Max trades to execute (default: self.max_new_positions)
            include_eod: Include EOD model in scan
            include_intraday: Include intraday model in scan
            dry_run: If True, scan and analyze but don't execute trades
        
        Returns:
            Complete execution report with account status, candidates, and results
        """
        if max_trades is None:
            max_trades = self.max_new_positions
        
        timestamp_start = datetime.utcnow()
        
        account = self.get_account_status()
        if account.get("error"):
            return {
                "success": False,
                "error": account.get("error"),
                "timestamp": timestamp_start.isoformat(),
            }
        
        equity = account.get("equity", 100000)
        current_symbols = set(account.get("current_symbols", []))
        
        logger.info(f"[UnifiedAutoTrader] Starting unified scan → analyze → trade")
        logger.info(f"[UnifiedAutoTrader] Account: ${equity:,.2f} equity, {len(current_symbols)} positions")
        
        candidates = self.scan_universe_merged(
            symbols=symbols,
            include_eod=include_eod,
            include_intraday=include_intraday,
        )
        logger.info(f"[UnifiedAutoTrader] Found {len(candidates)} candidates with score >= {self.min_combined_score}")
        
        candidates = [c for c in candidates if c.symbol not in current_symbols]
        logger.info(f"[UnifiedAutoTrader] {len(candidates)} candidates after filtering existing positions")
        
        available_cash = account.get("cash", 0)
        candidates = self.enrich_candidates(candidates, equity, available_cash)
        logger.info(f"[UnifiedAutoTrader] {len(candidates)} candidates enriched with pricing (cash: ${available_cash:,.2f})")
        
        candidates = sorted(candidates, key=lambda x: x.combined_score, reverse=True)
        candidates = candidates[:max_trades]
        
        executed_orders = []
        if not dry_run and candidates:
            for candidate in candidates:
                logger.info(
                    f"[UnifiedAutoTrader] Executing bracket order: "
                    f"BUY {candidate.shares} {candidate.symbol} @ ~${candidate.current_price:.2f} | "
                    f"Stop: ${candidate.stop_loss_price:.2f} | Target: ${candidate.take_profit_price:.2f}"
                )
                result = self.execute_bracket_order(candidate)
                executed_orders.append(result)
                logger.info(f"[UnifiedAutoTrader] Order result: {result.status} - {result.order_id}")
        
        timestamp_end = datetime.utcnow()
        account_after = self.get_account_status() if not dry_run and executed_orders else account
        
        return {
            "success": True,
            "dry_run": dry_run,
            "timestamp_start": timestamp_start.isoformat(),
            "timestamp_end": timestamp_end.isoformat(),
            "duration_seconds": (timestamp_end - timestamp_start).total_seconds(),
            "models_used": {
                "eod": include_eod,
                "intraday": include_intraday,
                "merged": include_eod and include_intraday,
            },
            "parameters": {
                "min_combined_score": self.min_combined_score,
                "stop_loss_pct": f"{self.stop_loss_pct * 100:.1f}%",
                "take_profit_pct": f"{self.take_profit_pct * 100:.1f}%",
                "max_position_pct": f"{self.max_position_pct * 100:.1f}%",
                "require_high_conviction": self.require_high_conviction,
            },
            "account_before": {
                "equity": account.get("equity"),
                "cash": account.get("cash"),
                "positions_count": account.get("positions_count"),
            },
            "account_after": {
                "equity": account_after.get("equity"),
                "cash": account_after.get("cash"),
                "positions_count": account_after.get("positions_count"),
            } if not dry_run else None,
            "scan_results": {
                "symbols_scanned": len(symbols or MOONSHOT_UNIVERSE),
                "candidates_found": len(candidates),
                "high_conviction_count": sum(1 for c in candidates if c.high_conviction),
            },
            "candidates": [
                {
                    "symbol": c.symbol,
                    "combined_score": round(c.combined_score, 3),
                    "eod_confidence": round(c.eod_confidence, 3),
                    "intraday_confidence": round(c.intraday_confidence, 3),
                    "high_conviction": c.high_conviction,
                    "signal_sources": c.signal_sources,
                    "current_price": c.current_price,
                    "stop_loss": c.stop_loss_price,
                    "take_profit": c.take_profit_price,
                    "shares": c.shares,
                    "position_value": round(c.position_value, 2),
                    "risk_reward": round(c.risk_reward, 1),
                }
                for c in candidates
            ],
            "executed_orders": [
                {
                    "symbol": o.symbol,
                    "order_id": o.order_id,
                    "status": o.status,
                    "entry_price": o.entry_price,
                    "stop_loss": o.stop_loss_price,
                    "take_profit": o.take_profit_price,
                    "shares": o.shares,
                    "position_value": round(o.position_value, 2),
                    "legs": o.legs,
                    "error": o.error,
                }
                for o in executed_orders
            ] if not dry_run else [],
            "summary": {
                "orders_attempted": len(candidates) if not dry_run else 0,
                "orders_successful": sum(1 for o in executed_orders if o.status in ["new", "accepted", "filled"]),
                "orders_failed": sum(1 for o in executed_orders if o.status in ["rejected", "error"]),
                "total_value_traded": sum(o.position_value for o in executed_orders if o.status not in ["rejected", "error"]),
            },
            "compliance_note": "PAPER TRADING ONLY - All orders executed on Alpaca paper account",
        }
    
    def analyze_positions(self) -> List[PositionAnalysis]:
        """
        Analyze all current positions for potential adjustments.
        
        Checks:
        - Deteriorating positions (losing momentum, model score dropped)
        - Big winners ready for profit taking
        - Positions held too long without significant gains
        
        Returns list of position analyses with recommended actions.
        """
        if not self.alpaca.is_configured:
            return []
        
        try:
            positions = self.alpaca.get_positions()
        except Exception as e:
            logger.error(f"Failed to get positions for analysis: {e}")
            return []
        
        if not positions:
            return []
        
        analyses = []
        
        symbols = [p.symbol for p in positions]
        model_scores = {}
        try:
            result = combined_moonshot_scan(symbols=symbols, include_eod=True, include_intraday=True)
            for c in result.get('combined_candidates', []):
                model_scores[c['symbol']] = c['combined_score']
        except Exception as e:
            logger.warning(f"Model re-scan failed: {e}")
        
        for pos in positions:
            pnl_pct = pos.unrealized_plpc
            model_score = model_scores.get(pos.symbol, 0.5)
            
            action = "hold"
            reason = "Position looks healthy"
            
            if pnl_pct <= -12:
                action = "close"
                reason = f"Deep loss ({pnl_pct:.1f}%) - exit to preserve capital"
            elif pnl_pct <= -8 and model_score < 0.4:
                action = "close"
                reason = f"Losing position ({pnl_pct:.1f}%) with weak model score ({model_score:.2f})"
            elif pnl_pct <= -5 and model_score < 0.3:
                action = "close"
                reason = f"Deteriorating - loss {pnl_pct:.1f}%, model score dropped to {model_score:.2f}"
            elif pnl_pct >= 40:
                action = "partial_close"
                reason = f"Take partial profits at +{pnl_pct:.1f}% gain"
            elif pnl_pct >= 25 and model_score < 0.4:
                action = "close"
                reason = f"Lock in +{pnl_pct:.1f}% gain - model score weak ({model_score:.2f})"
            elif pnl_pct >= 15 and model_score < 0.3:
                action = "close"
                reason = f"Protect +{pnl_pct:.1f}% gain - momentum fading"
            
            analyses.append(PositionAnalysis(
                symbol=pos.symbol,
                qty=int(pos.qty),
                avg_entry_price=pos.avg_entry_price,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pl,
                unrealized_pnl_pct=pnl_pct,
                market_value=pos.market_value,
                action=action,
                reason=reason,
                model_score=model_score,
            ))
        
        return analyses
    
    def manage_positions(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute position management based on analysis.
        
        Closes deteriorating positions and takes profits on big winners.
        
        Returns summary of actions taken.
        """
        analyses = self.analyze_positions()
        
        if not analyses:
            return {
                "positions_analyzed": 0,
                "adjustments_made": 0,
                "actions": [],
            }
        
        actions_taken = []
        
        for analysis in analyses:
            if analysis.action == "hold":
                continue
            
            if analysis.action == "close":
                if dry_run:
                    actions_taken.append({
                        "symbol": analysis.symbol,
                        "action": "close",
                        "reason": analysis.reason,
                        "pnl_pct": analysis.unrealized_pnl_pct,
                        "dry_run": True,
                        "status": "would_execute",
                    })
                else:
                    try:
                        order = OrderTicket(
                            symbol=analysis.symbol,
                            qty=analysis.qty,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            time_in_force=TimeInForce.DAY,
                        )
                        result = self.alpaca.place_order(order)
                        actions_taken.append({
                            "symbol": analysis.symbol,
                            "action": "close",
                            "reason": analysis.reason,
                            "pnl_pct": analysis.unrealized_pnl_pct,
                            "order_id": result.order_id,
                            "status": result.status.value,
                        })
                        logger.info(f"[PositionMgmt] Closed {analysis.symbol}: {analysis.reason}")
                    except Exception as e:
                        actions_taken.append({
                            "symbol": analysis.symbol,
                            "action": "close",
                            "reason": analysis.reason,
                            "error": str(e),
                            "status": "failed",
                        })
            
            elif analysis.action == "partial_close":
                close_qty = max(1, int(analysis.qty * 0.5))
                
                if dry_run:
                    actions_taken.append({
                        "symbol": analysis.symbol,
                        "action": "partial_close",
                        "qty_to_close": close_qty,
                        "reason": analysis.reason,
                        "pnl_pct": analysis.unrealized_pnl_pct,
                        "dry_run": True,
                        "status": "would_execute",
                    })
                else:
                    try:
                        order = OrderTicket(
                            symbol=analysis.symbol,
                            qty=close_qty,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            time_in_force=TimeInForce.DAY,
                        )
                        result = self.alpaca.place_order(order)
                        actions_taken.append({
                            "symbol": analysis.symbol,
                            "action": "partial_close",
                            "qty_closed": close_qty,
                            "reason": analysis.reason,
                            "pnl_pct": analysis.unrealized_pnl_pct,
                            "order_id": result.order_id,
                            "status": result.status.value,
                        })
                        logger.info(f"[PositionMgmt] Partial close {analysis.symbol} ({close_qty} shares): {analysis.reason}")
                    except Exception as e:
                        actions_taken.append({
                            "symbol": analysis.symbol,
                            "action": "partial_close",
                            "error": str(e),
                            "status": "failed",
                        })
        
        return {
            "positions_analyzed": len(analyses),
            "adjustments_made": len(actions_taken),
            "holds": sum(1 for a in analyses if a.action == "hold"),
            "closes": sum(1 for a in actions_taken if a.get("action") == "close"),
            "partial_closes": sum(1 for a in actions_taken if a.get("action") == "partial_close"),
            "actions": actions_taken,
            "all_positions": [
                {
                    "symbol": a.symbol,
                    "pnl_pct": round(a.unrealized_pnl_pct, 2),
                    "model_score": round(a.model_score, 3),
                    "action": a.action,
                    "reason": a.reason,
                }
                for a in analyses
            ],
        }

    def _log_trade(self, result: BracketOrderResult, candidate: UnifiedCandidate):
        """Log trade for audit."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"unified_bracket_{result.symbol}_{timestamp}.json"
        
        log_data = {
            "timestamp": result.timestamp,
            "order_id": result.order_id,
            "status": result.status,
            "symbol": result.symbol,
            "shares": result.shares,
            "entry_price": result.entry_price,
            "stop_loss_price": result.stop_loss_price,
            "take_profit_price": result.take_profit_price,
            "position_value": result.position_value,
            "legs": result.legs,
            "error": result.error,
            "model_analysis": {
                "combined_score": candidate.combined_score,
                "eod_confidence": candidate.eod_confidence,
                "intraday_confidence": candidate.intraday_confidence,
                "high_conviction": candidate.high_conviction,
                "signal_sources": candidate.signal_sources,
            },
        }
        
        try:
            with open(self.log_dir / filename, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")


_unified_auto_trader: Optional[UnifiedAutoTrader] = None


def get_unified_auto_trader(**kwargs) -> UnifiedAutoTrader:
    """Get singleton UnifiedAutoTrader instance."""
    global _unified_auto_trader
    if _unified_auto_trader is None:
        _unified_auto_trader = UnifiedAutoTrader(**kwargs)
    return _unified_auto_trader
