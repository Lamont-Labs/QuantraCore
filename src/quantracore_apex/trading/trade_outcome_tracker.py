"""
Trade Outcome Tracker.

Monitors closed positions and records outcomes for learning.
Feeds data back into model improvement loop.
"""

import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Recorded outcome of a closed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str
    pnl: float
    pnl_pct: float
    hold_duration_hours: float
    exit_reason: str
    strategy: str
    entry_score: float
    entry_conviction: str
    was_profitable: bool
    hit_stop_loss: bool
    hit_take_profit: bool
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        d["exit_time"] = self.exit_time.isoformat()
        return d


class TradeOutcomeTracker:
    """
    Tracks and records trade outcomes for learning.
    
    Monitors closed positions from Alpaca and records:
    - Entry/exit prices and times
    - P&L and hold duration
    - Whether stop-loss or take-profit was hit
    - Original prediction score
    
    This data feeds into the learning loop for model improvement.
    """
    
    OUTCOMES_DIR = Path("logs/trade_outcomes/")
    LEARNING_DATA_DIR = Path("data/learning/")
    
    def __init__(self):
        self._outcomes: List[TradeOutcome] = []
        self._lock = threading.Lock()
        self._last_check: Optional[datetime] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self.OUTCOMES_DIR.mkdir(parents=True, exist_ok=True)
        self.LEARNING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        self._load_existing_outcomes()
        logger.info(f"[TradeOutcomeTracker] Initialized with {len(self._outcomes)} historical outcomes")
    
    def _load_existing_outcomes(self):
        """Load outcomes from previous sessions."""
        try:
            outcome_files = sorted(self.OUTCOMES_DIR.glob("*.json"))
            for f in outcome_files[-100:]:
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                        data["entry_time"] = datetime.fromisoformat(data["entry_time"])
                        data["exit_time"] = datetime.fromisoformat(data["exit_time"])
                        self._outcomes.append(TradeOutcome(**data))
                except Exception as e:
                    logger.debug(f"Could not load outcome {f}: {e}")
        except Exception as e:
            logger.error(f"Error loading outcomes: {e}")
    
    def record_outcome(self, outcome: TradeOutcome):
        """Record a new trade outcome."""
        with self._lock:
            self._outcomes.append(outcome)
            if len(self._outcomes) > 1000:
                self._outcomes = self._outcomes[-1000:]
        
        try:
            filename = f"{outcome.symbol}_{outcome.exit_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.OUTCOMES_DIR / filename
            with open(filepath, "w") as f:
                json.dump(outcome.to_dict(), f, indent=2)
            
            logger.info(
                f"[TradeOutcomeTracker] Recorded {outcome.symbol}: "
                f"{'PROFIT' if outcome.was_profitable else 'LOSS'} "
                f"{outcome.pnl_pct:.1f}% in {outcome.hold_duration_hours:.1f}h"
            )
        except Exception as e:
            logger.error(f"Error saving outcome: {e}")
    
    def check_closed_positions(self) -> List[TradeOutcome]:
        """Check Alpaca for recently closed positions and record outcomes."""
        new_outcomes = []
        
        try:
            from src.quantracore_apex.broker.adapters.alpaca_adapter import AlpacaPaperAdapter
            
            adapter = AlpacaPaperAdapter()
            if not adapter.is_configured:
                logger.warning("[TradeOutcomeTracker] Alpaca not configured")
                return []
            
            all_orders = adapter.get_orders(status="all", limit=200)
            
            existing_symbols = {(o.symbol, o.exit_time.date()) for o in self._outcomes}
            
            for order in all_orders:
                if order.get("status") != "filled":
                    continue
                if order.get("side") != "sell":
                    continue
                
                symbol = order.get("symbol")
                filled_at = order.get("filled_at")
                
                if filled_at:
                    exit_time = datetime.fromisoformat(filled_at.replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    continue
                
                if (symbol, exit_time.date()) in existing_symbols:
                    continue
                
                entry_info = self._find_entry_for_exit(symbol, exit_time, adapter, all_orders)
                if not entry_info:
                    continue
                
                exit_price = float(order.get("filled_avg_price", 0))
                entry_price = entry_info.get("entry_price", exit_price)
                quantity = int(float(order.get("filled_qty", 0)))
                
                pnl = (exit_price - entry_price) * quantity
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0
                
                entry_time = entry_info.get("entry_time", exit_time - timedelta(days=1))
                hold_duration = (exit_time - entry_time).total_seconds() / 3600
                
                hit_stop = pnl_pct <= -7
                hit_tp = pnl_pct >= 40
                
                exit_reason = "take_profit" if hit_tp else "stop_loss" if hit_stop else "manual"
                
                outcome = TradeOutcome(
                    symbol=symbol,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    side="LONG",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    hold_duration_hours=hold_duration,
                    exit_reason=exit_reason,
                    strategy=entry_info.get("strategy", "unified"),
                    entry_score=entry_info.get("entry_score", 0),
                    entry_conviction=entry_info.get("conviction", "unknown"),
                    was_profitable=pnl > 0,
                    hit_stop_loss=hit_stop,
                    hit_take_profit=hit_tp,
                )
                
                self.record_outcome(outcome)
                new_outcomes.append(outcome)
            
            self._last_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"[TradeOutcomeTracker] Error checking closed positions: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return new_outcomes
    
    def _find_entry_for_exit(
        self, 
        symbol: str, 
        exit_time: datetime, 
        adapter: Any,
        all_orders: Optional[List[Dict]] = None
    ) -> Optional[Dict[str, Any]]:
        """Find the entry order that matches this exit."""
        try:
            if all_orders is None:
                orders = adapter.get_orders(status="all", limit=200)
            else:
                orders = all_orders
            
            matching_entries = []
            for order in orders:
                if order.get("symbol") != symbol:
                    continue
                if order.get("side") != "buy":
                    continue
                if order.get("status") != "filled":
                    continue
                
                filled_at = order.get("filled_at")
                if not filled_at:
                    continue
                
                entry_time = datetime.fromisoformat(filled_at.replace("Z", "+00:00")).replace(tzinfo=None)
                
                if entry_time < exit_time:
                    matching_entries.append({
                        "entry_time": entry_time,
                        "entry_price": float(order.get("filled_avg_price", 0)),
                        "strategy": "unified",
                        "entry_score": 0,
                        "conviction": "unknown",
                        "client_order_id": order.get("client_order_id"),
                    })
            
            if matching_entries:
                return max(matching_entries, key=lambda x: x["entry_time"])
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not find entry for {symbol}: {e}")
            return None
    
    def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get trading statistics for the learning loop."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self._lock:
            recent = [o for o in self._outcomes if o.exit_time >= cutoff]
        
        if not recent:
            return {
                "period_days": days,
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl_pct": 0,
                "avg_hold_hours": 0,
            }
        
        wins = sum(1 for o in recent if o.was_profitable)
        total_pnl_pct = sum(o.pnl_pct for o in recent)
        total_hold = sum(o.hold_duration_hours for o in recent)
        
        by_strategy = {}
        for o in recent:
            if o.strategy not in by_strategy:
                by_strategy[o.strategy] = {"wins": 0, "total": 0, "pnl_sum": 0}
            by_strategy[o.strategy]["total"] += 1
            if o.was_profitable:
                by_strategy[o.strategy]["wins"] += 1
            by_strategy[o.strategy]["pnl_sum"] += o.pnl_pct
        
        strategy_stats = {}
        for strat, data in by_strategy.items():
            strategy_stats[strat] = {
                "trades": data["total"],
                "win_rate": data["wins"] / data["total"] * 100 if data["total"] else 0,
                "avg_pnl_pct": data["pnl_sum"] / data["total"] if data["total"] else 0,
            }
        
        stop_losses = sum(1 for o in recent if o.hit_stop_loss)
        take_profits = sum(1 for o in recent if o.hit_take_profit)
        
        return {
            "period_days": days,
            "total_trades": len(recent),
            "winning_trades": wins,
            "losing_trades": len(recent) - wins,
            "win_rate": wins / len(recent) * 100,
            "avg_pnl_pct": total_pnl_pct / len(recent),
            "total_pnl_pct": total_pnl_pct,
            "avg_hold_hours": total_hold / len(recent),
            "stop_losses_hit": stop_losses,
            "take_profits_hit": take_profits,
            "by_strategy": strategy_stats,
            "best_trade": max(recent, key=lambda x: x.pnl_pct).to_dict() if recent else None,
            "worst_trade": min(recent, key=lambda x: x.pnl_pct).to_dict() if recent else None,
        }
    
    def get_learning_data(self) -> List[Dict[str, Any]]:
        """Get trade outcomes formatted for model training."""
        with self._lock:
            return [o.to_dict() for o in self._outcomes]
    
    def export_for_training(self, filepath: Optional[str] = None) -> str:
        """Export outcomes to JSON for model retraining."""
        if filepath is None:
            filepath = str(self.LEARNING_DATA_DIR / f"outcomes_{datetime.utcnow().strftime('%Y%m%d')}.json")
        
        data = self.get_learning_data()
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"[TradeOutcomeTracker] Exported {len(data)} outcomes to {filepath}")
        return filepath
    
    def _monitor_loop(self):
        """Background loop to check for closed positions."""
        logger.info("[TradeOutcomeTracker] Monitor started")
        
        while self._running:
            try:
                self.check_closed_positions()
                threading.Event().wait(300)
            except Exception as e:
                logger.error(f"[TradeOutcomeTracker] Monitor error: {e}")
                threading.Event().wait(60)
        
        logger.info("[TradeOutcomeTracker] Monitor stopped")
    
    def start_monitoring(self):
        """Start background monitoring of closed positions."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("[TradeOutcomeTracker] Background monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None


_tracker_instance: Optional[TradeOutcomeTracker] = None


def get_trade_outcome_tracker() -> TradeOutcomeTracker:
    """Get or create the singleton tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = TradeOutcomeTracker()
    return _tracker_instance
