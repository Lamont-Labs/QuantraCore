"""
Self-Learning Feedback Loop for QuantraCore Apex.

Captures trade outcomes and feeds them back into ApexLab for continuous learning.
The system learns from its own trades - what worked, what didn't.

Flow:
1. Alpha Factory generates signals → Portfolio executes trades
2. FeedbackTracker captures entry context (features, protocols, scores)
3. When trade closes, FeedbackTracker records outcome (P&L, holding time, etc.)
4. Completed trades are formatted as new ApexLab training samples
5. When batch threshold reached → ApexCore retraining triggered

This creates a true self-improving loop where the model gets smarter
from real trading experience (paper or live).
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class TradeContext:
    """Context captured at trade entry."""
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    side: str
    
    quantra_score: float
    protocol_flags: List[str]
    omega_flags: List[str]
    
    feature_snapshot: Dict[str, float]
    apexcore_prediction: Optional[Dict[str, Any]] = None
    
    signal_source: str = "alpha_factory"


@dataclass
class TradeOutcome:
    """Outcome captured at trade exit."""
    exit_time: datetime
    exit_price: float
    pnl_dollars: float
    pnl_percent: float
    holding_minutes: int
    exit_reason: str
    
    max_favorable: float = 0.0
    max_adverse: float = 0.0


@dataclass
class CompletedTrade:
    """Complete trade record for learning."""
    trade_id: str
    context: TradeContext
    outcome: TradeOutcome
    
    label: str = ""
    confidence: float = 0.0
    
    def to_training_sample(self) -> Dict[str, Any]:
        """Convert to ApexLab training format."""
        label = self._compute_label()
        
        return {
            "symbol": self.context.symbol,
            "timestamp": self.context.entry_time.isoformat(),
            "features": self.context.feature_snapshot,
            "label": label,
            "metadata": {
                "source": "feedback_loop",
                "trade_id": self.trade_id,
                "quantra_score": self.context.quantra_score,
                "protocols": self.context.protocol_flags,
                "omega_flags": self.context.omega_flags,
                "pnl_percent": self.outcome.pnl_percent,
                "holding_minutes": self.outcome.holding_minutes,
                "exit_reason": self.outcome.exit_reason,
            }
        }
    
    def _compute_label(self) -> str:
        """Compute training label from outcome."""
        pnl = self.outcome.pnl_percent
        
        if pnl >= 5.0:
            return "STRONG_WIN"
        elif pnl >= 2.0:
            return "WIN"
        elif pnl >= 0.5:
            return "MARGINAL_WIN"
        elif pnl >= -0.5:
            return "SCRATCH"
        elif pnl >= -2.0:
            return "LOSS"
        else:
            return "STRONG_LOSS"


class FeedbackTracker:
    """
    Tracks open trades and records outcomes for learning.
    
    Usage:
        tracker = FeedbackTracker()
        
        # On entry
        tracker.record_entry(symbol, price, qty, side, context_dict)
        
        # On exit
        tracker.record_exit(symbol, exit_price, exit_reason)
        
        # Periodic check for retraining
        if tracker.should_retrain():
            samples = tracker.get_training_batch()
            # Feed to ApexLab
    """
    
    def __init__(
        self,
        data_dir: str = "data/feedback",
        batch_threshold: int = 50,
        min_retrain_hours: int = 24,
    ):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        self._batch_threshold = batch_threshold
        self._min_retrain_hours = min_retrain_hours
        
        self._open_trades: Dict[str, TradeContext] = {}
        self._completed_trades: List[CompletedTrade] = []
        self._last_retrain: Optional[datetime] = None
        self._trade_counter = 0
        self._lock = threading.Lock()
        
        self._load_state()
    
    def record_entry(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        side: str,
        quantra_score: float,
        protocol_flags: List[str],
        omega_flags: List[str],
        feature_snapshot: Dict[str, float],
        apexcore_prediction: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record trade entry with full context.
        
        Returns trade_id for later matching with exit.
        """
        with self._lock:
            self._trade_counter += 1
            trade_id = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self._trade_counter}"
            
            context = TradeContext(
                symbol=symbol,
                entry_time=datetime.utcnow(),
                entry_price=entry_price,
                quantity=quantity,
                side=side,
                quantra_score=quantra_score,
                protocol_flags=protocol_flags,
                omega_flags=omega_flags,
                feature_snapshot=feature_snapshot,
                apexcore_prediction=apexcore_prediction,
            )
            
            self._open_trades[symbol] = context
            
            logger.info(
                f"[FeedbackLoop] Entry recorded: {side} {quantity} {symbol} @ {entry_price:.2f} "
                f"(score={quantra_score:.1f}, protocols={len(protocol_flags)})"
            )
            
            return trade_id
    
    def record_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = "signal",
        max_favorable: float = 0.0,
        max_adverse: float = 0.0,
    ) -> Optional[CompletedTrade]:
        """
        Record trade exit and compute outcome.
        
        Returns CompletedTrade if entry was found, None otherwise.
        """
        with self._lock:
            if symbol not in self._open_trades:
                logger.warning(f"[FeedbackLoop] No open trade for {symbol}")
                return None
            
            context = self._open_trades.pop(symbol)
            exit_time = datetime.utcnow()
            
            if context.side == "LONG":
                pnl_pct = ((exit_price - context.entry_price) / context.entry_price) * 100
            else:
                pnl_pct = ((context.entry_price - exit_price) / context.entry_price) * 100
            
            pnl_dollars = pnl_pct / 100 * (context.entry_price * context.quantity)
            holding = exit_time - context.entry_time
            holding_minutes = int(holding.total_seconds() / 60)
            
            outcome = TradeOutcome(
                exit_time=exit_time,
                exit_price=exit_price,
                pnl_dollars=pnl_dollars,
                pnl_percent=pnl_pct,
                holding_minutes=holding_minutes,
                exit_reason=exit_reason,
                max_favorable=max_favorable,
                max_adverse=max_adverse,
            )
            
            trade_id = f"{symbol}_{context.entry_time.strftime('%Y%m%d_%H%M%S')}"
            completed = CompletedTrade(
                trade_id=trade_id,
                context=context,
                outcome=outcome,
            )
            
            self._completed_trades.append(completed)
            self._save_trade(completed)
            
            emoji = "✅" if pnl_pct >= 0 else "❌"
            logger.info(
                f"[FeedbackLoop] {emoji} Exit recorded: {symbol} @ {exit_price:.2f} "
                f"P&L: {pnl_pct:+.2f}% (${pnl_dollars:+.2f}) after {holding_minutes}min"
            )
            
            return completed
    
    def should_retrain(self) -> bool:
        """Check if conditions met for retraining."""
        with self._lock:
            if len(self._completed_trades) < self._batch_threshold:
                return False
            
            if self._last_retrain:
                hours_since = (datetime.utcnow() - self._last_retrain).total_seconds() / 3600
                if hours_since < self._min_retrain_hours:
                    return False
            
            return True
    
    def get_training_batch(self) -> List[Dict[str, Any]]:
        """
        Get completed trades as training samples and clear buffer.
        
        Call this when should_retrain() returns True.
        """
        with self._lock:
            samples = [t.to_training_sample() for t in self._completed_trades]
            
            self._archive_batch()
            
            self._completed_trades = []
            self._last_retrain = datetime.utcnow()
            
            logger.info(f"[FeedbackLoop] Generated {len(samples)} training samples")
            
            return samples
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback loop statistics."""
        with self._lock:
            if not self._completed_trades:
                avg_pnl = 0.0
                win_rate = 0.0
            else:
                pnls = [t.outcome.pnl_percent for t in self._completed_trades]
                avg_pnl = sum(pnls) / len(pnls)
                win_rate = sum(1 for p in pnls if p >= 0) / len(pnls) * 100
            
            return {
                "open_trades": len(self._open_trades),
                "completed_trades": len(self._completed_trades),
                "batch_threshold": self._batch_threshold,
                "avg_pnl_percent": avg_pnl,
                "win_rate": win_rate,
                "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
                "ready_for_retrain": self.should_retrain(),
            }
    
    def _save_trade(self, trade: CompletedTrade):
        """Save completed trade to disk."""
        trades_file = self._data_dir / "completed_trades.jsonl"
        
        record = {
            "trade_id": trade.trade_id,
            "symbol": trade.context.symbol,
            "entry_time": trade.context.entry_time.isoformat(),
            "exit_time": trade.outcome.exit_time.isoformat(),
            "entry_price": trade.context.entry_price,
            "exit_price": trade.outcome.exit_price,
            "quantity": trade.context.quantity,
            "side": trade.context.side,
            "pnl_percent": trade.outcome.pnl_percent,
            "pnl_dollars": trade.outcome.pnl_dollars,
            "holding_minutes": trade.outcome.holding_minutes,
            "quantra_score": trade.context.quantra_score,
            "protocol_flags": trade.context.protocol_flags,
            "omega_flags": trade.context.omega_flags,
            "exit_reason": trade.outcome.exit_reason,
            "label": trade._compute_label(),
        }
        
        with open(trades_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def _archive_batch(self):
        """Archive current batch before clearing."""
        if not self._completed_trades:
            return
        
        archive_dir = self._data_dir / "archives"
        archive_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_file = archive_dir / f"batch_{timestamp}.json"
        
        batch = [t.to_training_sample() for t in self._completed_trades]
        with open(archive_file, "w") as f:
            json.dump(batch, f, indent=2, default=str)
        
        logger.info(f"[FeedbackLoop] Archived {len(batch)} trades to {archive_file}")
    
    def _load_state(self):
        """Load persisted state on startup."""
        state_file = self._data_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self._trade_counter = state.get("trade_counter", 0)
                if state.get("last_retrain"):
                    self._last_retrain = datetime.fromisoformat(state["last_retrain"])
                logger.info("[FeedbackLoop] Loaded persisted state")
            except Exception as e:
                logger.warning(f"[FeedbackLoop] Could not load state: {e}")
    
    def _save_state(self):
        """Persist state to disk."""
        state_file = self._data_dir / "state.json"
        state = {
            "trade_counter": self._trade_counter,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)


_feedback_tracker: Optional[FeedbackTracker] = None


def get_feedback_tracker() -> FeedbackTracker:
    """Get global feedback tracker instance."""
    global _feedback_tracker
    if _feedback_tracker is None:
        _feedback_tracker = FeedbackTracker()
    return _feedback_tracker


def trigger_retrain_if_ready():
    """
    Check if retraining conditions are met and trigger if so.
    
    This integrates with ApexLab to add new samples and kick off training.
    """
    tracker = get_feedback_tracker()
    
    if not tracker.should_retrain():
        return False
    
    samples = tracker.get_training_batch()
    
    if not samples:
        return False
    
    try:
        samples_file = Path("data/apexlab/feedback_samples.json")
        samples_file.parent.mkdir(parents=True, exist_ok=True)
        
        existing = []
        if samples_file.exists():
            with open(samples_file) as f:
                existing = json.load(f)
        
        existing.extend(samples)
        
        with open(samples_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        
        logger.info(
            f"[FeedbackLoop] Added {len(samples)} samples to ApexLab training queue. "
            f"Total: {len(existing)} samples ready for next training run."
        )
        
        return True
        
    except Exception as e:
        logger.error(f"[FeedbackLoop] Failed to queue samples: {e}")
        return False


__all__ = [
    "TradeContext",
    "TradeOutcome", 
    "CompletedTrade",
    "FeedbackTracker",
    "get_feedback_tracker",
    "trigger_retrain_if_ready",
]
