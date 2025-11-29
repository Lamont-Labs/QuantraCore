"""
Trade Outcome Tracker for Self-Learning Integration.

Records trade outcomes and feeds them back to ApexLab for:
- Model improvement through real trading feedback
- Quality scoring calibration
- Performance analysis and attribution
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .models import TradeOutcome, ExitReason

logger = logging.getLogger(__name__)


class TradeOutcomeTracker:
    """
    Tracks trade outcomes and integrates with ApexLab feedback loop.
    
    Responsibilities:
    1. Record all trade entries and exits
    2. Calculate realized P&L and statistics
    3. Generate training samples for ApexLab
    4. Track performance metrics by signal quality tier
    5. Identify which signal characteristics lead to profits/losses
    """
    
    def __init__(
        self,
        feedback_path: str = "data/apexlab/autonomous_feedback.json",
        trade_log_path: str = "data/logs/trade_log.json",
        auto_save: bool = True,
        max_samples: int = 10000,
    ):
        self.feedback_path = Path(feedback_path)
        self.trade_log_path = Path(trade_log_path)
        self.auto_save = auto_save
        self.max_samples = max_samples
        
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._outcomes: List[TradeOutcome] = []
        self._pending_samples: List[Dict[str, Any]] = []
        
        self._stats = {
            "total_trades": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "largest_winner": 0.0,
            "largest_loser": 0.0,
            "avg_bars_held": 0.0,
            "by_exit_reason": {},
            "by_quality_tier": {},
        }
        
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Load existing feedback samples if they exist."""
        if self.feedback_path.exists():
            try:
                with open(self.feedback_path, "r") as f:
                    data = json.load(f)
                    self._pending_samples = data.get("samples", [])
                    self._stats = data.get("stats", self._stats)
                    logger.info(
                        f"[TradeOutcomeTracker] Loaded {len(self._pending_samples)} existing samples"
                    )
            except Exception as e:
                logger.warning(f"[TradeOutcomeTracker] Could not load existing data: {e}")
    
    def record_outcome(self, outcome: TradeOutcome) -> None:
        """
        Record a trade outcome and generate feedback sample.
        
        Called when a position is closed by PositionMonitor.
        """
        self._outcomes.append(outcome)
        self._update_stats(outcome)
        
        sample = outcome.to_training_sample()
        sample["recorded_at"] = datetime.utcnow().isoformat()
        self._pending_samples.append(sample)
        
        if len(self._pending_samples) > self.max_samples:
            self._pending_samples = self._pending_samples[-self.max_samples:]
        
        logger.info(
            f"[TradeOutcomeTracker] Recorded outcome: {outcome.symbol} | "
            f"PnL={outcome.realized_pnl_pct*100:.2f}% | Exit={outcome.exit_reason.value}"
        )
        
        if self.auto_save:
            self.save()
    
    def _update_stats(self, outcome: TradeOutcome) -> None:
        """Update running statistics with new outcome."""
        self._stats["total_trades"] += 1
        self._stats["total_pnl"] += outcome.realized_pnl
        
        if outcome.was_profitable:
            self._stats["profitable_trades"] += 1
            self._stats["largest_winner"] = max(
                self._stats["largest_winner"], outcome.realized_pnl
            )
        else:
            self._stats["losing_trades"] += 1
            self._stats["largest_loser"] = min(
                self._stats["largest_loser"], outcome.realized_pnl
            )
        
        exit_reason = outcome.exit_reason.value
        if exit_reason not in self._stats["by_exit_reason"]:
            self._stats["by_exit_reason"][exit_reason] = {
                "count": 0, "total_pnl": 0.0, "profitable": 0
            }
        self._stats["by_exit_reason"][exit_reason]["count"] += 1
        self._stats["by_exit_reason"][exit_reason]["total_pnl"] += outcome.realized_pnl
        if outcome.was_profitable:
            self._stats["by_exit_reason"][exit_reason]["profitable"] += 1
        
        quality_tier = outcome.original_quality_tier or "unknown"
        if quality_tier not in self._stats["by_quality_tier"]:
            self._stats["by_quality_tier"][quality_tier] = {
                "count": 0, "total_pnl": 0.0, "profitable": 0, "avg_score": 0.0
            }
        tier_stats = self._stats["by_quality_tier"][quality_tier]
        tier_stats["count"] += 1
        tier_stats["total_pnl"] += outcome.realized_pnl
        if outcome.was_profitable:
            tier_stats["profitable"] += 1
        tier_stats["avg_score"] = (
            (tier_stats["avg_score"] * (tier_stats["count"] - 1) + outcome.original_quantrascore)
            / tier_stats["count"]
        )
        
        total = self._stats["total_trades"]
        profitable = self._stats["profitable_trades"]
        losing = self._stats["losing_trades"]
        
        if profitable > 0:
            winners = [o.realized_pnl for o in self._outcomes if o.was_profitable]
            self._stats["avg_winner"] = sum(winners) / len(winners)
        
        if losing > 0:
            losers = [o.realized_pnl for o in self._outcomes if not o.was_profitable]
            self._stats["avg_loser"] = sum(losers) / len(losers)
        
        bars = [o.bars_held for o in self._outcomes]
        self._stats["avg_bars_held"] = sum(bars) / len(bars) if bars else 0.0
    
    def save(self) -> None:
        """Save feedback samples and statistics to disk."""
        try:
            with open(self.feedback_path, "w") as f:
                json.dump({
                    "samples": self._pending_samples,
                    "stats": self._stats,
                    "updated_at": datetime.utcnow().isoformat(),
                }, f, indent=2, default=str)
            
            if self._outcomes:
                with open(self.trade_log_path, "w") as f:
                    json.dump({
                        "trades": [o.to_dict() for o in self._outcomes],
                        "updated_at": datetime.utcnow().isoformat(),
                    }, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"[TradeOutcomeTracker] Failed to save: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics."""
        total = self._stats["total_trades"]
        profitable = self._stats["profitable_trades"]
        
        return {
            **self._stats,
            "win_rate": profitable / total if total > 0 else 0.0,
            "profit_factor": self._calculate_profit_factor(),
            "expectancy": self._calculate_expectancy(),
            "samples_pending": len(self._pending_samples),
        }
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        gross_profit = sum(o.realized_pnl for o in self._outcomes if o.was_profitable)
        gross_loss = abs(sum(o.realized_pnl for o in self._outcomes if not o.was_profitable))
        
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_expectancy(self) -> float:
        """Calculate expectancy (average P&L per trade)."""
        if not self._outcomes:
            return 0.0
        return self._stats["total_pnl"] / len(self._outcomes)
    
    def get_samples_for_training(self) -> List[Dict[str, Any]]:
        """Get samples in format suitable for ApexLab training."""
        return self._pending_samples.copy()
    
    def export_to_apexlab(self, apexlab_path: str = "data/apexlab/feedback_samples.json") -> int:
        """
        Export samples to main ApexLab feedback file.
        
        Merges with existing samples if present.
        """
        apexlab_file = Path(apexlab_path)
        
        existing_samples = []
        if apexlab_file.exists():
            try:
                with open(apexlab_file, "r") as f:
                    existing_samples = json.load(f)
            except Exception:
                existing_samples = []
        
        new_count = 0
        existing_ids = {s.get("trade_id") for s in existing_samples}
        
        for sample in self._pending_samples:
            if sample.get("trade_id") not in existing_ids:
                sample["source"] = "autonomous_trading"
                existing_samples.append(sample)
                new_count += 1
        
        existing_samples = existing_samples[-self.max_samples:]
        
        try:
            with open(apexlab_file, "w") as f:
                json.dump(existing_samples, f, indent=2, default=str)
            
            logger.info(
                f"[TradeOutcomeTracker] Exported {new_count} new samples to ApexLab "
                f"(total: {len(existing_samples)})"
            )
        except Exception as e:
            logger.error(f"[TradeOutcomeTracker] Failed to export to ApexLab: {e}")
        
        return new_count
    
    def get_performance_by_score_range(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by QuantraScore ranges."""
        ranges = {
            "90-100": {"min": 90, "max": 100, "trades": [], "pnl": 0.0},
            "80-89": {"min": 80, "max": 89, "trades": [], "pnl": 0.0},
            "70-79": {"min": 70, "max": 79, "trades": [], "pnl": 0.0},
            "60-69": {"min": 60, "max": 69, "trades": [], "pnl": 0.0},
            "<60": {"min": 0, "max": 59, "trades": [], "pnl": 0.0},
        }
        
        for outcome in self._outcomes:
            score = outcome.original_quantrascore
            for range_name, range_data in ranges.items():
                if range_data["min"] <= score <= range_data["max"]:
                    range_data["trades"].append(outcome)
                    range_data["pnl"] += outcome.realized_pnl
                    break
        
        result = {}
        for range_name, range_data in ranges.items():
            trades = range_data["trades"]
            count = len(trades)
            profitable = sum(1 for t in trades if t.was_profitable)
            
            result[range_name] = {
                "count": count,
                "win_rate": profitable / count if count > 0 else 0.0,
                "total_pnl": range_data["pnl"],
                "avg_pnl": range_data["pnl"] / count if count > 0 else 0.0,
            }
        
        return result
    
    def get_recent_outcomes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the N most recent outcomes."""
        return [o.to_dict() for o in self._outcomes[-n:]]
    
    def clear(self) -> None:
        """Clear all tracked outcomes (use with caution)."""
        self._outcomes = []
        self._pending_samples = []
        self._stats = {
            "total_trades": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "largest_winner": 0.0,
            "largest_loser": 0.0,
            "avg_bars_held": 0.0,
            "by_exit_reason": {},
            "by_quality_tier": {},
        }
        logger.warning("[TradeOutcomeTracker] All data cleared")
