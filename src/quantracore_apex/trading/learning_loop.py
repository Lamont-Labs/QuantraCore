"""
Learning Loop Module.

Analyzes trade outcomes to identify what's working and what isn't.
Provides feedback for strategy improvement.
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy: str
    period_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float
    avg_hold_hours: float
    best_trade_pnl: float
    worst_trade_pnl: float
    stop_loss_rate: float
    take_profit_rate: float
    status: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LearningLoop:
    """
    Analyzes trading performance and provides learning insights.
    
    Monitors:
    - Overall win rate and P&L
    - Per-strategy performance
    - Pattern recognition (what setups work best)
    - Risk management effectiveness
    
    Outputs:
    - Weekly performance reports
    - Strategy recommendations
    - Model retraining triggers
    """
    
    REPORTS_DIR = Path("logs/learning_reports/")
    
    WIN_RATE_THRESHOLD = 50.0
    MIN_TRADES_FOR_EVALUATION = 5
    UNDERPERFORMING_THRESHOLD = 40.0
    RETRAIN_TRIGGER_TRADES = 50
    
    def __init__(self):
        self._lock = threading.Lock()
        self._last_analysis: Optional[datetime] = None
        self._performance_history: List[Dict[str, Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("[LearningLoop] Initialized")
    
    def analyze_performance(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze recent trading performance.
        
        Returns comprehensive analysis with recommendations.
        """
        try:
            from src.quantracore_apex.trading.trade_outcome_tracker import get_trade_outcome_tracker
            
            tracker = get_trade_outcome_tracker()
            stats = tracker.get_stats(days=days)
            
            if stats["total_trades"] == 0:
                return {
                    "status": "insufficient_data",
                    "message": f"No trades in the last {days} days",
                    "recommendation": "Run scans to generate trading signals",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            
            overall_status = self._evaluate_overall_performance(stats)
            
            strategy_analysis = {}
            if "by_strategy" in stats:
                for strat, data in stats["by_strategy"].items():
                    strategy_analysis[strat] = self._evaluate_strategy(strat, data, days)
            
            recommendations = self._generate_recommendations(stats, strategy_analysis)
            
            should_retrain = self._check_retrain_trigger(stats)
            
            analysis = {
                "period_days": days,
                "timestamp": datetime.utcnow().isoformat(),
                "overall": {
                    "status": overall_status,
                    "total_trades": stats["total_trades"],
                    "win_rate": round(stats["win_rate"], 1),
                    "avg_pnl_pct": round(stats["avg_pnl_pct"], 2),
                    "total_pnl_pct": round(stats["total_pnl_pct"], 2),
                    "avg_hold_hours": round(stats["avg_hold_hours"], 1),
                },
                "risk_management": {
                    "stop_losses_hit": stats.get("stop_losses_hit", 0),
                    "take_profits_hit": stats.get("take_profits_hit", 0),
                    "stop_loss_rate": round(
                        stats.get("stop_losses_hit", 0) / stats["total_trades"] * 100, 1
                    ) if stats["total_trades"] else 0,
                    "take_profit_rate": round(
                        stats.get("take_profits_hit", 0) / stats["total_trades"] * 100, 1
                    ) if stats["total_trades"] else 0,
                },
                "by_strategy": strategy_analysis,
                "recommendations": recommendations,
                "model_status": {
                    "should_retrain": should_retrain,
                    "trades_since_last_retrain": stats["total_trades"],
                    "retrain_threshold": self.RETRAIN_TRIGGER_TRADES,
                },
                "best_trade": stats.get("best_trade"),
                "worst_trade": stats.get("worst_trade"),
            }
            
            self._save_report(analysis)
            
            with self._lock:
                self._performance_history.append(analysis)
                if len(self._performance_history) > 100:
                    self._performance_history = self._performance_history[-100:]
                self._last_analysis = datetime.utcnow()
            
            return analysis
            
        except Exception as e:
            logger.error(f"[LearningLoop] Analysis error: {e}")
            import traceback
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    def _evaluate_overall_performance(self, stats: Dict[str, Any]) -> str:
        """Evaluate overall trading performance."""
        win_rate = stats.get("win_rate", 0)
        avg_pnl = stats.get("avg_pnl_pct", 0)
        total_trades = stats.get("total_trades", 0)
        
        if total_trades < self.MIN_TRADES_FOR_EVALUATION:
            return "insufficient_data"
        
        if win_rate >= 60 and avg_pnl >= 5:
            return "excellent"
        elif win_rate >= 50 and avg_pnl >= 2:
            return "good"
        elif win_rate >= 40 and avg_pnl >= 0:
            return "acceptable"
        elif win_rate >= 30:
            return "needs_improvement"
        else:
            return "poor"
    
    def _evaluate_strategy(
        self, 
        strategy: str, 
        data: Dict[str, Any], 
        days: int
    ) -> Dict[str, Any]:
        """Evaluate a specific strategy's performance."""
        trades = data.get("trades", 0)
        win_rate = data.get("win_rate", 0)
        avg_pnl = data.get("avg_pnl_pct", 0)
        
        if trades < self.MIN_TRADES_FOR_EVALUATION:
            status = "insufficient_data"
            recommendations = ["Need more trades to evaluate"]
        elif win_rate >= 60:
            status = "excellent"
            recommendations = ["Continue using this strategy", "Consider increasing allocation"]
        elif win_rate >= 50:
            status = "good"
            recommendations = ["Strategy performing well", "Monitor for consistency"]
        elif win_rate >= self.UNDERPERFORMING_THRESHOLD:
            status = "acceptable"
            recommendations = ["Review entry criteria", "Consider tighter stops"]
        else:
            status = "underperforming"
            recommendations = [
                "Consider disabling temporarily",
                "Review and refine entry signals",
                "Check if market conditions have changed"
            ]
        
        return {
            "trades": trades,
            "win_rate": round(win_rate, 1),
            "avg_pnl_pct": round(avg_pnl, 2),
            "status": status,
            "recommendations": recommendations,
        }
    
    def _generate_recommendations(
        self, 
        stats: Dict[str, Any], 
        strategy_analysis: Dict[str, Dict]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        win_rate = stats.get("win_rate", 0)
        stop_loss_rate = stats.get("stop_losses_hit", 0) / stats.get("total_trades", 1) * 100
        take_profit_rate = stats.get("take_profits_hit", 0) / stats.get("total_trades", 1) * 100
        avg_hold = stats.get("avg_hold_hours", 0)
        
        if win_rate < 45:
            recommendations.append("Win rate below target - review entry signal thresholds")
        
        if stop_loss_rate > 30:
            recommendations.append("High stop-loss hit rate - consider wider stops or better entries")
        
        if take_profit_rate < 20 and win_rate > 50:
            recommendations.append("Many winners not hitting take-profit - consider partial profit taking")
        
        if avg_hold > 120:
            recommendations.append("Average hold time exceeds 5 days - review time-based exit rules")
        elif avg_hold < 4:
            recommendations.append("Very short hold times - verify this isn't premature exit")
        
        for strat, analysis in strategy_analysis.items():
            if analysis.get("status") == "underperforming":
                recommendations.append(f"Strategy '{strat}' underperforming - consider disabling")
            elif analysis.get("status") == "excellent":
                recommendations.append(f"Strategy '{strat}' performing well - consider increasing allocation")
        
        if not recommendations:
            recommendations.append("System performing within acceptable parameters")
        
        return recommendations
    
    def _check_retrain_trigger(self, stats: Dict[str, Any]) -> bool:
        """Check if model should be retrained based on accumulated trades."""
        total_trades = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0)
        
        if total_trades >= self.RETRAIN_TRIGGER_TRADES:
            return True
        
        if total_trades >= 20 and win_rate < 35:
            return True
        
        return False
    
    def _save_report(self, analysis: Dict[str, Any]):
        """Save analysis report to file."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = self.REPORTS_DIR / f"learning_report_{timestamp}.json"
            
            with open(filepath, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"[LearningLoop] Report saved to {filepath}")
        except Exception as e:
            logger.error(f"[LearningLoop] Failed to save report: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get learning loop status."""
        return {
            "last_analysis": self._last_analysis.isoformat() if self._last_analysis else None,
            "reports_generated": len(self._performance_history),
            "running": self._running,
            "thresholds": {
                "win_rate_target": self.WIN_RATE_THRESHOLD,
                "min_trades_for_eval": self.MIN_TRADES_FOR_EVALUATION,
                "underperforming_threshold": self.UNDERPERFORMING_THRESHOLD,
                "retrain_trigger_trades": self.RETRAIN_TRIGGER_TRADES,
            },
        }
    
    def get_recent_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis reports."""
        with self._lock:
            return self._performance_history[-limit:]
    
    def _weekly_analysis_loop(self):
        """Background loop for weekly analysis."""
        logger.info("[LearningLoop] Weekly analysis loop started")
        
        while self._running:
            try:
                now = datetime.utcnow()
                
                should_run = False
                if self._last_analysis is None:
                    should_run = True
                elif (now - self._last_analysis).days >= 7:
                    should_run = True
                
                if should_run:
                    logger.info("[LearningLoop] Running weekly analysis")
                    self.analyze_performance(days=7)
                
                threading.Event().wait(3600)
                
            except Exception as e:
                logger.error(f"[LearningLoop] Analysis loop error: {e}")
                threading.Event().wait(300)
        
        logger.info("[LearningLoop] Weekly analysis loop stopped")
    
    def start(self):
        """Start the learning loop."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._weekly_analysis_loop, daemon=True)
        self._thread.start()
        logger.info("[LearningLoop] Started")
    
    def stop(self):
        """Stop the learning loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None


_learning_loop_instance: Optional[LearningLoop] = None


def get_learning_loop() -> LearningLoop:
    """Get or create the singleton learning loop instance."""
    global _learning_loop_instance
    if _learning_loop_instance is None:
        _learning_loop_instance = LearningLoop()
    return _learning_loop_instance
