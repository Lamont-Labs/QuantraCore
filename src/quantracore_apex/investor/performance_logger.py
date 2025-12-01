"""
Automated Performance Metrics Logger.

Captures daily performance metrics for investor reporting:
- Returns (daily, cumulative)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Drawdown tracking
- Benchmark comparisons

All data stored in investor_logs for institutional due diligence.
"""

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

PERFORMANCE_LOGS_DIR = Path("investor_logs/performance")


@dataclass
class DailyPerformanceSnapshot:
    """Daily performance metrics snapshot."""
    date: str
    timestamp: str
    
    starting_equity: float
    ending_equity: float
    daily_pnl: float
    daily_return_pct: float
    
    cumulative_return_pct: float
    high_water_mark: float
    current_drawdown: float
    current_drawdown_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    
    realized_pnl: float
    unrealized_pnl: float
    
    trades_executed: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    gross_exposure: float
    net_exposure: float
    leverage: float
    
    sharpe_ratio_30d: Optional[float] = None
    sortino_ratio_30d: Optional[float] = None
    calmar_ratio_ytd: Optional[float] = None
    
    spy_return_pct: Optional[float] = None
    alpha_vs_spy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MonthlyPerformanceSummary:
    """Monthly performance summary for investor reports."""
    month: str
    year: int
    
    starting_equity: float
    ending_equity: float
    net_pnl: float
    return_pct: float
    
    trading_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    best_day_pnl: float
    best_day_date: str
    worst_day_pnl: float
    worst_day_date: str
    
    max_drawdown_pct: float
    avg_daily_return_pct: float
    volatility_annualized: float
    
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceLogger:
    """
    Automated performance logging for investor due diligence.
    
    Captures all metrics an institutional investor would need
    to evaluate strategy performance.
    """
    
    def __init__(self):
        PERFORMANCE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._daily_returns: List[float] = []
        self._load_historical_returns()
    
    def _load_historical_returns(self):
        """Load historical returns from logs."""
        try:
            history_file = PERFORMANCE_LOGS_DIR / "return_history.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    data = json.load(f)
                    self._daily_returns = data.get("daily_returns", [])
        except Exception as e:
            logger.warning(f"Could not load return history: {e}")
    
    def _save_historical_returns(self):
        """Save historical returns to logs."""
        history_file = PERFORMANCE_LOGS_DIR / "return_history.json"
        with open(history_file, "w") as f:
            json.dump({
                "daily_returns": self._daily_returns[-365:],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
    
    def log_daily_performance(
        self,
        starting_equity: float,
        ending_equity: float,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        trades_executed: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
        gross_exposure: float = 0.0,
        net_exposure: float = 0.0,
        spy_return_pct: Optional[float] = None,
    ) -> DailyPerformanceSnapshot:
        """
        Log daily performance snapshot.
        
        Called at end of each trading day to capture metrics.
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        
        daily_pnl = ending_equity - starting_equity
        daily_return_pct = (daily_pnl / starting_equity * 100) if starting_equity > 0 else 0.0
        
        self._daily_returns.append(daily_return_pct)
        if len(self._daily_returns) > 365:
            self._daily_returns = self._daily_returns[-365:]
        
        initial_equity = 100000
        cumulative_return_pct = ((ending_equity - initial_equity) / initial_equity * 100)
        
        high_water_mark = max(ending_equity, self._get_high_water_mark())
        current_drawdown = high_water_mark - ending_equity
        current_drawdown_pct = (current_drawdown / high_water_mark * 100) if high_water_mark > 0 else 0.0
        max_dd, max_dd_pct = self._calculate_max_drawdown()
        
        leverage = gross_exposure / ending_equity if ending_equity > 0 else 0.0
        win_rate = (winning_trades / trades_executed * 100) if trades_executed > 0 else 0.0
        
        sharpe = self._calculate_sharpe_ratio(30)
        sortino = self._calculate_sortino_ratio(30)
        calmar = self._calculate_calmar_ratio()
        
        alpha = daily_return_pct - spy_return_pct if spy_return_pct is not None else None
        
        snapshot = DailyPerformanceSnapshot(
            date=date_str,
            timestamp=now.isoformat(),
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            daily_pnl=daily_pnl,
            daily_return_pct=round(daily_return_pct, 4),
            cumulative_return_pct=round(cumulative_return_pct, 4),
            high_water_mark=high_water_mark,
            current_drawdown=current_drawdown,
            current_drawdown_pct=round(current_drawdown_pct, 4),
            max_drawdown=max_dd,
            max_drawdown_pct=round(max_dd_pct, 4),
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            trades_executed=trades_executed,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            leverage=round(leverage, 2),
            sharpe_ratio_30d=round(sharpe, 4) if sharpe else None,
            sortino_ratio_30d=round(sortino, 4) if sortino else None,
            calmar_ratio_ytd=round(calmar, 4) if calmar else None,
            spy_return_pct=spy_return_pct,
            alpha_vs_spy=round(alpha, 4) if alpha is not None else None,
        )
        
        self._save_daily_snapshot(snapshot)
        self._save_historical_returns()
        self._update_high_water_mark(high_water_mark)
        
        return snapshot
    
    def _get_high_water_mark(self) -> float:
        """Get stored high water mark."""
        try:
            hwm_file = PERFORMANCE_LOGS_DIR / "high_water_mark.json"
            if hwm_file.exists():
                with open(hwm_file, "r") as f:
                    return json.load(f).get("hwm", 100000)
        except:
            pass
        return 100000
    
    def _update_high_water_mark(self, hwm: float):
        """Update high water mark."""
        hwm_file = PERFORMANCE_LOGS_DIR / "high_water_mark.json"
        with open(hwm_file, "w") as f:
            json.dump({
                "hwm": hwm,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, f)
    
    def _calculate_max_drawdown(self) -> tuple:
        """Calculate maximum drawdown from return history."""
        if not self._daily_returns:
            return 0.0, 0.0
        
        cumulative = 100000
        peak = cumulative
        max_dd = 0.0
        
        for ret in self._daily_returns:
            cumulative = cumulative * (1 + ret / 100)
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        
        max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0.0
        return max_dd, max_dd_pct
    
    def _calculate_sharpe_ratio(self, days: int = 30) -> Optional[float]:
        """Calculate Sharpe ratio over specified period."""
        returns = self._daily_returns[-days:]
        if len(returns) < 5:
            return None
        
        avg_return = sum(returns) / len(returns)
        std_dev = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))
        
        if std_dev == 0:
            return None
        
        annualized_return = avg_return * 252
        annualized_std = std_dev * math.sqrt(252)
        risk_free_rate = 5.0
        
        return (annualized_return - risk_free_rate) / annualized_std
    
    def _calculate_sortino_ratio(self, days: int = 30) -> Optional[float]:
        """Calculate Sortino ratio (downside deviation only)."""
        returns = self._daily_returns[-days:]
        if len(returns) < 5:
            return None
        
        avg_return = sum(returns) / len(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return None
        
        downside_dev = math.sqrt(sum(r ** 2 for r in negative_returns) / len(returns))
        
        if downside_dev == 0:
            return None
        
        annualized_return = avg_return * 252
        annualized_downside = downside_dev * math.sqrt(252)
        risk_free_rate = 5.0
        
        return (annualized_return - risk_free_rate) / annualized_downside
    
    def _calculate_calmar_ratio(self) -> Optional[float]:
        """Calculate Calmar ratio (return / max drawdown)."""
        if not self._daily_returns:
            return None
        
        total_return = 1.0
        for r in self._daily_returns:
            total_return *= (1 + r / 100)
        
        cagr = (total_return ** (252 / len(self._daily_returns)) - 1) * 100
        _, max_dd_pct = self._calculate_max_drawdown()
        
        if max_dd_pct == 0:
            return None
        
        return cagr / max_dd_pct
    
    def _save_daily_snapshot(self, snapshot: DailyPerformanceSnapshot):
        """Save daily snapshot to log file."""
        date_str = datetime.now(timezone.utc).strftime("%Y%m")
        log_file = PERFORMANCE_LOGS_DIR / f"daily_performance_{date_str}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(snapshot.to_dict()) + "\n")
        
        logger.info(f"Logged daily performance: {snapshot.daily_return_pct:.2f}%")
    
    def generate_monthly_summary(self, year: int, month: int) -> Optional[MonthlyPerformanceSummary]:
        """Generate monthly performance summary."""
        date_str = f"{year}{month:02d}"
        log_file = PERFORMANCE_LOGS_DIR / f"daily_performance_{date_str}.jsonl"
        
        if not log_file.exists():
            return None
        
        snapshots = []
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    snapshots.append(json.loads(line))
        
        if not snapshots:
            return None
        
        daily_pnls = [s["daily_pnl"] for s in snapshots]
        daily_returns = [s["daily_return_pct"] for s in snapshots]
        
        best_idx = daily_pnls.index(max(daily_pnls))
        worst_idx = daily_pnls.index(min(daily_pnls))
        
        avg_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns) if daily_returns else 0
        volatility = math.sqrt(variance) * math.sqrt(252)
        
        summary = MonthlyPerformanceSummary(
            month=f"{year}-{month:02d}",
            year=year,
            starting_equity=snapshots[0]["starting_equity"],
            ending_equity=snapshots[-1]["ending_equity"],
            net_pnl=sum(daily_pnls),
            return_pct=sum(daily_returns),
            trading_days=len(snapshots),
            total_trades=sum(s["trades_executed"] for s in snapshots),
            winning_trades=sum(s["winning_trades"] for s in snapshots),
            losing_trades=sum(s["losing_trades"] for s in snapshots),
            win_rate=sum(s["winning_trades"] for s in snapshots) / max(1, sum(s["trades_executed"] for s in snapshots)) * 100,
            best_day_pnl=max(daily_pnls),
            best_day_date=snapshots[best_idx]["date"],
            worst_day_pnl=min(daily_pnls),
            worst_day_date=snapshots[worst_idx]["date"],
            max_drawdown_pct=max(s["max_drawdown_pct"] for s in snapshots),
            avg_daily_return_pct=avg_return,
            volatility_annualized=volatility,
        )
        
        summary_file = PERFORMANCE_LOGS_DIR / f"monthly_summary_{date_str}.json"
        with open(summary_file, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        return summary
    
    def get_performance_status(self) -> Dict[str, Any]:
        """Get current performance status for API."""
        now = datetime.now(timezone.utc)
        
        return {
            "status": "operational",
            "last_update": now.isoformat(),
            "metrics": {
                "return_history_days": len(self._daily_returns),
                "sharpe_30d": self._calculate_sharpe_ratio(30),
                "sortino_30d": self._calculate_sortino_ratio(30),
                "calmar_ytd": self._calculate_calmar_ratio(),
                "max_drawdown": self._calculate_max_drawdown()[1],
                "high_water_mark": self._get_high_water_mark(),
            },
            "log_files": str(PERFORMANCE_LOGS_DIR),
        }


_performance_logger = None


def get_performance_logger() -> PerformanceLogger:
    """Get singleton performance logger instance."""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger
