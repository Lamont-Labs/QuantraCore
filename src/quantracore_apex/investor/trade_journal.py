"""
Investor Trade Journal - Paper Trade Logging for Investor Reporting.

Logs all paper trades to a dedicated repository folder with clean,
investor-friendly formatting for due diligence and reporting.
"""

import os
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)

INVESTOR_LOGS_DIR = "investor_logs"


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"


@dataclass
class InvestorTradeEntry:
    """Single trade entry for investor reporting."""
    trade_id: str
    timestamp: str
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    notional_value: float
    order_type: str
    status: str
    broker: str
    signal_source: str
    quantrascore: float
    regime: str
    risk_tier: str
    protocols_fired: List[str]
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    hold_duration_hours: Optional[float] = None
    exit_reason: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_csv_row(self) -> List[Any]:
        """Convert to CSV row format."""
        return [
            self.trade_id,
            self.timestamp,
            self.symbol,
            self.direction,
            self.entry_price,
            self.quantity,
            self.notional_value,
            self.order_type,
            self.status,
            self.broker,
            self.signal_source,
            self.quantrascore,
            self.regime,
            self.risk_tier,
            ";".join(self.protocols_fired[:10]),
            self.exit_price or "",
            self.exit_timestamp or "",
            self.realized_pnl or "",
            self.realized_pnl_pct or "",
            self.hold_duration_hours or "",
            self.exit_reason or "",
            self.notes,
        ]
    
    @staticmethod
    def csv_headers() -> List[str]:
        return [
            "trade_id",
            "timestamp",
            "symbol",
            "direction",
            "entry_price",
            "quantity",
            "notional_value",
            "order_type",
            "status",
            "broker",
            "signal_source",
            "quantrascore",
            "regime",
            "risk_tier",
            "protocols_fired",
            "exit_price",
            "exit_timestamp",
            "realized_pnl",
            "realized_pnl_pct",
            "hold_duration_hours",
            "exit_reason",
            "notes",
        ]


@dataclass
class DailySummary:
    """Daily trading summary for investors."""
    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    open_trades: int
    gross_pnl: float
    net_pnl: float
    win_rate: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    total_volume: float
    symbols_traded: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InvestorTradeJournal:
    """
    Investor-grade trade journal for paper trading records.
    
    Maintains:
    1. JSON trade log (machine-readable)
    2. CSV trade log (spreadsheet-compatible)
    3. Daily summaries
    4. Cumulative performance metrics
    """
    
    def __init__(self, base_dir: str = INVESTOR_LOGS_DIR):
        self.base_dir = Path(base_dir)
        self.trades_dir = self.base_dir / "trades"
        self.summaries_dir = self.base_dir / "summaries"
        
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        
        self._trade_counter = 0
        self._session_trades: List[InvestorTradeEntry] = []
        self._cumulative_stats = self._load_cumulative_stats()
        
        logger.info(f"[InvestorTradeJournal] Initialized at {self.base_dir}")
    
    def _load_cumulative_stats(self) -> Dict[str, Any]:
        """Load cumulative statistics from disk."""
        stats_path = self.base_dir / "cumulative_stats.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "total_trades": 0,
            "total_winning": 0,
            "total_losing": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "total_volume": 0.0,
            "largest_winner": 0.0,
            "largest_loser": 0.0,
            "first_trade_date": None,
            "last_trade_date": None,
            "symbols_traded": [],
        }
    
    def _save_cumulative_stats(self):
        """Save cumulative statistics to disk."""
        stats_path = self.base_dir / "cumulative_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self._cumulative_stats, f, indent=2)
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"TRD_{timestamp}_{self._trade_counter:06d}"
    
    def log_trade_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        order_type: str,
        broker: str,
        signal_source: str = "ApexEngine",
        quantrascore: float = 50.0,
        regime: str = "unknown",
        risk_tier: str = "medium",
        protocols_fired: Optional[List[str]] = None,
        notes: str = "",
    ) -> InvestorTradeEntry:
        """
        Log a new trade entry (position opened).
        
        Returns the trade entry for tracking.
        """
        trade_id = self._generate_trade_id()
        timestamp = datetime.utcnow().isoformat()
        notional = entry_price * quantity
        
        entry = InvestorTradeEntry(
            trade_id=trade_id,
            timestamp=timestamp,
            symbol=symbol.upper(),
            direction=direction.upper(),
            entry_price=entry_price,
            quantity=quantity,
            notional_value=notional,
            order_type=order_type.upper(),
            status=TradeStatus.OPEN.value,
            broker=broker,
            signal_source=signal_source,
            quantrascore=quantrascore,
            regime=regime,
            risk_tier=risk_tier,
            protocols_fired=protocols_fired or [],
            notes=notes,
        )
        
        self._session_trades.append(entry)
        self._append_to_json_log(entry)
        self._append_to_csv_log(entry)
        self._update_cumulative_stats(entry)
        
        logger.info(
            f"[InvestorTradeJournal] Trade opened: {trade_id} "
            f"{direction} {quantity} {symbol} @ ${entry_price:.2f}"
        )
        
        return entry
    
    def log_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "signal",
        notes: str = "",
    ) -> Optional[InvestorTradeEntry]:
        """
        Log a trade exit (position closed).
        
        Updates the existing trade entry with exit details and P&L.
        """
        entry = self._find_trade(trade_id)
        if not entry:
            logger.warning(f"[InvestorTradeJournal] Trade not found: {trade_id}")
            return None
        
        exit_timestamp = datetime.utcnow()
        entry_time = datetime.fromisoformat(entry.timestamp)
        hold_hours = (exit_timestamp - entry_time).total_seconds() / 3600
        
        if entry.direction == "LONG":
            pnl = (exit_price - entry.entry_price) * entry.quantity
            pnl_pct = ((exit_price / entry.entry_price) - 1) * 100
        else:
            pnl = (entry.entry_price - exit_price) * entry.quantity
            pnl_pct = ((entry.entry_price / exit_price) - 1) * 100
        
        entry.exit_price = exit_price
        entry.exit_timestamp = exit_timestamp.isoformat()
        entry.realized_pnl = pnl
        entry.realized_pnl_pct = pnl_pct
        entry.hold_duration_hours = hold_hours
        entry.exit_reason = exit_reason
        entry.status = TradeStatus.CLOSED.value
        if notes:
            entry.notes = (entry.notes + " | " + notes).strip(" | ")
        
        self._update_trade_in_log(entry)
        self._update_cumulative_stats_exit(entry)
        
        logger.info(
            f"[InvestorTradeJournal] Trade closed: {trade_id} "
            f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)"
        )
        
        return entry
    
    def _find_trade(self, trade_id: str) -> Optional[InvestorTradeEntry]:
        """Find a trade by ID."""
        for trade in self._session_trades:
            if trade.trade_id == trade_id:
                return trade
        
        today = datetime.utcnow().strftime("%Y%m%d")
        json_path = self.trades_dir / f"trades_{today}.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    trades = json.load(f)
                    for t in trades:
                        if t.get("trade_id") == trade_id:
                            return InvestorTradeEntry(**t)
            except:
                pass
        
        return None
    
    def _append_to_json_log(self, entry: InvestorTradeEntry):
        """Append trade to daily JSON log."""
        today = datetime.utcnow().strftime("%Y%m%d")
        json_path = self.trades_dir / f"trades_{today}.json"
        
        trades = []
        if json_path.exists():
            try:
                with open(json_path) as f:
                    trades = json.load(f)
            except:
                pass
        
        trades.append(entry.to_dict())
        
        with open(json_path, "w") as f:
            json.dump(trades, f, indent=2, default=str)
    
    def _append_to_csv_log(self, entry: InvestorTradeEntry):
        """Append trade to daily CSV log."""
        today = datetime.utcnow().strftime("%Y%m%d")
        csv_path = self.trades_dir / f"trades_{today}.csv"
        
        write_header = not csv_path.exists()
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(InvestorTradeEntry.csv_headers())
            writer.writerow(entry.to_csv_row())
    
    def _update_trade_in_log(self, entry: InvestorTradeEntry):
        """Update an existing trade in the JSON log."""
        today = datetime.utcnow().strftime("%Y%m%d")
        json_path = self.trades_dir / f"trades_{today}.json"
        
        if not json_path.exists():
            return
        
        try:
            with open(json_path) as f:
                trades = json.load(f)
            
            for i, t in enumerate(trades):
                if t.get("trade_id") == entry.trade_id:
                    trades[i] = entry.to_dict()
                    break
            
            with open(json_path, "w") as f:
                json.dump(trades, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to update trade log: {e}")
    
    def _update_cumulative_stats(self, entry: InvestorTradeEntry):
        """Update cumulative stats for new trade."""
        self._cumulative_stats["total_trades"] += 1
        self._cumulative_stats["total_volume"] += entry.notional_value
        
        if entry.symbol not in self._cumulative_stats["symbols_traded"]:
            self._cumulative_stats["symbols_traded"].append(entry.symbol)
        
        if not self._cumulative_stats["first_trade_date"]:
            self._cumulative_stats["first_trade_date"] = entry.timestamp
        self._cumulative_stats["last_trade_date"] = entry.timestamp
        
        self._save_cumulative_stats()
    
    def _update_cumulative_stats_exit(self, entry: InvestorTradeEntry):
        """Update cumulative stats after trade exit."""
        if entry.realized_pnl is None:
            return
        
        pnl = entry.realized_pnl
        self._cumulative_stats["gross_pnl"] += pnl
        self._cumulative_stats["net_pnl"] += pnl
        
        if pnl > 0:
            self._cumulative_stats["total_winning"] += 1
            if pnl > self._cumulative_stats["largest_winner"]:
                self._cumulative_stats["largest_winner"] = pnl
        else:
            self._cumulative_stats["total_losing"] += 1
            if pnl < self._cumulative_stats["largest_loser"]:
                self._cumulative_stats["largest_loser"] = pnl
        
        self._save_cumulative_stats()
    
    def generate_daily_summary(self, date: Optional[str] = None) -> Optional[DailySummary]:
        """Generate summary for a specific date."""
        date = date or datetime.utcnow().strftime("%Y%m%d")
        json_path = self.trades_dir / f"trades_{date}.json"
        
        if not json_path.exists():
            return None
        
        try:
            with open(json_path) as f:
                trades = json.load(f)
        except:
            return None
        
        closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
        open_trades = [t for t in trades if t.get("status") == "OPEN"]
        
        winners = [t for t in closed_trades if (t.get("realized_pnl") or 0) > 0]
        losers = [t for t in closed_trades if (t.get("realized_pnl") or 0) < 0]
        
        gross_pnl = sum(t.get("realized_pnl", 0) for t in closed_trades)
        win_rate = len(winners) / len(closed_trades) if closed_trades else 0
        
        avg_winner = sum(t.get("realized_pnl", 0) for t in winners) / len(winners) if winners else 0
        avg_loser = sum(t.get("realized_pnl", 0) for t in losers) / len(losers) if losers else 0
        
        largest_winner = max((t.get("realized_pnl", 0) for t in winners), default=0)
        largest_loser = min((t.get("realized_pnl", 0) for t in losers), default=0)
        
        total_volume = sum(t.get("notional_value", 0) for t in trades)
        symbols = list(set(t.get("symbol", "") for t in trades))
        
        summary = DailySummary(
            date=date,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            open_trades=len(open_trades),
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl,
            win_rate=win_rate,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            total_volume=total_volume,
            symbols_traded=symbols,
        )
        
        summary_path = self.summaries_dir / f"summary_{date}.json"
        with open(summary_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        return summary
    
    def get_cumulative_stats(self) -> Dict[str, Any]:
        """Get cumulative trading statistics."""
        stats = self._cumulative_stats.copy()
        
        total = stats.get("total_winning", 0) + stats.get("total_losing", 0)
        stats["win_rate"] = stats["total_winning"] / total if total > 0 else 0
        stats["symbols_count"] = len(stats.get("symbols_traded", []))
        
        return stats
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get most recent trades."""
        all_trades = []
        
        for json_file in sorted(self.trades_dir.glob("trades_*.json"), reverse=True):
            try:
                with open(json_file) as f:
                    trades = json.load(f)
                    all_trades.extend(trades)
                    if len(all_trades) >= limit:
                        break
            except:
                continue
        
        return sorted(all_trades, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
    
    def export_all_trades_csv(self) -> str:
        """Export all trades to a single CSV file."""
        export_path = self.base_dir / "all_trades_export.csv"
        
        all_trades = []
        for json_file in sorted(self.trades_dir.glob("trades_*.json")):
            try:
                with open(json_file) as f:
                    trades = json.load(f)
                    all_trades.extend(trades)
            except:
                continue
        
        with open(export_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(InvestorTradeEntry.csv_headers())
            for trade in sorted(all_trades, key=lambda x: x.get("timestamp", "")):
                entry = InvestorTradeEntry(**trade)
                writer.writerow(entry.to_csv_row())
        
        logger.info(f"[InvestorTradeJournal] Exported {len(all_trades)} trades to {export_path}")
        return str(export_path)


_journal_instance: Optional[InvestorTradeJournal] = None


def get_trade_journal() -> InvestorTradeJournal:
    """Get singleton trade journal instance."""
    global _journal_instance
    if _journal_instance is None:
        _journal_instance = InvestorTradeJournal()
    return _journal_instance


__all__ = [
    "InvestorTradeJournal",
    "InvestorTradeEntry",
    "DailySummary",
    "get_trade_journal",
]
