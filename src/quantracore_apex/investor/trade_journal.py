"""
Investor Trade Journal - Comprehensive Paper Trade Logging.

Logs all paper trades with complete institutional-grade detail for:
- Due diligence packages
- Investor reporting
- Regulatory audits
- Performance attribution
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
class AccountSnapshot:
    """Account state at time of trade."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    day_pnl: float
    total_pnl: float
    open_positions_count: int
    total_exposure: float
    exposure_pct: float
    margin_used: float
    margin_available: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SignalQuality:
    """Signal quality metrics for the trade."""
    quantrascore: float
    score_bucket: str
    confidence: float
    monster_runner_score: float
    monster_runner_fired: bool
    runner_probability: float
    avoid_trade_probability: float
    quality_tier: str
    entropy_state: str
    suppression_state: str
    drift_state: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketContext:
    """Market conditions at time of trade."""
    regime: str
    vix_level: float
    vix_percentile: float
    sector_momentum: str
    market_breadth: float
    spy_change_pct: float
    trading_session: str
    market_phase: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskAssessment:
    """Risk assessment for the trade."""
    risk_tier: str
    risk_approved: bool
    risk_score: float
    max_position_size: float
    actual_position_size: float
    position_pct_of_max: float
    stop_loss_price: Optional[float]
    stop_loss_pct: Optional[float]
    take_profit_price: Optional[float]
    take_profit_pct: Optional[float]
    risk_reward_ratio: Optional[float]
    max_drawdown_allowed: float
    volatility_adjusted: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProtocolAnalysis:
    """Protocol analysis details."""
    protocols_fired: List[str]
    protocol_count: int
    tier_protocols: List[str]
    monster_runner_protocols: List[str]
    learning_protocols: List[str]
    omega_alerts: List[str]
    omega_blocked: bool
    consensus_direction: str
    protocol_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionDetails:
    """Order execution details."""
    order_id: str
    order_type: str
    time_in_force: str
    limit_price: Optional[float]
    stop_price: Optional[float]
    filled_price: float
    slippage: float
    slippage_pct: float
    commission: float
    execution_time_ms: int
    fill_quality: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InvestorTradeEntry:
    """
    Comprehensive trade entry with all investor-required information.
    
    Contains complete audit trail for:
    - Due diligence reviews
    - Regulatory compliance
    - Performance attribution
    - Risk monitoring
    """
    trade_id: str
    timestamp: str
    symbol: str
    company_name: str
    sector: str
    direction: str
    entry_price: float
    quantity: float
    notional_value: float
    status: str
    broker: str
    
    account_snapshot: AccountSnapshot
    signal_quality: SignalQuality
    market_context: MarketContext
    risk_assessment: RiskAssessment
    protocol_analysis: ProtocolAnalysis
    execution_details: ExecutionDetails
    
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    hold_duration_hours: Optional[float] = None
    exit_reason: Optional[str] = None
    
    research_notes: str = ""
    compliance_notes: str = ""
    audit_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "sector": self.sector,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "notional_value": self.notional_value,
            "status": self.status,
            "broker": self.broker,
            "account_snapshot": self.account_snapshot.to_dict(),
            "signal_quality": self.signal_quality.to_dict(),
            "market_context": self.market_context.to_dict(),
            "risk_assessment": self.risk_assessment.to_dict(),
            "protocol_analysis": self.protocol_analysis.to_dict(),
            "execution_details": self.execution_details.to_dict(),
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp,
            "realized_pnl": self.realized_pnl,
            "realized_pnl_pct": self.realized_pnl_pct,
            "hold_duration_hours": self.hold_duration_hours,
            "exit_reason": self.exit_reason,
            "research_notes": self.research_notes,
            "compliance_notes": self.compliance_notes,
            "audit_hash": self.audit_hash,
        }
        return data
    
    def to_csv_row(self) -> List[Any]:
        """Convert to flat CSV row format."""
        return [
            self.trade_id,
            self.timestamp,
            self.symbol,
            self.company_name,
            self.sector,
            self.direction,
            self.entry_price,
            self.quantity,
            self.notional_value,
            self.status,
            self.broker,
            self.account_snapshot.equity,
            self.account_snapshot.cash,
            self.account_snapshot.portfolio_value,
            self.account_snapshot.exposure_pct,
            self.account_snapshot.open_positions_count,
            self.signal_quality.quantrascore,
            self.signal_quality.score_bucket,
            self.signal_quality.confidence,
            self.signal_quality.monster_runner_score,
            self.signal_quality.quality_tier,
            self.signal_quality.entropy_state,
            self.market_context.regime,
            self.market_context.vix_level,
            self.market_context.vix_percentile,
            self.market_context.spy_change_pct,
            self.risk_assessment.risk_tier,
            self.risk_assessment.risk_approved,
            self.risk_assessment.risk_score,
            self.risk_assessment.stop_loss_pct or "",
            self.risk_assessment.take_profit_pct or "",
            self.risk_assessment.risk_reward_ratio or "",
            self.protocol_analysis.protocol_count,
            ";".join(self.protocol_analysis.protocols_fired[:15]),
            ";".join(self.protocol_analysis.omega_alerts[:5]),
            self.execution_details.order_type,
            self.execution_details.slippage_pct,
            self.execution_details.commission,
            self.exit_price or "",
            self.exit_timestamp or "",
            self.realized_pnl or "",
            self.realized_pnl_pct or "",
            self.hold_duration_hours or "",
            self.exit_reason or "",
            self.research_notes[:200],
            self.audit_hash,
        ]
    
    @staticmethod
    def csv_headers() -> List[str]:
        return [
            "trade_id",
            "timestamp",
            "symbol",
            "company_name",
            "sector",
            "direction",
            "entry_price",
            "quantity",
            "notional_value",
            "status",
            "broker",
            "account_equity",
            "account_cash",
            "portfolio_value",
            "exposure_pct",
            "open_positions",
            "quantrascore",
            "score_bucket",
            "confidence",
            "monster_runner_score",
            "quality_tier",
            "entropy_state",
            "regime",
            "vix_level",
            "vix_percentile",
            "spy_change_pct",
            "risk_tier",
            "risk_approved",
            "risk_score",
            "stop_loss_pct",
            "take_profit_pct",
            "risk_reward_ratio",
            "protocol_count",
            "protocols_fired",
            "omega_alerts",
            "order_type",
            "slippage_pct",
            "commission",
            "exit_price",
            "exit_timestamp",
            "realized_pnl",
            "realized_pnl_pct",
            "hold_duration_hours",
            "exit_reason",
            "research_notes",
            "audit_hash",
        ]


@dataclass
class DailySummary:
    """Comprehensive daily trading summary for investors."""
    date: str
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    open_trades: int
    
    gross_pnl: float
    net_pnl: float
    commission_paid: float
    
    win_rate: float
    profit_factor: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    avg_hold_time_hours: float
    
    total_volume: float
    avg_trade_size: float
    max_exposure: float
    
    starting_equity: float
    ending_equity: float
    equity_change: float
    equity_change_pct: float
    
    vix_open: float
    vix_close: float
    spy_change_pct: float
    
    symbols_traded: List[str]
    best_symbol: str
    worst_symbol: str
    
    long_trades: int
    short_trades: int
    long_pnl: float
    short_pnl: float
    
    regimes_traded: Dict[str, int]
    risk_tiers_used: Dict[str, int]
    
    omega_blocks: int
    avg_quantrascore: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MonthlyReport:
    """Monthly performance report for investors."""
    month: str
    year: int
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    gross_pnl: float
    net_pnl: float
    total_commission: float
    
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    
    best_day: str
    best_day_pnl: float
    worst_day: str
    worst_day_pnl: float
    
    trading_days: int
    profitable_days: int
    losing_days: int
    
    avg_daily_pnl: float
    std_daily_pnl: float
    
    starting_equity: float
    ending_equity: float
    return_pct: float
    
    top_winners: List[Dict[str, Any]]
    top_losers: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InvestorTradeJournal:
    """
    Institutional-grade trade journal for investor reporting.
    
    Maintains comprehensive records for:
    1. Due diligence packages
    2. Regulatory audits
    3. Investor updates
    4. Performance attribution
    5. Risk monitoring
    """
    
    def __init__(self, base_dir: str = INVESTOR_LOGS_DIR):
        self.base_dir = Path(base_dir)
        self.trades_dir = self.base_dir / "trades"
        self.summaries_dir = self.base_dir / "summaries"
        self.reports_dir = self.base_dir / "reports"
        self.exports_dir = self.base_dir / "exports"
        
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        
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
            "total_commission": 0.0,
            "total_volume": 0.0,
            "largest_winner": 0.0,
            "largest_loser": 0.0,
            "max_drawdown": 0.0,
            "peak_equity": 0.0,
            "first_trade_date": None,
            "last_trade_date": None,
            "symbols_traded": [],
            "trading_days": 0,
            "profitable_days": 0,
            "total_long_trades": 0,
            "total_short_trades": 0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "avg_hold_time_hours": 0.0,
            "total_hold_time_hours": 0.0,
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
    
    def _generate_audit_hash(self, entry: Dict[str, Any]) -> str:
        """Generate audit hash for trade verification."""
        import hashlib
        data_str = json.dumps(entry, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def log_comprehensive_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        order_type: str,
        broker: str,
        order_id: str = "",
        company_name: str = "",
        sector: str = "Unknown",
        account_equity: float = 100000.0,
        account_cash: float = 100000.0,
        buying_power: float = 400000.0,
        portfolio_value: float = 100000.0,
        day_pnl: float = 0.0,
        total_pnl: float = 0.0,
        open_positions_count: int = 0,
        total_exposure: float = 0.0,
        margin_used: float = 0.0,
        quantrascore: float = 50.0,
        score_bucket: str = "neutral",
        confidence: float = 0.5,
        monster_runner_score: float = 0.0,
        monster_runner_fired: bool = False,
        runner_probability: float = 0.0,
        avoid_trade_probability: float = 0.0,
        quality_tier: str = "C",
        entropy_state: str = "mid",
        suppression_state: str = "none",
        drift_state: str = "none",
        regime: str = "unknown",
        vix_level: float = 20.0,
        vix_percentile: float = 50.0,
        sector_momentum: str = "neutral",
        market_breadth: float = 0.5,
        spy_change_pct: float = 0.0,
        trading_session: str = "regular",
        market_phase: str = "open",
        risk_tier: str = "medium",
        risk_approved: bool = True,
        risk_score: float = 50.0,
        max_position_size: float = 10000.0,
        stop_loss_price: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        risk_reward_ratio: Optional[float] = None,
        max_drawdown_allowed: float = 0.02,
        volatility_adjusted: bool = True,
        protocols_fired: Optional[List[str]] = None,
        tier_protocols: Optional[List[str]] = None,
        monster_runner_protocols: Optional[List[str]] = None,
        omega_alerts: Optional[List[str]] = None,
        omega_blocked: bool = False,
        consensus_direction: str = "neutral",
        protocol_confidence: float = 0.5,
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        slippage: float = 0.0,
        commission: float = 0.0,
        execution_time_ms: int = 0,
        research_notes: str = "",
        compliance_notes: str = "",
    ) -> InvestorTradeEntry:
        """
        Log a comprehensive trade entry with all investor-required information.
        """
        trade_id = self._generate_trade_id()
        timestamp = datetime.utcnow().isoformat()
        notional = entry_price * quantity
        exposure_pct = (total_exposure / account_equity * 100) if account_equity > 0 else 0
        slippage_pct = (slippage / entry_price * 100) if entry_price > 0 else 0
        
        account_snapshot = AccountSnapshot(
            equity=account_equity,
            cash=account_cash,
            buying_power=buying_power,
            portfolio_value=portfolio_value,
            day_pnl=day_pnl,
            total_pnl=total_pnl,
            open_positions_count=open_positions_count,
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            margin_used=margin_used,
            margin_available=buying_power - margin_used,
        )
        
        signal_quality = SignalQuality(
            quantrascore=quantrascore,
            score_bucket=score_bucket,
            confidence=confidence,
            monster_runner_score=monster_runner_score,
            monster_runner_fired=monster_runner_fired,
            runner_probability=runner_probability,
            avoid_trade_probability=avoid_trade_probability,
            quality_tier=quality_tier,
            entropy_state=entropy_state,
            suppression_state=suppression_state,
            drift_state=drift_state,
        )
        
        market_context = MarketContext(
            regime=regime,
            vix_level=vix_level,
            vix_percentile=vix_percentile,
            sector_momentum=sector_momentum,
            market_breadth=market_breadth,
            spy_change_pct=spy_change_pct,
            trading_session=trading_session,
            market_phase=market_phase,
        )
        
        risk_assessment = RiskAssessment(
            risk_tier=risk_tier,
            risk_approved=risk_approved,
            risk_score=risk_score,
            max_position_size=max_position_size,
            actual_position_size=notional,
            position_pct_of_max=(notional / max_position_size * 100) if max_position_size > 0 else 0,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_price=take_profit_price,
            take_profit_pct=take_profit_pct,
            risk_reward_ratio=risk_reward_ratio,
            max_drawdown_allowed=max_drawdown_allowed,
            volatility_adjusted=volatility_adjusted,
        )
        
        all_protocols = protocols_fired or []
        protocol_analysis = ProtocolAnalysis(
            protocols_fired=all_protocols,
            protocol_count=len(all_protocols),
            tier_protocols=tier_protocols or [p for p in all_protocols if p.startswith("T")],
            monster_runner_protocols=monster_runner_protocols or [p for p in all_protocols if p.startswith("MR")],
            learning_protocols=[p for p in all_protocols if p.startswith("LP")],
            omega_alerts=omega_alerts or [],
            omega_blocked=omega_blocked,
            consensus_direction=consensus_direction,
            protocol_confidence=protocol_confidence,
        )
        
        fill_quality = "good" if slippage_pct < 0.1 else ("fair" if slippage_pct < 0.5 else "poor")
        execution_details = ExecutionDetails(
            order_id=order_id,
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            filled_price=entry_price,
            slippage=slippage,
            slippage_pct=slippage_pct,
            commission=commission,
            execution_time_ms=execution_time_ms,
            fill_quality=fill_quality,
        )
        
        entry = InvestorTradeEntry(
            trade_id=trade_id,
            timestamp=timestamp,
            symbol=symbol.upper(),
            company_name=company_name or symbol.upper(),
            sector=sector,
            direction=direction.upper(),
            entry_price=entry_price,
            quantity=quantity,
            notional_value=notional,
            status=TradeStatus.OPEN.value,
            broker=broker,
            account_snapshot=account_snapshot,
            signal_quality=signal_quality,
            market_context=market_context,
            risk_assessment=risk_assessment,
            protocol_analysis=protocol_analysis,
            execution_details=execution_details,
            research_notes=research_notes,
            compliance_notes=compliance_notes,
        )
        
        entry.audit_hash = self._generate_audit_hash(entry.to_dict())
        
        self._session_trades.append(entry)
        self._append_to_json_log(entry)
        self._append_to_csv_log(entry)
        self._update_cumulative_stats(entry)
        
        logger.info(
            f"[InvestorTradeJournal] Trade logged: {trade_id} "
            f"{direction} {quantity} {symbol} @ ${entry_price:.2f} "
            f"(QS: {quantrascore:.1f}, Regime: {regime})"
        )
        
        return entry
    
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
        **kwargs,
    ) -> InvestorTradeEntry:
        """
        Simplified trade entry for backward compatibility.
        Calls comprehensive logging with defaults.
        """
        return self.log_comprehensive_trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            order_type=order_type,
            broker=broker,
            quantrascore=quantrascore,
            regime=regime,
            risk_tier=risk_tier,
            protocols_fired=protocols_fired,
            research_notes=notes,
            **kwargs,
        )
    
    def log_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "signal",
        commission: float = 0.0,
        notes: str = "",
    ) -> Optional[InvestorTradeEntry]:
        """
        Log a trade exit with P&L calculation.
        """
        entry = self._find_trade(trade_id)
        if not entry:
            logger.warning(f"[InvestorTradeJournal] Trade not found: {trade_id}")
            return None
        
        exit_timestamp = datetime.utcnow()
        entry_time = datetime.fromisoformat(entry.timestamp)
        hold_hours = (exit_timestamp - entry_time).total_seconds() / 3600
        
        if entry.direction == "LONG":
            pnl = (exit_price - entry.entry_price) * entry.quantity - commission
            pnl_pct = ((exit_price / entry.entry_price) - 1) * 100
        else:
            pnl = (entry.entry_price - exit_price) * entry.quantity - commission
            pnl_pct = ((entry.entry_price / exit_price) - 1) * 100
        
        entry.exit_price = exit_price
        entry.exit_timestamp = exit_timestamp.isoformat()
        entry.realized_pnl = pnl
        entry.realized_pnl_pct = pnl_pct
        entry.hold_duration_hours = hold_hours
        entry.exit_reason = exit_reason
        entry.status = TradeStatus.CLOSED.value
        if notes:
            entry.research_notes = (entry.research_notes + " | EXIT: " + notes).strip(" | ")
        
        entry.audit_hash = self._generate_audit_hash(entry.to_dict())
        
        self._update_trade_in_log(entry)
        self._update_cumulative_stats_exit(entry, commission)
        
        logger.info(
            f"[InvestorTradeJournal] Trade closed: {trade_id} "
            f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) "
            f"Hold: {hold_hours:.1f}h"
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
                            return self._dict_to_entry(t)
            except:
                pass
        
        return None
    
    def _dict_to_entry(self, data: Dict[str, Any]) -> InvestorTradeEntry:
        """Convert dictionary back to InvestorTradeEntry."""
        return InvestorTradeEntry(
            trade_id=data["trade_id"],
            timestamp=data["timestamp"],
            symbol=data["symbol"],
            company_name=data.get("company_name", data["symbol"]),
            sector=data.get("sector", "Unknown"),
            direction=data["direction"],
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            notional_value=data["notional_value"],
            status=data["status"],
            broker=data["broker"],
            account_snapshot=AccountSnapshot(**data.get("account_snapshot", {})) if data.get("account_snapshot") else AccountSnapshot(0,0,0,0,0,0,0,0,0,0,0),
            signal_quality=SignalQuality(**data.get("signal_quality", {})) if data.get("signal_quality") else SignalQuality(50,"neutral",0.5,0,False,0,0,"C","mid","none","none"),
            market_context=MarketContext(**data.get("market_context", {})) if data.get("market_context") else MarketContext("unknown",20,50,"neutral",0.5,0,"regular","open"),
            risk_assessment=RiskAssessment(**data.get("risk_assessment", {})) if data.get("risk_assessment") else RiskAssessment("medium",True,50,10000,0,0,None,None,None,None,None,0.02,True),
            protocol_analysis=ProtocolAnalysis(**data.get("protocol_analysis", {})) if data.get("protocol_analysis") else ProtocolAnalysis([],0,[],[],[],[],False,"neutral",0.5),
            execution_details=ExecutionDetails(**data.get("execution_details", {})) if data.get("execution_details") else ExecutionDetails("","MARKET","day",None,None,0,0,0,0,0,"good"),
            exit_price=data.get("exit_price"),
            exit_timestamp=data.get("exit_timestamp"),
            realized_pnl=data.get("realized_pnl"),
            realized_pnl_pct=data.get("realized_pnl_pct"),
            hold_duration_hours=data.get("hold_duration_hours"),
            exit_reason=data.get("exit_reason"),
            research_notes=data.get("research_notes", ""),
            compliance_notes=data.get("compliance_notes", ""),
            audit_hash=data.get("audit_hash", ""),
        )
    
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
        
        if entry.direction == "LONG":
            self._cumulative_stats["total_long_trades"] += 1
        else:
            self._cumulative_stats["total_short_trades"] += 1
        
        if entry.symbol not in self._cumulative_stats["symbols_traded"]:
            self._cumulative_stats["symbols_traded"].append(entry.symbol)
        
        if not self._cumulative_stats["first_trade_date"]:
            self._cumulative_stats["first_trade_date"] = entry.timestamp
        self._cumulative_stats["last_trade_date"] = entry.timestamp
        
        self._save_cumulative_stats()
    
    def _update_cumulative_stats_exit(self, entry: InvestorTradeEntry, commission: float = 0.0):
        """Update cumulative stats after trade exit."""
        if entry.realized_pnl is None:
            return
        
        pnl = entry.realized_pnl
        self._cumulative_stats["gross_pnl"] += pnl
        self._cumulative_stats["net_pnl"] += pnl
        self._cumulative_stats["total_commission"] += commission
        
        if entry.hold_duration_hours:
            self._cumulative_stats["total_hold_time_hours"] += entry.hold_duration_hours
            closed_trades = self._cumulative_stats["total_winning"] + self._cumulative_stats["total_losing"]
            if closed_trades > 0:
                self._cumulative_stats["avg_hold_time_hours"] = (
                    self._cumulative_stats["total_hold_time_hours"] / (closed_trades + 1)
                )
        
        if pnl > 0:
            self._cumulative_stats["total_winning"] += 1
            if entry.direction == "LONG":
                self._cumulative_stats["long_pnl"] += pnl
            else:
                self._cumulative_stats["short_pnl"] += pnl
            if pnl > self._cumulative_stats["largest_winner"]:
                self._cumulative_stats["largest_winner"] = pnl
        else:
            self._cumulative_stats["total_losing"] += 1
            if entry.direction == "LONG":
                self._cumulative_stats["long_pnl"] += pnl
            else:
                self._cumulative_stats["short_pnl"] += pnl
            if pnl < self._cumulative_stats["largest_loser"]:
                self._cumulative_stats["largest_loser"] = pnl
        
        self._save_cumulative_stats()
    
    def generate_daily_summary(self, date: Optional[str] = None) -> Optional[DailySummary]:
        """Generate comprehensive daily summary."""
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
        total_commission = sum(t.get("execution_details", {}).get("commission", 0) for t in trades)
        net_pnl = gross_pnl - total_commission
        
        win_rate = len(winners) / len(closed_trades) if closed_trades else 0
        
        winner_pnl = sum(t.get("realized_pnl", 0) for t in winners)
        loser_pnl = abs(sum(t.get("realized_pnl", 0) for t in losers))
        if loser_pnl > 0:
            profit_factor = winner_pnl / loser_pnl
        elif winner_pnl > 0:
            profit_factor = 999.99
        else:
            profit_factor = 0.0
        
        avg_winner = winner_pnl / len(winners) if winners else 0
        avg_loser = -loser_pnl / len(losers) if losers else 0
        
        largest_winner = max((t.get("realized_pnl", 0) for t in winners), default=0)
        largest_loser = min((t.get("realized_pnl", 0) for t in losers), default=0)
        
        hold_times = [t.get("hold_duration_hours", 0) for t in closed_trades if t.get("hold_duration_hours")]
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
        
        total_volume = sum(t.get("notional_value", 0) for t in trades)
        avg_trade_size = total_volume / len(trades) if trades else 0
        
        symbols = list(set(t.get("symbol", "") for t in trades))
        
        long_trades = [t for t in trades if t.get("direction") == "LONG"]
        short_trades = [t for t in trades if t.get("direction") == "SHORT"]
        long_pnl = sum(t.get("realized_pnl", 0) for t in long_trades if t.get("realized_pnl"))
        short_pnl = sum(t.get("realized_pnl", 0) for t in short_trades if t.get("realized_pnl"))
        
        regimes = {}
        risk_tiers = {}
        total_qs = 0
        for t in trades:
            regime = t.get("market_context", {}).get("regime", "unknown")
            regimes[regime] = regimes.get(regime, 0) + 1
            tier = t.get("risk_assessment", {}).get("risk_tier", "medium")
            risk_tiers[tier] = risk_tiers.get(tier, 0) + 1
            total_qs += t.get("signal_quality", {}).get("quantrascore", 50)
        
        avg_qs = total_qs / len(trades) if trades else 50
        
        omega_blocks = sum(1 for t in trades if t.get("protocol_analysis", {}).get("omega_blocked", False))
        
        symbol_pnl = {}
        for t in closed_trades:
            sym = t.get("symbol", "")
            symbol_pnl[sym] = symbol_pnl.get(sym, 0) + (t.get("realized_pnl") or 0)
        
        best_symbol = max(symbol_pnl.items(), key=lambda x: x[1], default=("", 0))[0]
        worst_symbol = min(symbol_pnl.items(), key=lambda x: x[1], default=("", 0))[0]
        
        summary = DailySummary(
            date=date,
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            open_trades=len(open_trades),
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            commission_paid=total_commission,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            avg_hold_time_hours=avg_hold_time,
            total_volume=total_volume,
            avg_trade_size=avg_trade_size,
            max_exposure=0,
            starting_equity=100000,
            ending_equity=100000 + net_pnl,
            equity_change=net_pnl,
            equity_change_pct=net_pnl / 100000 * 100 if net_pnl else 0,
            vix_open=20,
            vix_close=20,
            spy_change_pct=0,
            symbols_traded=symbols,
            best_symbol=best_symbol,
            worst_symbol=worst_symbol,
            long_trades=len(long_trades),
            short_trades=len(short_trades),
            long_pnl=long_pnl,
            short_pnl=short_pnl,
            regimes_traded=regimes,
            risk_tiers_used=risk_tiers,
            omega_blocks=omega_blocks,
            avg_quantrascore=avg_qs,
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
        
        if stats["largest_loser"] < 0:
            stats["profit_factor"] = stats["largest_winner"] / abs(stats["largest_loser"])
        elif stats["largest_winner"] > 0:
            stats["profit_factor"] = 999.99
        else:
            stats["profit_factor"] = 0.0
        
        stats["symbols_count"] = len(stats.get("symbols_traded", []))
        stats["long_win_rate"] = 0
        stats["short_win_rate"] = 0
        
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
        """Export all trades to a consolidated CSV file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        export_path = self.exports_dir / f"all_trades_export_{timestamp}.csv"
        
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
                entry = self._dict_to_entry(trade)
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
    "AccountSnapshot",
    "SignalQuality",
    "MarketContext",
    "RiskAssessment",
    "ProtocolAnalysis",
    "ExecutionDetails",
    "DailySummary",
    "MonthlyReport",
    "get_trade_journal",
]
