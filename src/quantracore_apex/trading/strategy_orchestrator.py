"""
Multi-Strategy Orchestrator for QuantraCore Apex.

Manages multiple concurrent trading strategies (Swing, Scalp, HFT, MonsterRunner)
with centralized position management, risk arbitration, and broker access.

LONG ONLY - No short positions ever.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from threading import Lock
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Supported trading strategy types."""
    SWING = "swing"
    SCALP = "scalp"
    HFT = "hft"
    MONSTER_RUNNER = "monster_runner"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"


class SignalDirection(Enum):
    """Trade direction - LONG ONLY system."""
    LONG = "long"
    EXIT = "exit"


class IntentStatus(Enum):
    """Status of a trading intent."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    strategy_type: StrategyType
    name: str
    enabled: bool = True
    budget_pct: float = 0.20
    max_positions: int = 5
    max_position_pct: float = 0.10
    default_stop_loss_pct: float = 0.08
    default_take_profit_pct: float = 0.50
    hold_time_hours: tuple = (24, 120)
    priority: int = 1
    min_score_threshold: float = 0.5
    

STRATEGY_CONFIGS: Dict[StrategyType, StrategyConfig] = {
    StrategyType.SWING: StrategyConfig(
        strategy_type=StrategyType.SWING,
        name="Swing Trading (EOD)",
        budget_pct=0.40,
        max_positions=6,
        max_position_pct=0.10,
        default_stop_loss_pct=0.08,
        default_take_profit_pct=0.50,
        hold_time_hours=(48, 120),
        priority=2,
        min_score_threshold=0.5,
    ),
    StrategyType.SCALP: StrategyConfig(
        strategy_type=StrategyType.SCALP,
        name="Scalping (Intraday)",
        budget_pct=0.20,
        max_positions=4,
        max_position_pct=0.05,
        default_stop_loss_pct=0.02,
        default_take_profit_pct=0.05,
        hold_time_hours=(0.5, 8),
        priority=3,
        min_score_threshold=0.6,
    ),
    StrategyType.HFT: StrategyConfig(
        strategy_type=StrategyType.HFT,
        name="High Frequency",
        budget_pct=0.10,
        max_positions=3,
        max_position_pct=0.03,
        default_stop_loss_pct=0.01,
        default_take_profit_pct=0.02,
        hold_time_hours=(0.01, 1),
        priority=4,
        min_score_threshold=0.7,
    ),
    StrategyType.MONSTER_RUNNER: StrategyConfig(
        strategy_type=StrategyType.MONSTER_RUNNER,
        name="MonsterRunner (Extreme Moves)",
        budget_pct=0.25,
        max_positions=3,
        max_position_pct=0.08,
        default_stop_loss_pct=0.15,
        default_take_profit_pct=1.00,
        hold_time_hours=(24, 168),
        priority=1,
        min_score_threshold=0.8,
    ),
    StrategyType.MOMENTUM: StrategyConfig(
        strategy_type=StrategyType.MOMENTUM,
        name="Momentum",
        budget_pct=0.15,
        max_positions=4,
        max_position_pct=0.05,
        default_stop_loss_pct=0.05,
        default_take_profit_pct=0.15,
        hold_time_hours=(4, 48),
        priority=3,
        min_score_threshold=0.55,
    ),
    StrategyType.BREAKOUT: StrategyConfig(
        strategy_type=StrategyType.BREAKOUT,
        name="Breakout",
        budget_pct=0.15,
        max_positions=3,
        max_position_pct=0.06,
        default_stop_loss_pct=0.06,
        default_take_profit_pct=0.20,
        hold_time_hours=(8, 72),
        priority=2,
        min_score_threshold=0.6,
    ),
}


@dataclass
class TradingIntent:
    """
    A trading intent from a strategy.
    
    All strategies emit intents which are then arbitrated by the Risk Arbiter.
    """
    intent_id: str
    strategy_type: StrategyType
    symbol: str
    direction: SignalDirection
    score: float
    requested_shares: int = 0
    requested_value: float = 0.0
    entry_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: IntentStatus = IntentStatus.PENDING
    rejection_reason: str = ""
    created_at: str = ""
    processed_at: str = ""
    order_id: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.intent_id:
            import uuid
            self.intent_id = f"{self.strategy_type.value}_{uuid.uuid4().hex[:8]}"


@dataclass
class StrategyPosition:
    """A position opened by a specific strategy."""
    symbol: str
    strategy_type: StrategyType
    shares: int
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    entry_time: str
    order_id: str
    intent_id: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.enabled = config.enabled
        self.name = config.name
        
    @abstractmethod
    def generate_signals(self, symbols: List[str]) -> List[TradingIntent]:
        """Generate trading intents for the given symbols."""
        pass
    
    @abstractmethod
    def get_exit_signals(self, positions: List[StrategyPosition]) -> List[TradingIntent]:
        """Generate exit intents for current positions."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        return {
            "name": self.name,
            "type": self.config.strategy_type.value,
            "enabled": self.enabled,
            "budget_pct": self.config.budget_pct,
            "max_positions": self.config.max_positions,
            "priority": self.config.priority,
        }


class RiskArbiter:
    """
    Central risk management and intent arbitration.
    
    Enforces:
    - LONG ONLY (no shorts ever)
    - Per-strategy budget limits
    - Symbol exclusivity (one position per symbol across all strategies)
    - Max position limits per strategy
    - Priority-based conflict resolution
    - Global max exposure
    """
    
    def __init__(
        self,
        max_total_positions: int = 15,
        max_total_exposure_pct: float = 0.95,
        min_cash_reserve_pct: float = 0.05,
    ):
        self.max_total_positions = max_total_positions
        self.max_total_exposure_pct = max_total_exposure_pct
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self._lock = Lock()
        self.active_positions: Dict[str, StrategyPosition] = {}
        self.strategy_exposure: Dict[StrategyType, float] = {}
        self.pending_intents: List[TradingIntent] = []
        self.processed_intents: List[TradingIntent] = []
        
    def arbitrate(
        self,
        intents: List[TradingIntent],
        equity: float,
        available_cash: float,
        current_positions: List[str],
    ) -> List[TradingIntent]:
        """
        Arbitrate a batch of intents and return approved ones.
        
        Higher priority strategies win conflicts.
        """
        with self._lock:
            approved = []
            remaining_cash = available_cash * (1 - self.min_cash_reserve_pct)
            
            intents = sorted(
                intents, 
                key=lambda x: (STRATEGY_CONFIGS[x.strategy_type].priority, -x.score)
            )
            
            symbols_claimed = set(current_positions)
            strategy_position_counts: Dict[StrategyType, int] = {}
            
            for pos in self.active_positions.values():
                strategy_position_counts[pos.strategy_type] = \
                    strategy_position_counts.get(pos.strategy_type, 0) + 1
            
            for intent in intents:
                rejection = self._check_intent(
                    intent, equity, remaining_cash, symbols_claimed, 
                    strategy_position_counts, current_positions
                )
                
                if rejection:
                    intent.status = IntentStatus.REJECTED
                    intent.rejection_reason = rejection
                    intent.processed_at = datetime.now().isoformat()
                    self.processed_intents.append(intent)
                    continue
                
                intent.status = IntentStatus.APPROVED
                intent.processed_at = datetime.now().isoformat()
                approved.append(intent)
                
                if intent.direction == SignalDirection.LONG:
                    symbols_claimed.add(intent.symbol)
                    remaining_cash -= intent.requested_value
                    strategy_position_counts[intent.strategy_type] = \
                        strategy_position_counts.get(intent.strategy_type, 0) + 1
            
            return approved
    
    def _check_intent(
        self,
        intent: TradingIntent,
        equity: float,
        remaining_cash: float,
        symbols_claimed: set,
        strategy_counts: Dict[StrategyType, int],
        current_positions: List[str],
    ) -> Optional[str]:
        """Check if an intent should be rejected. Returns rejection reason or None."""
        
        if intent.direction != SignalDirection.LONG and intent.direction != SignalDirection.EXIT:
            return "Only LONG positions allowed (no shorting)"
        
        if intent.direction == SignalDirection.EXIT:
            if intent.symbol not in current_positions:
                return f"Cannot exit {intent.symbol} - not in positions"
            return None
        
        if intent.symbol in symbols_claimed:
            return f"Symbol {intent.symbol} already claimed by another strategy"
        
        if intent.symbol in current_positions:
            return f"Already holding position in {intent.symbol}"
        
        config = STRATEGY_CONFIGS[intent.strategy_type]
        current_count = strategy_counts.get(intent.strategy_type, 0)
        if current_count >= config.max_positions:
            return f"Max positions ({config.max_positions}) reached for {intent.strategy_type.value}"
        
        total_positions = sum(strategy_counts.values())
        if total_positions >= self.max_total_positions:
            return f"Max total positions ({self.max_total_positions}) reached"
        
        if intent.score < config.min_score_threshold:
            return f"Score {intent.score:.2f} below threshold {config.min_score_threshold}"
        
        max_position_value = equity * config.max_position_pct
        if intent.requested_value > max_position_value:
            return f"Position value ${intent.requested_value:,.2f} exceeds max ${max_position_value:,.2f}"
        
        if intent.requested_value > remaining_cash:
            return f"Insufficient cash: need ${intent.requested_value:,.2f}, have ${remaining_cash:,.2f}"
        
        return None
    
    def register_position(self, position: StrategyPosition):
        """Register a new position from an executed intent."""
        with self._lock:
            self.active_positions[position.symbol] = position
            
    def remove_position(self, symbol: str):
        """Remove a closed position."""
        with self._lock:
            if symbol in self.active_positions:
                del self.active_positions[symbol]
    
    def get_strategy_positions(self, strategy_type: StrategyType) -> List[StrategyPosition]:
        """Get all positions for a specific strategy."""
        with self._lock:
            return [p for p in self.active_positions.values() 
                    if p.strategy_type == strategy_type]
    
    def get_status(self) -> Dict[str, Any]:
        """Get arbiter status."""
        with self._lock:
            positions_by_strategy = {}
            for pos in self.active_positions.values():
                key = pos.strategy_type.value
                if key not in positions_by_strategy:
                    positions_by_strategy[key] = []
                positions_by_strategy[key].append(pos.symbol)
            
            return {
                "total_positions": len(self.active_positions),
                "max_positions": self.max_total_positions,
                "positions_by_strategy": positions_by_strategy,
                "pending_intents": len(self.pending_intents),
                "processed_today": len(self.processed_intents),
            }


class StrategyOrchestrator:
    """
    Central orchestrator for all trading strategies.
    
    Manages strategy lifecycle, coordinates signal generation,
    arbitrates intents, and routes orders through the broker.
    """
    
    def __init__(
        self,
        broker_adapter=None,
        log_dir: str = "investor_logs/multi_strategy/",
    ):
        self.strategies: Dict[StrategyType, BaseStrategy] = {}
        self.arbiter = RiskArbiter()
        self.broker = broker_adapter
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.last_run: Dict[StrategyType, str] = {}
        self.enabled = True
        
        logger.info("[StrategyOrchestrator] Initialized multi-strategy system")
    
    def register_strategy(self, strategy: BaseStrategy):
        """Register a strategy with the orchestrator."""
        with self._lock:
            self.strategies[strategy.config.strategy_type] = strategy
            logger.info(f"[StrategyOrchestrator] Registered strategy: {strategy.name}")
    
    def unregister_strategy(self, strategy_type: StrategyType):
        """Unregister a strategy."""
        with self._lock:
            if strategy_type in self.strategies:
                del self.strategies[strategy_type]
                logger.info(f"[StrategyOrchestrator] Unregistered: {strategy_type.value}")
    
    def enable_strategy(self, strategy_type: StrategyType, enabled: bool = True):
        """Enable or disable a strategy."""
        with self._lock:
            if strategy_type in self.strategies:
                self.strategies[strategy_type].enabled = enabled
                logger.info(f"[StrategyOrchestrator] {strategy_type.value} enabled={enabled}")
    
    def run_cycle(
        self,
        symbols: List[str],
        equity: float,
        available_cash: float,
        current_positions: List[str],
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run one full cycle across all enabled strategies.
        
        1. Collect intents from all enabled strategies
        2. Arbitrate intents (priority, budget, conflicts)
        3. Execute approved intents through broker
        4. Return comprehensive report
        """
        cycle_start = datetime.now()
        all_intents: List[TradingIntent] = []
        strategy_reports = {}
        
        with self._lock:
            enabled_strategies = [
                s for s in self.strategies.values() if s.enabled
            ]
        
        for strategy in enabled_strategies:
            try:
                intents = strategy.generate_signals(symbols)
                all_intents.extend(intents)
                strategy_reports[strategy.config.strategy_type.value] = {
                    "intents_generated": len(intents),
                    "symbols_analyzed": len(symbols),
                }
                self.last_run[strategy.config.strategy_type] = cycle_start.isoformat()
            except Exception as e:
                logger.error(f"[StrategyOrchestrator] {strategy.name} error: {e}")
                strategy_reports[strategy.config.strategy_type.value] = {
                    "error": str(e),
                }
        
        approved_intents = self.arbiter.arbitrate(
            all_intents, equity, available_cash, current_positions
        )
        
        executed_orders = []
        if not dry_run and self.broker and approved_intents:
            for intent in approved_intents:
                try:
                    order_result = self._execute_intent(intent)
                    executed_orders.append(order_result)
                except Exception as e:
                    logger.error(f"[StrategyOrchestrator] Execution error: {e}")
                    intent.status = IntentStatus.FAILED
        
        report = {
            "timestamp": cycle_start.isoformat(),
            "dry_run": dry_run,
            "strategies_run": len(enabled_strategies),
            "strategy_reports": strategy_reports,
            "total_intents": len(all_intents),
            "approved_intents": len(approved_intents),
            "rejected_intents": len(all_intents) - len(approved_intents),
            "executed_orders": len(executed_orders),
            "approved_details": [
                {
                    "symbol": i.symbol,
                    "strategy": i.strategy_type.value,
                    "score": i.score,
                    "direction": i.direction.value,
                    "value": i.requested_value,
                }
                for i in approved_intents
            ],
            "arbiter_status": self.arbiter.get_status(),
            "cycle_duration_ms": (datetime.now() - cycle_start).total_seconds() * 1000,
        }
        
        self._log_cycle(report)
        return report
    
    def _execute_intent(self, intent: TradingIntent) -> Dict[str, Any]:
        """Execute an approved intent through the broker."""
        if intent.direction == SignalDirection.LONG:
            config = STRATEGY_CONFIGS[intent.strategy_type]
            result = self.broker.place_bracket_order(
                symbol=intent.symbol,
                qty=intent.requested_shares,
                side="buy",
                limit_price=intent.entry_price,
                stop_loss_pct=config.default_stop_loss_pct,
                take_profit_pct=config.default_take_profit_pct,
            )
            
            if result.get("status") in ["accepted", "filled", "new"]:
                intent.status = IntentStatus.EXECUTED
                intent.order_id = result.get("order_id", "")
                
                position = StrategyPosition(
                    symbol=intent.symbol,
                    strategy_type=intent.strategy_type,
                    shares=intent.requested_shares,
                    entry_price=intent.entry_price,
                    stop_loss_price=intent.stop_loss_price,
                    take_profit_price=intent.take_profit_price,
                    entry_time=datetime.now().isoformat(),
                    order_id=intent.order_id,
                    intent_id=intent.intent_id,
                )
                self.arbiter.register_position(position)
            else:
                intent.status = IntentStatus.FAILED
                
            return result
        
        elif intent.direction == SignalDirection.EXIT:
            result = self.broker.close_position(intent.symbol)
            if result.get("success"):
                intent.status = IntentStatus.EXECUTED
                self.arbiter.remove_position(intent.symbol)
            return result
        
        return {"error": "Unknown direction"}
    
    def _log_cycle(self, report: Dict[str, Any]):
        """Log cycle report to file."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"cycle_{date_str}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(report) + "\n")
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "strategies_registered": len(self.strategies),
                "strategies": {
                    st.value: self.strategies[st].get_status() 
                    for st in self.strategies
                },
                "last_runs": {
                    st.value: ts for st, ts in self.last_run.items()
                },
                "arbiter": self.arbiter.get_status(),
            }


_orchestrator: Optional[StrategyOrchestrator] = None


def get_orchestrator() -> StrategyOrchestrator:
    """Get or create the global strategy orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        from ..broker.adapters.alpaca_adapter import AlpacaPaperAdapter
        _orchestrator = StrategyOrchestrator(broker_adapter=AlpacaPaperAdapter())
    return _orchestrator
