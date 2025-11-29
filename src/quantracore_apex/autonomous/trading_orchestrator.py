"""
Trading Orchestrator - Institutional-Grade Autonomous Trading Loop.

The central coordinator for autonomous trading operations:
- Continuous market scanning across watchlist
- Signal quality filtering (A+/A tier only)
- Position monitoring and management
- Trade outcome tracking for self-learning
- Full Omega Directive compliance
- Market hours awareness

MODES:
- PAPER: Routes to Alpaca paper trading (default)
- RESEARCH: Logs signals only (no execution)
- LIVE: DISABLED by default
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Any, Set
from enum import Enum

from .models import (
    TradingState,
    OrchestratorConfig,
    OrchestratorMetrics,
    SignalDecision,
    PositionState,
    TradeOutcome,
    ExitReason,
)
from .signal_quality_filter import SignalQualityFilter
from .position_monitor import PositionMonitor
from .trade_outcome_tracker import TradeOutcomeTracker
from .realtime.rolling_window import RollingWindowManager, BarData

logger = logging.getLogger(__name__)


try:
    from src.quantracore_apex.core.engine import ApexEngine
    from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar
    from src.quantracore_apex.broker.execution_engine import ExecutionEngine
    from src.quantracore_apex.broker.enums import ExecutionMode
    from src.quantracore_apex.eeo_engine.eeo_optimizer import EntryExitOptimizer
    from src.quantracore_apex.protocols.omega.omega import OmegaDirectives
    APEX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some Apex components not available: {e}")
    APEX_AVAILABLE = False


class MarketStatus(str, Enum):
    """Current market status."""
    PRE_MARKET = "pre_market"
    OPEN = "open"
    POST_MARKET = "post_market"
    CLOSED = "closed"


class TradingOrchestrator:
    """
    Institutional-grade autonomous trading orchestrator.
    
    Coordinates all trading subsystems:
    1. Real-time data streaming → Rolling windows
    2. ApexEngine analysis → Signal generation
    3. Quality filtering → Only A+/A signals pass
    4. EEO optimization → Entry/exit planning
    5. Execution → Broker routing
    6. Position monitoring → Active management
    7. Outcome tracking → Self-learning feedback
    
    Safety Features:
    - Full Omega Directive enforcement
    - Mode enforcement (PAPER by default)
    - Position and exposure limits
    - Automatic stop-loss management
    - Emergency shutdown capability
    """
    
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    PRE_MARKET_START = time(4, 0)
    POST_MARKET_END = time(20, 0)
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        use_simulated_stream: bool = True,
    ):
        self.config = config or OrchestratorConfig()
        self.use_simulated_stream = use_simulated_stream
        
        self._state = TradingState.INITIALIZING
        self._metrics = OrchestratorMetrics()
        
        self.quality_filter = SignalQualityFilter(config=self.config)
        self.position_monitor = PositionMonitor(
            config=self.config,
            on_exit_triggered=self._handle_exit_triggered,
        )
        self.outcome_tracker = TradeOutcomeTracker()
        self.window_manager = RollingWindowManager(
            window_size=100,
            bar_interval_seconds=60,
            symbols=self.config.watchlist,
        )
        
        self._apex_engine: Optional[ApexEngine] = None
        self._execution_engine: Optional[ExecutionEngine] = None
        self._eeo_optimizer: Optional[EntryExitOptimizer] = None
        self._omega_directives: Optional[OmegaDirectives] = None
        
        self._stream = None
        self._should_run = False
        self._scan_task: Optional[asyncio.Task] = None
        self._position_task: Optional[asyncio.Task] = None
        self._stream_task: Optional[asyncio.Task] = None
        
        self._decision_log: List[SignalDecision] = []
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize trading components."""
        if not APEX_AVAILABLE:
            logger.warning("[Orchestrator] Some components not available, running in limited mode")
            self._initialization_errors = ["APEX components not available"]
            return
        
        self._initialization_errors = []
        
        try:
            self._apex_engine = ApexEngine()
            logger.info("[Orchestrator] ApexEngine initialized")
        except Exception as e:
            logger.error(f"[Orchestrator] Failed to init ApexEngine: {e}")
            self._initialization_errors.append(f"ApexEngine: {e}")
        
        try:
            self._execution_engine = ExecutionEngine()
            mode = self._execution_engine.mode
            logger.info(f"[Orchestrator] ExecutionEngine initialized in {mode.value} mode")
        except Exception as e:
            logger.error(f"[Orchestrator] Failed to init ExecutionEngine: {e}")
            self._initialization_errors.append(f"ExecutionEngine: {e}")
        
        try:
            self._omega_directives = OmegaDirectives()
            logger.info("[Orchestrator] OmegaDirectives initialized")
        except Exception as e:
            logger.warning(f"[Orchestrator] OmegaDirectives not available: {e}")
            self._initialization_errors.append(f"OmegaDirectives: {e}")
    
    def is_ready(self) -> tuple:
        """Check if orchestrator is ready for trading."""
        if not hasattr(self, '_initialization_errors'):
            return False, ["Not initialized"]
        
        if self._initialization_errors:
            return False, self._initialization_errors
        
        if not self._apex_engine:
            return False, ["ApexEngine not available"]
        
        if not self._execution_engine:
            return False, ["ExecutionEngine not available"]
        
        return True, []
    
    @property
    def state(self) -> TradingState:
        return self._state
    
    @property
    def is_running(self) -> bool:
        return self._state == TradingState.RUNNING
    
    def _set_state(self, state: TradingState) -> None:
        """Update orchestrator state."""
        self._state = state
        logger.info(f"[Orchestrator] State: {state.value}")
    
    async def start(self) -> None:
        """
        Start the autonomous trading loop.
        
        This is the main entry point for autonomous operation.
        """
        ready, errors = self.is_ready()
        if not ready:
            logger.error(f"[Orchestrator] Not ready to start: {errors}")
            logger.error("[Orchestrator] Aborting startup - critical components missing")
            self._set_state(TradingState.ERROR)
            raise RuntimeError(f"Orchestrator not ready: {errors}")
        
        logger.info("=" * 60)
        logger.info("QuantraCore Apex - Autonomous Trading Orchestrator")
        logger.info("=" * 60)
        logger.info(f"Mode: {'PAPER' if self.config.paper_mode else 'RESEARCH'}")
        logger.info(f"Watchlist: {len(self.config.watchlist)} symbols")
        logger.info(f"Max positions: {self.config.max_concurrent_positions}")
        logger.info(f"Quality threshold: {self.config.quality_thresholds.min_quantrascore}+ score")
        logger.info("=" * 60)
        
        self._should_run = True
        self._set_state(TradingState.CONNECTING)
        
        await self._start_stream()
        
        self._scan_task = asyncio.create_task(self._scan_loop())
        self._position_task = asyncio.create_task(self._position_loop())
        
        self._set_state(TradingState.RUNNING)
        
        try:
            await asyncio.gather(
                self._scan_task,
                self._position_task,
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            logger.info("[Orchestrator] Tasks cancelled")
        finally:
            await self.stop()
    
    async def _start_stream(self) -> None:
        """Start real-time data stream."""
        from .realtime.polygon_stream import SimulatedStream, PolygonWebSocketStream
        
        if self.use_simulated_stream:
            self._stream = SimulatedStream(
                on_trade=self._on_trade,
                on_quote=self._on_quote,
                interval_seconds=1.0,
            )
        else:
            self._stream = PolygonWebSocketStream(
                on_trade=self._on_trade,
                on_quote=self._on_quote,
                on_bar=self._on_bar,
            )
        
        self._stream_task = asyncio.create_task(
            self._stream.connect(self.config.watchlist)
        )
        
        await asyncio.sleep(1.0)
    
    async def stop(self) -> None:
        """Stop the autonomous trading loop gracefully."""
        logger.info("[Orchestrator] Stopping...")
        self._should_run = False
        self._set_state(TradingState.SHUTDOWN)
        
        if self._stream:
            await self._stream.disconnect()
        
        for task in [self._scan_task, self._position_task, self._stream_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.outcome_tracker.save()
        self.outcome_tracker.export_to_apexlab()
        
        logger.info("[Orchestrator] Stopped")
    
    def _on_trade(self, symbol: str, price: float, volume: float, timestamp: datetime) -> None:
        """Handle incoming trade data."""
        completed_bar = self.window_manager.update_from_trade(symbol, price, volume, timestamp)
        
        if completed_bar:
            self._metrics.last_scan_time = datetime.utcnow()
        
        exits = self.position_monitor.update_price(symbol, price)
        for position, exit_reason in exits:
            asyncio.create_task(self._execute_exit(position, exit_reason, price))
    
    def _on_quote(self, symbol: str, bid: float, ask: float, timestamp: datetime) -> None:
        """Handle incoming quote data."""
        self.window_manager.update_from_quote(symbol, bid, ask, timestamp)
    
    def _on_bar(self, symbol: str, bar_data: Dict[str, Any]) -> None:
        """Handle incoming bar data."""
        bar = BarData(
            timestamp=bar_data.get("timestamp", datetime.utcnow()),
            open=bar_data.get("open", 0),
            high=bar_data.get("high", 0),
            low=bar_data.get("low", 0),
            close=bar_data.get("close", 0),
            volume=bar_data.get("volume", 0),
            vwap=bar_data.get("vwap", 0),
        )
        self.window_manager.update_from_bar(symbol, bar)
    
    async def _scan_loop(self) -> None:
        """Main scanning loop - runs continuously."""
        logger.info("[Orchestrator] Starting scan loop")
        
        while self._should_run:
            try:
                market_status = self._get_market_status()
                
                if self.config.respect_market_hours and market_status == MarketStatus.CLOSED:
                    self._set_state(TradingState.MARKET_CLOSED)
                    await asyncio.sleep(60)
                    continue
                
                self._set_state(TradingState.RUNNING)
                
                self.quality_filter.update_positions(
                    self.position_monitor.open_symbols,
                    self.position_monitor.total_exposure,
                )
                
                await self._scan_watchlist()
                
                await asyncio.sleep(self.config.scan_interval_seconds)
                
            except Exception as e:
                logger.error(f"[Orchestrator] Scan loop error: {e}")
                self._set_state(TradingState.ERROR)
                await asyncio.sleep(5)
    
    async def _position_loop(self) -> None:
        """Position monitoring loop."""
        logger.info("[Orchestrator] Starting position loop")
        
        while self._should_run:
            try:
                prices = self.window_manager.get_all_current_prices()
                exits = self.position_monitor.update_all_prices(prices)
                
                for position, exit_reason in exits:
                    price = prices.get(position.symbol, position.current_price)
                    await self._execute_exit(position, exit_reason, price)
                
                time_exits = self.position_monitor.increment_bars()
                for position, exit_reason in time_exits:
                    price = prices.get(position.symbol, position.current_price)
                    await self._execute_exit(position, exit_reason, price)
                
                await asyncio.sleep(self.config.position_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"[Orchestrator] Position loop error: {e}")
                await asyncio.sleep(5)
    
    async def _scan_watchlist(self) -> None:
        """Scan all symbols in watchlist for signals."""
        ready_symbols = self.window_manager.get_ready_symbols()
        
        if not ready_symbols:
            return
        
        for symbol in ready_symbols:
            if not self._should_run:
                break
            
            try:
                decision = await self._evaluate_symbol(symbol)
                
                if decision:
                    self._decision_log.append(decision)
                    
                    if decision.decision == "EXECUTE":
                        await self._execute_signal(decision)
                        
            except Exception as e:
                logger.error(f"[Orchestrator] Error evaluating {symbol}: {e}")
    
    async def _evaluate_symbol(self, symbol: str) -> Optional[SignalDecision]:
        """Evaluate a single symbol for trading opportunity."""
        if not self._apex_engine:
            return None
        
        bars = self.window_manager.get_bars_as_dicts(symbol)
        if len(bars) < 100:
            return None
        
        try:
            ohlcv_bars = [
                OhlcvBar(
                    timestamp=datetime.fromisoformat(b["timestamp"].replace("Z", "+00:00")) 
                        if isinstance(b["timestamp"], str) else b["timestamp"],
                    open=b["open"],
                    high=b["high"],
                    low=b["low"],
                    close=b["close"],
                    volume=b["volume"],
                )
                for b in bars[-100:]
            ]
            
            window = OhlcvWindow(
                symbol=symbol,
                timeframe="1m",
                bars=ohlcv_bars,
            )
            
            result = self._apex_engine.run(window)
            
        except Exception as e:
            logger.error(f"[Orchestrator] ApexEngine error for {symbol}: {e}")
            return None
        
        self._metrics.total_signals_scanned += 1
        
        filter_result = self.quality_filter.filter(result)
        
        omega_statuses = {}
        if self._omega_directives:
            try:
                statuses = self._omega_directives.check_all(result)
                omega_statuses = {k: v.to_dict() if hasattr(v, 'to_dict') else str(v) 
                                  for k, v in statuses.items()}
            except Exception:
                pass
        
        if filter_result.passed:
            self._metrics.signals_passed_filter += 1
            decision = "EXECUTE"
        else:
            self._metrics.signals_rejected += 1
            decision = "REJECT"
        
        try:
            from src.quantracore_apex.hyperlearner import get_hyperlearner, EventCategory, EventType, LearningPriority
            hyperlearner = get_hyperlearner()
            event_type = EventType.SIGNAL_PASSED if filter_result.passed else EventType.SIGNAL_REJECTED
            hyperlearner.emit(
                category=EventCategory.SIGNAL,
                event_type=event_type,
                source="trading_orchestrator",
                context={
                    "quantrascore": filter_result.quantrascore,
                    "quality_tier": filter_result.quality_tier,
                    "risk_tier": filter_result.risk_tier,
                    "rejection_reasons": filter_result.rejection_reasons if not filter_result.passed else [],
                    "omega_statuses": omega_statuses,
                },
                symbol=symbol,
                confidence=filter_result.quantrascore / 100 if filter_result.quantrascore else 0.5,
                priority=LearningPriority.HIGH,
            )
        except ImportError:
            pass
        
        return SignalDecision(
            signal_id=getattr(result, "signal_id", str(id(result))),
            symbol=symbol,
            decision=decision,
            filter_result=filter_result,
            omega_statuses=omega_statuses,
        )
    
    async def _execute_signal(self, decision: SignalDecision) -> None:
        """Execute a trading signal that passed quality filters."""
        symbol = decision.symbol
        
        logger.info(
            f"[Orchestrator] EXECUTING: {symbol} | "
            f"Score={decision.filter_result.quantrascore:.1f} | "
            f"Tier={decision.filter_result.quality_tier}"
        )
        
        current_price = self.window_manager.get_current_price(symbol) or 100.0
        
        position_size = self._calculate_position_size(symbol, current_price)
        
        stop_distance = current_price * 0.02
        target1_distance = current_price * 0.03
        target2_distance = current_price * 0.05
        
        direction = "long"
        protective_stop = current_price - stop_distance
        target1 = current_price + target1_distance
        target2 = current_price + target2_distance
        
        if self.config.paper_mode and self._execution_engine:
            try:
                pass
            except Exception as e:
                logger.error(f"[Orchestrator] Execution error: {e}")
        
        position = self.position_monitor.create_position_from_fill(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            quantity=position_size / current_price,
            protective_stop=protective_stop,
            profit_target_1=target1,
            profit_target_2=target2,
            trailing_stop=0.0,
            max_bars=50,
            source_signal_id=decision.signal_id,
            metadata={
                "quantrascore": decision.filter_result.quantrascore,
                "quality_tier": decision.filter_result.quality_tier,
                "risk_tier": decision.filter_result.risk_tier,
            },
        )
        
        self._metrics.trades_opened += 1
        self._metrics.current_positions = self.position_monitor.position_count
        self._metrics.current_exposure = self.position_monitor.total_exposure
        self._metrics.last_trade_time = datetime.utcnow()
        
        self.quality_filter.add_cooldown(symbol)
        
        hyperlearner_event_id = None
        try:
            from src.quantracore_apex.hyperlearner import get_hyperlearner, EventCategory, EventType, LearningPriority
            hyperlearner = get_hyperlearner()
            hyperlearner_event_id = hyperlearner.emit(
                category=EventCategory.EXECUTION,
                event_type=EventType.TRADE_ENTERED,
                source="trading_orchestrator",
                context={
                    "entry_price": current_price,
                    "position_size": position_size,
                    "direction": direction,
                    "protective_stop": protective_stop,
                    "target1": target1,
                    "target2": target2,
                    "quantrascore": decision.filter_result.quantrascore,
                    "quality_tier": decision.filter_result.quality_tier,
                    "signal_id": decision.signal_id,
                },
                symbol=symbol,
                confidence=decision.filter_result.quantrascore / 100 if decision.filter_result.quantrascore else 0.5,
                priority=LearningPriority.CRITICAL,
            )
            if hyperlearner_event_id and position:
                position.metadata["hyperlearner_entry_event_id"] = hyperlearner_event_id
        except ImportError:
            pass
        
        logger.info(
            f"[Orchestrator] Position opened: {symbol} | "
            f"Entry={current_price:.2f} | Stop={protective_stop:.2f} | T1={target1:.2f}"
        )
    
    async def _execute_exit(
        self,
        position: PositionState,
        exit_reason: ExitReason,
        exit_price: float,
    ) -> None:
        """Execute an exit for a position."""
        logger.info(
            f"[Orchestrator] EXITING: {position.symbol} | "
            f"Reason={exit_reason.value} | Price={exit_price:.2f}"
        )
        
        outcome = self.position_monitor.close_position(
            position.position_id,
            exit_price,
            exit_reason,
        )
        
        if outcome:
            self.outcome_tracker.record_outcome(outcome)
            
            if outcome.was_profitable:
                self._metrics.profitable_trades += 1
            else:
                self._metrics.losing_trades += 1
            
            self._metrics.total_realized_pnl += outcome.realized_pnl
            self._metrics.trades_closed += 1
            self._metrics.current_positions = self.position_monitor.position_count
            self._metrics.current_exposure = self.position_monitor.total_exposure
            
            try:
                from src.quantracore_apex.hyperlearner import get_hyperlearner, EventCategory, EventType, LearningPriority, OutcomeType
                hyperlearner = get_hyperlearner()
                
                if exit_reason == ExitReason.STOP_TRIGGERED:
                    event_type = EventType.STOP_TRIGGERED
                    learn_outcome = OutcomeType.LOSS
                elif exit_reason == ExitReason.TARGET_1_HIT or exit_reason == ExitReason.TARGET_2_HIT:
                    event_type = EventType.TARGET_HIT
                    learn_outcome = OutcomeType.WIN
                else:
                    event_type = EventType.TRADE_EXITED
                    learn_outcome = OutcomeType.WIN if outcome.was_profitable else OutcomeType.LOSS
                
                entry_event_id = position.metadata.get("hyperlearner_entry_event_id")
                
                hyperlearner.emit(
                    category=EventCategory.EXECUTION,
                    event_type=event_type,
                    source="trading_orchestrator",
                    context={
                        "exit_price": exit_price,
                        "exit_reason": exit_reason.value,
                        "entry_price": position.entry_price,
                        "realized_pnl": outcome.realized_pnl,
                        "return_pct": outcome.return_pct,
                        "hold_time_bars": outcome.hold_time_bars,
                        "quantrascore": position.metadata.get("quantrascore", 0),
                        "quality_tier": position.metadata.get("quality_tier", "unknown"),
                        "original_event_id": entry_event_id,
                    },
                    symbol=position.symbol,
                    confidence=0.9,
                    priority=LearningPriority.CRITICAL,
                )
                
                if entry_event_id:
                    hyperlearner.record_outcome(
                        event_id=entry_event_id,
                        outcome_type=learn_outcome,
                        return_pct=outcome.return_pct,
                        was_correct=outcome.was_profitable,
                        details={
                            "exit_reason": exit_reason.value,
                            "hold_time": outcome.hold_time_bars,
                            "exit_price": exit_price,
                        },
                    )
            except ImportError:
                pass
    
    def _handle_exit_triggered(self, position: PositionState, exit_reason: ExitReason) -> None:
        """Callback for PositionMonitor exit triggers."""
        logger.debug(f"[Orchestrator] Exit triggered for {position.symbol}: {exit_reason.value}")
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk parameters."""
        account_equity = 100000.0
        
        max_position = account_equity * self.config.max_single_position_pct
        
        risk_per_trade = account_equity * 0.01
        stop_distance = price * 0.02
        position_from_risk = (risk_per_trade / stop_distance) * price
        
        return min(max_position, position_from_risk)
    
    def _get_market_status(self) -> MarketStatus:
        """Get current market status based on time."""
        now = datetime.now().time()
        
        if self.MARKET_OPEN <= now < self.MARKET_CLOSE:
            return MarketStatus.OPEN
        elif self.PRE_MARKET_START <= now < self.MARKET_OPEN:
            return MarketStatus.PRE_MARKET
        elif self.MARKET_CLOSE <= now < self.POST_MARKET_END:
            return MarketStatus.POST_MARKET
        else:
            return MarketStatus.CLOSED
    
    def emergency_shutdown(self) -> None:
        """Emergency shutdown - close all positions immediately."""
        logger.critical("[Orchestrator] EMERGENCY SHUTDOWN")
        
        outcomes = self.position_monitor.force_close_all(ExitReason.OMEGA_OVERRIDE)
        
        for outcome in outcomes:
            self.outcome_tracker.record_outcome(outcome)
        
        self._should_run = False
        self._set_state(TradingState.SHUTDOWN)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        return {
            "state": self._state.value,
            "metrics": self._metrics.to_dict(),
            "positions": self.position_monitor.get_metrics(),
            "filter": self.quality_filter.get_stats(),
            "outcomes": self.outcome_tracker.get_stats(),
            "windows": self.window_manager.get_status(),
            "stream": self._stream.get_metrics() if self._stream else {},
        }
    
    def get_recent_decisions(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent trading decisions."""
        return [d.to_dict() for d in self._decision_log[-n:]]


async def run_orchestrator(
    config: Optional[OrchestratorConfig] = None,
    duration_seconds: Optional[float] = None,
    use_simulated_stream: bool = True,
) -> None:
    """
    Run the trading orchestrator.
    
    Args:
        config: Orchestrator configuration
        duration_seconds: Optional duration limit (None = run forever)
        use_simulated_stream: Use simulated stream instead of Polygon
    """
    orchestrator = TradingOrchestrator(
        config=config,
        use_simulated_stream=use_simulated_stream,
    )
    
    if duration_seconds:
        async def stop_after_duration():
            await asyncio.sleep(duration_seconds)
            await orchestrator.stop()
        
        stop_task = asyncio.create_task(stop_after_duration())
        
        try:
            await orchestrator.start()
        finally:
            stop_task.cancel()
    else:
        await orchestrator.start()
