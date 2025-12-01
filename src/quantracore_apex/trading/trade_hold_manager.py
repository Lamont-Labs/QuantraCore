"""
Trade Hold Manager with Continuation Probability

Manages active position hold times based on real-time continuation probability.
Extends trades when momentum suggests the move will continue, exits when
exhaustion or reversal signals are detected.

LEGAL NOTICE:
- This module operates EXCLUSIVELY on Alpaca PAPER trading accounts
- All outputs are for RESEARCH and EDUCATIONAL purposes only
- This is NOT financial advice and should not be treated as such
- Past performance does NOT guarantee future results

Version: 1.0
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from ..prediction.continuation_estimator import estimate_continuation, ContinuationEstimate
from ..core.schemas import OhlcvBar

logger = logging.getLogger(__name__)

_alpaca_client_cache: Dict[str, Any] = {}

def get_cached_alpaca_client():
    """Get or create cached Alpaca client for performance."""
    global _alpaca_client_cache
    
    if "stock_client" not in _alpaca_client_cache:
        api_key = os.environ.get("ALPACA_PAPER_API_KEY")
        api_secret = os.environ.get("ALPACA_PAPER_API_SECRET")
        
        if api_key and api_secret:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                _alpaca_client_cache["stock_client"] = StockHistoricalDataClient(api_key, api_secret)
                logger.info("[TradeHoldManager] Alpaca client cached")
            except Exception as e:
                logger.warning(f"[TradeHoldManager] Could not create Alpaca client: {e}")
                return None
    
    return _alpaca_client_cache.get("stock_client")


def fetch_bars_for_symbol(symbol: str, timeframe: str = "15Min", limit: int = 50) -> List[OhlcvBar]:
    """Fetch OHLCV bars from Alpaca and convert to OhlcvBar objects."""
    client = get_cached_alpaca_client()
    if not client:
        return []
    
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        
        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        
        tf = tf_map.get(timeframe, TimeFrame(15, TimeFrameUnit.Minute))
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            limit=limit,
        )
        
        bars_response = client.get_stock_bars(request)
        
        bars_data = []
        if hasattr(bars_response, 'data') and symbol in bars_response.data:
            bars_data = bars_response.data[symbol]
        elif hasattr(bars_response, '__iter__'):
            try:
                bars_data = list(bars_response[symbol]) if symbol in bars_response else []
            except (TypeError, KeyError):
                bars_data = []
        
        ohlcv_bars = []
        for bar in bars_data:
            ohlcv_bars.append(OhlcvBar(
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
                timestamp=bar.timestamp.isoformat() if hasattr(bar, 'timestamp') else datetime.utcnow().isoformat(),
            ))
        
        return ohlcv_bars
    except Exception as e:
        logger.error(f"[TradeHoldManager] Error fetching bars for {symbol}: {e}")
        return []


class HoldDecision(str, Enum):
    """Trade hold decision types."""
    HOLD_STRONG = "hold_strong"
    HOLD_NORMAL = "hold_normal"
    REDUCE_PARTIAL = "reduce_partial"
    EXIT_NOW = "exit_now"
    TRAIL_STOP = "trail_stop"


@dataclass
class PositionHoldStatus:
    """Status of an active position's hold decision."""
    symbol: str
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    continuation_probability: float
    reversal_probability: float
    trend_strength: float
    momentum_status: str
    exhaustion_level: float
    confidence: float
    
    hold_decision: HoldDecision
    hold_reason: str
    suggested_action: str
    
    adjusted_stop: Optional[float] = None
    adjusted_target: Optional[float] = None
    hold_extension_bars: int = 0
    
    last_update: str = ""
    compliance_note: str = "Research analysis only - not trading advice"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "continuation": {
                "probability": self.continuation_probability,
                "reversal_probability": self.reversal_probability,
                "trend_strength": self.trend_strength,
                "momentum_status": self.momentum_status,
                "exhaustion_level": self.exhaustion_level,
                "confidence": self.confidence,
            },
            "decision": {
                "hold_decision": self.hold_decision.value,
                "hold_reason": self.hold_reason,
                "suggested_action": self.suggested_action,
                "adjusted_stop": self.adjusted_stop,
                "adjusted_target": self.adjusted_target,
                "hold_extension_bars": self.hold_extension_bars,
            },
            "last_update": self.last_update,
            "compliance_note": self.compliance_note,
        }


@dataclass
class TradeHoldConfig:
    """Configuration for trade hold decisions."""
    strong_hold_threshold: float = 0.70
    normal_hold_threshold: float = 0.55
    reduce_threshold: float = 0.40
    exit_threshold: float = 0.30
    
    trail_on_high_continuation: bool = True
    trail_atr_multiplier: float = 1.5
    
    extend_target_on_continuation: bool = True
    target_extension_pct: float = 0.50
    
    max_hold_extension_bars: int = 20
    
    check_interval_seconds: int = 60


class TradeHoldManager:
    """
    Manages trade hold times based on continuation probability.
    
    Features:
    - Real-time continuation probability monitoring
    - Dynamic hold/exit decisions
    - Stop loss adjustment based on momentum
    - Target extension for strong trends
    - Exhaustion and reversal detection
    
    PAPER TRADING ONLY - No real money at risk
    """
    
    def __init__(self, config: Optional[TradeHoldConfig] = None):
        self.config = config or TradeHoldConfig()
        self._position_cache: Dict[str, PositionHoldStatus] = {}
        
        logger.info("[TradeHoldManager] Initialized with continuation-based hold logic")
    
    def analyze_position(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        quantity: float,
        original_stop: Optional[float] = None,
        original_target: Optional[float] = None,
    ) -> PositionHoldStatus:
        """
        Analyze a position and determine hold/exit decision.
        
        Args:
            symbol: Stock symbol
            entry_price: Position entry price
            current_price: Current market price
            quantity: Position size
            original_stop: Original stop loss price
            original_target: Original target price
            
        Returns:
            PositionHoldStatus with continuation analysis and decision
        """
        try:
            bars = fetch_bars_for_symbol(symbol, timeframe="15Min", limit=50)
            
            if not bars or len(bars) < 20:
                logger.warning(f"[TradeHoldManager] Insufficient bars for {symbol}")
                return self._create_default_status(
                    symbol, entry_price, current_price, quantity
                )
            
            continuation = estimate_continuation(bars, lookback=20)
            
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            hold_decision, hold_reason, suggested_action = self._make_hold_decision(
                continuation, unrealized_pnl_pct, current_price, entry_price
            )
            
            adjusted_stop = self._calculate_adjusted_stop(
                continuation, current_price, entry_price, original_stop, bars
            )
            
            adjusted_target = self._calculate_adjusted_target(
                continuation, current_price, entry_price, original_target
            )
            
            hold_extension = self._calculate_hold_extension(continuation)
            
            status = PositionHoldStatus(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=round(unrealized_pnl, 2),
                unrealized_pnl_pct=round(unrealized_pnl_pct, 2),
                continuation_probability=continuation.continuation_probability,
                reversal_probability=continuation.reversal_probability,
                trend_strength=continuation.trend_strength,
                momentum_status=continuation.momentum_status,
                exhaustion_level=continuation.exhaustion_level,
                confidence=continuation.confidence,
                hold_decision=hold_decision,
                hold_reason=hold_reason,
                suggested_action=suggested_action,
                adjusted_stop=adjusted_stop,
                adjusted_target=adjusted_target,
                hold_extension_bars=hold_extension,
                last_update=datetime.utcnow().isoformat(),
            )
            
            self._position_cache[symbol] = status
            
            logger.info(
                f"[TradeHoldManager] {symbol}: {hold_decision.value} "
                f"(cont={continuation.continuation_probability:.1%}, "
                f"momentum={continuation.momentum_status})"
            )
            
            return status
            
        except Exception as e:
            logger.error(f"[TradeHoldManager] Error analyzing {symbol}: {e}")
            return self._create_default_status(
                symbol, entry_price, current_price, quantity
            )
    
    def _make_hold_decision(
        self,
        continuation: ContinuationEstimate,
        pnl_pct: float,
        current_price: float,
        entry_price: float,
    ) -> tuple:
        """Make hold/exit decision based on continuation metrics."""
        cont_prob = continuation.continuation_probability
        exhaustion = continuation.exhaustion_level
        momentum = continuation.momentum_status
        
        if cont_prob >= self.config.strong_hold_threshold:
            if momentum == "healthy":
                return (
                    HoldDecision.HOLD_STRONG,
                    f"Strong continuation ({cont_prob:.0%}) with healthy momentum",
                    "Hold position, consider extending target"
                )
            else:
                return (
                    HoldDecision.TRAIL_STOP,
                    f"Strong continuation ({cont_prob:.0%}) but momentum {momentum}",
                    "Hold with trailing stop to protect gains"
                )
        
        elif cont_prob >= self.config.normal_hold_threshold:
            if exhaustion > 0.5:
                return (
                    HoldDecision.TRAIL_STOP,
                    f"Normal continuation ({cont_prob:.0%}) but high exhaustion ({exhaustion:.0%})",
                    "Tighten stop to protect gains"
                )
            else:
                return (
                    HoldDecision.HOLD_NORMAL,
                    f"Adequate continuation ({cont_prob:.0%})",
                    "Hold position per original plan"
                )
        
        elif cont_prob >= self.config.reduce_threshold:
            if pnl_pct > 2.0:
                return (
                    HoldDecision.REDUCE_PARTIAL,
                    f"Weakening continuation ({cont_prob:.0%}) with profits",
                    "Consider reducing position by 50%"
                )
            else:
                return (
                    HoldDecision.TRAIL_STOP,
                    f"Weakening continuation ({cont_prob:.0%})",
                    "Tighten stop significantly"
                )
        
        else:
            if momentum == "diverging" or exhaustion > 0.7:
                return (
                    HoldDecision.EXIT_NOW,
                    f"Low continuation ({cont_prob:.0%}) with exhaustion/divergence",
                    "Exit position to preserve capital"
                )
            else:
                return (
                    HoldDecision.REDUCE_PARTIAL,
                    f"Low continuation ({cont_prob:.0%})",
                    "Reduce exposure significantly"
                )
    
    def _calculate_adjusted_stop(
        self,
        continuation: ContinuationEstimate,
        current_price: float,
        entry_price: float,
        original_stop: Optional[float],
        bars: list,
    ) -> Optional[float]:
        """Calculate adjusted stop loss based on continuation."""
        if not self.config.trail_on_high_continuation:
            return original_stop
        
        if not bars or len(bars) < 14:
            return original_stop
        
        try:
            import numpy as np
            highs = np.array([b.high for b in bars[-14:]])
            lows = np.array([b.low for b in bars[-14:]])
            closes = np.array([b.close for b in bars[-14:]])
            
            tr = np.maximum(
                highs - lows,
                np.maximum(
                    np.abs(highs - np.roll(closes, 1)),
                    np.abs(lows - np.roll(closes, 1))
                )
            )[1:]
            atr = float(np.mean(tr))
        except:
            return original_stop
        
        cont_prob = continuation.continuation_probability
        
        if cont_prob >= self.config.strong_hold_threshold:
            trail_distance = atr * self.config.trail_atr_multiplier
        elif cont_prob >= self.config.normal_hold_threshold:
            trail_distance = atr * (self.config.trail_atr_multiplier * 0.8)
        else:
            trail_distance = atr * (self.config.trail_atr_multiplier * 0.5)
        
        trailing_stop = current_price - trail_distance
        
        if original_stop:
            adjusted_stop = max(trailing_stop, original_stop)
            
            if current_price > entry_price * 1.02:
                breakeven_stop = entry_price * 1.001
                adjusted_stop = max(adjusted_stop, breakeven_stop)
        else:
            adjusted_stop = trailing_stop
        
        return round(adjusted_stop, 2)
    
    def _calculate_adjusted_target(
        self,
        continuation: ContinuationEstimate,
        current_price: float,
        entry_price: float,
        original_target: Optional[float],
    ) -> Optional[float]:
        """Calculate adjusted target based on continuation strength."""
        if not self.config.extend_target_on_continuation:
            return original_target
        
        if not original_target:
            return None
        
        cont_prob = continuation.continuation_probability
        
        if cont_prob >= self.config.strong_hold_threshold and continuation.momentum_status == "healthy":
            original_move = original_target - entry_price
            extension = original_move * self.config.target_extension_pct
            adjusted_target = original_target + extension
            return round(adjusted_target, 2)
        
        return original_target
    
    def _calculate_hold_extension(self, continuation: ContinuationEstimate) -> int:
        """Calculate how many bars to extend the hold based on continuation."""
        cont_prob = continuation.continuation_probability
        
        if cont_prob >= self.config.strong_hold_threshold:
            extension = min(
                int((cont_prob - 0.5) * 40),
                self.config.max_hold_extension_bars
            )
            return extension
        
        return 0
    
    def _create_default_status(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        quantity: float,
    ) -> PositionHoldStatus:
        """Create default status when analysis fails."""
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        return PositionHoldStatus(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=round(unrealized_pnl, 2),
            unrealized_pnl_pct=round(unrealized_pnl_pct, 2),
            continuation_probability=0.5,
            reversal_probability=0.5,
            trend_strength=0.0,
            momentum_status="unknown",
            exhaustion_level=0.0,
            confidence=0.0,
            hold_decision=HoldDecision.HOLD_NORMAL,
            hold_reason="Insufficient data for analysis",
            suggested_action="Hold per original plan",
            last_update=datetime.utcnow().isoformat(),
        )
    
    def analyze_all_positions(
        self,
        positions: List[Dict[str, Any]],
    ) -> Dict[str, PositionHoldStatus]:
        """
        Analyze all active positions.
        
        Args:
            positions: List of position dicts with symbol, entry_price, current_price, qty
            
        Returns:
            Dict of symbol -> PositionHoldStatus
        """
        results = {}
        
        for pos in positions:
            symbol = pos.get("symbol")
            if not symbol:
                continue
            
            status = self.analyze_position(
                symbol=symbol,
                entry_price=pos.get("entry_price", pos.get("avg_entry_price", 0)),
                current_price=pos.get("current_price", pos.get("market_value", 0) / max(pos.get("qty", 1), 1)),
                quantity=pos.get("qty", pos.get("quantity", 0)),
                original_stop=pos.get("stop_loss"),
                original_target=pos.get("target"),
            )
            results[symbol] = status
        
        return results
    
    def get_cached_status(self, symbol: str) -> Optional[PositionHoldStatus]:
        """Get cached position status."""
        return self._position_cache.get(symbol)
    
    def get_all_cached(self) -> Dict[str, PositionHoldStatus]:
        """Get all cached position statuses."""
        return self._position_cache.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all position hold decisions."""
        positions = list(self._position_cache.values())
        
        if not positions:
            return {
                "total_positions": 0,
                "decisions": {},
                "avg_continuation": 0,
                "positions_at_risk": 0,
                "compliance_note": "Research analysis only - not trading advice",
            }
        
        decisions = {}
        for d in HoldDecision:
            count = len([p for p in positions if p.hold_decision == d])
            if count > 0:
                decisions[d.value] = count
        
        avg_continuation = sum(p.continuation_probability for p in positions) / len(positions)
        at_risk = len([p for p in positions if p.continuation_probability < self.config.reduce_threshold])
        
        return {
            "total_positions": len(positions),
            "decisions": decisions,
            "avg_continuation": round(avg_continuation, 3),
            "positions_at_risk": at_risk,
            "last_update": datetime.utcnow().isoformat(),
            "compliance_note": "Research analysis only - not trading advice",
        }


_hold_manager_instance: Optional[TradeHoldManager] = None


def get_hold_manager() -> TradeHoldManager:
    """Get singleton TradeHoldManager instance."""
    global _hold_manager_instance
    if _hold_manager_instance is None:
        _hold_manager_instance = TradeHoldManager()
    return _hold_manager_instance
