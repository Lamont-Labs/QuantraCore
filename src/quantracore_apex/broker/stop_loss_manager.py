"""
Stop Loss Manager for Moonshot Positions

Provides automatic stop-loss management with three protection layers:
1. Hard Stop: -15% from entry (configurable)
2. Trailing Stop: Follows price up, locks in gains
3. Time Stop: Exit after N days if no breakout

Designed for moonshot strategy targeting 50%+ gains in 5 trading days.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class StopType(str, Enum):
    """Type of stop that was triggered."""
    HARD = "hard"           # Hit max loss threshold
    TRAILING = "trailing"   # Trailing stop triggered
    TIME = "time"           # Time limit reached without breakout
    MANUAL = "manual"       # Manually closed


class ExitSignal(str, Enum):
    """Exit signal status."""
    HOLD = "hold"           # No exit signal
    EXIT = "exit"           # Exit immediately
    WARNING = "warning"     # Approaching stop level


@dataclass
class StopConfig:
    """Configuration for stop-loss rules."""
    hard_stop_pct: float = 0.15          # -15% hard stop
    trailing_activation_pct: float = 0.10 # Activate trailing at +10%
    trailing_distance_pct: float = 0.08   # Trail 8% below high
    time_limit_days: int = 5              # Exit after 5 days if flat
    time_stop_min_gain_pct: float = 0.05  # Need +5% to avoid time stop
    breakout_target_pct: float = 0.50     # 50% moonshot target
    
    def to_dict(self) -> dict:
        return {
            "hard_stop_pct": self.hard_stop_pct,
            "trailing_activation_pct": self.trailing_activation_pct,
            "trailing_distance_pct": self.trailing_distance_pct,
            "time_limit_days": self.time_limit_days,
            "time_stop_min_gain_pct": self.time_stop_min_gain_pct,
            "breakout_target_pct": self.breakout_target_pct,
        }


@dataclass
class PositionStop:
    """Stop-loss tracking for a single position."""
    symbol: str
    entry_price: float
    entry_date: date
    quantity: float
    
    hard_stop_price: float = 0.0
    trailing_stop_price: float = 0.0
    trailing_active: bool = False
    highest_price: float = 0.0
    current_price: float = 0.0
    
    days_held: int = 0
    current_pnl_pct: float = 0.0
    max_pnl_pct: float = 0.0
    
    exit_signal: ExitSignal = ExitSignal.HOLD
    stop_type: Optional[StopType] = None
    stop_reason: str = ""
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date.isoformat(),
            "quantity": self.quantity,
            "hard_stop_price": round(self.hard_stop_price, 4),
            "trailing_stop_price": round(self.trailing_stop_price, 4),
            "trailing_active": self.trailing_active,
            "highest_price": round(self.highest_price, 4),
            "current_price": round(self.current_price, 4),
            "days_held": self.days_held,
            "current_pnl_pct": round(self.current_pnl_pct * 100, 2),
            "max_pnl_pct": round(self.max_pnl_pct * 100, 2),
            "exit_signal": self.exit_signal.value,
            "stop_type": self.stop_type.value if self.stop_type else None,
            "stop_reason": self.stop_reason,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class StopCheckResult:
    """Result of checking all stops for a position."""
    symbol: str
    exit_signal: ExitSignal
    stop_type: Optional[StopType]
    reason: str
    current_price: float
    stop_price: float
    pnl_pct: float
    days_held: int
    distance_to_stop_pct: float  # How close to stop (negative = below stop)
    distance_to_target_pct: float  # How far from moonshot target
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "exit_signal": self.exit_signal.value,
            "stop_type": self.stop_type.value if self.stop_type else None,
            "reason": self.reason,
            "current_price": round(self.current_price, 4),
            "stop_price": round(self.stop_price, 4),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "days_held": self.days_held,
            "distance_to_stop_pct": round(self.distance_to_stop_pct * 100, 2),
            "distance_to_target_pct": round(self.distance_to_target_pct * 100, 2),
        }


class StopLossManager:
    """
    Automatic Stop-Loss Manager for Moonshot Positions.
    
    Features:
    - Hard stop: Fixed percentage below entry (default -15%)
    - Trailing stop: Activates after +10%, trails 8% below highest
    - Time stop: Exit after 5 days if gain < 5%
    
    Usage:
        manager = StopLossManager()
        manager.register_position("PMAX", entry_price=2.86, entry_date=date.today(), qty=100)
        result = manager.check_stops("PMAX", current_price=2.50)
        if result.exit_signal == ExitSignal.EXIT:
            # Execute exit
    """
    
    def __init__(self, config: Optional[StopConfig] = None):
        self._config = config or StopConfig()
        self._positions: Dict[str, PositionStop] = {}
        self._exit_history: List[Dict] = []
        logger.info(f"[StopLossManager] Initialized with config: {self._config.to_dict()}")
    
    @property
    def config(self) -> StopConfig:
        return self._config
    
    def update_config(self, **kwargs) -> StopConfig:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        logger.info(f"[StopLossManager] Config updated: {self._config.to_dict()}")
        return self._config
    
    def register_position(
        self,
        symbol: str,
        entry_price: float,
        entry_date: date,
        quantity: float,
    ) -> PositionStop:
        """
        Register a new position with automatic stop levels.
        
        Args:
            symbol: Stock ticker
            entry_price: Average entry price
            entry_date: Date position was opened
            quantity: Number of shares
            
        Returns:
            PositionStop with calculated stop levels
        """
        symbol = symbol.upper()
        
        hard_stop = entry_price * (1 - self._config.hard_stop_pct)
        
        position = PositionStop(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=entry_date,
            quantity=quantity,
            hard_stop_price=hard_stop,
            trailing_stop_price=0.0,  # Not active until threshold hit
            trailing_active=False,
            highest_price=entry_price,
            current_price=entry_price,
            days_held=0,
        )
        
        self._positions[symbol] = position
        logger.info(
            f"[StopLossManager] Registered {symbol}: entry=${entry_price:.4f}, "
            f"hard_stop=${hard_stop:.4f} (-{self._config.hard_stop_pct*100:.0f}%)"
        )
        
        return position
    
    def sync_from_broker(
        self,
        positions: List[Dict],
        current_date: Optional[date] = None,
    ) -> List[PositionStop]:
        """
        Sync positions from broker data.
        
        Args:
            positions: List of position dicts with symbol, avg_price, qty, current_price
            current_date: Current trading date (defaults to today)
            
        Returns:
            List of registered PositionStop objects
        """
        current_date = current_date or date.today()
        result = []
        
        for pos in positions:
            symbol = pos.get("symbol", "").upper()
            if not symbol:
                continue
            
            entry_price = float(pos.get("avg_entry_price", 0) or pos.get("avg_price", 0))
            quantity = float(pos.get("qty", 0) or pos.get("quantity", 0))
            current_price = float(pos.get("current_price", entry_price))
            
            if entry_price <= 0 or quantity <= 0:
                continue
            
            if symbol in self._positions:
                ps = self._positions[symbol]
                ps.current_price = current_price
                ps.quantity = quantity
                ps = self._update_position_state(ps, current_price, current_date)
            else:
                entry_date = current_date - timedelta(days=3)
                ps = self.register_position(symbol, entry_price, entry_date, quantity)
                ps.current_price = current_price
                ps = self._update_position_state(ps, current_price, current_date)
            
            result.append(ps)
        
        return result
    
    def _update_position_state(
        self,
        position: PositionStop,
        current_price: float,
        current_date: date,
    ) -> PositionStop:
        """Update position state with current price and date."""
        position.current_price = current_price
        position.updated_at = datetime.utcnow()
        
        position.days_held = (current_date - position.entry_date).days
        
        if position.entry_price > 0:
            position.current_pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        if current_price > position.highest_price:
            position.highest_price = current_price
            position.max_pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        if (position.current_pnl_pct >= self._config.trailing_activation_pct 
            and not position.trailing_active):
            position.trailing_active = True
            position.trailing_stop_price = current_price * (1 - self._config.trailing_distance_pct)
            logger.info(
                f"[StopLossManager] {position.symbol}: Trailing stop ACTIVATED at "
                f"${position.trailing_stop_price:.4f} (gain: {position.current_pnl_pct*100:.1f}%)"
            )
        
        if position.trailing_active and current_price > position.highest_price:
            new_trail = current_price * (1 - self._config.trailing_distance_pct)
            if new_trail > position.trailing_stop_price:
                position.trailing_stop_price = new_trail
                logger.debug(
                    f"[StopLossManager] {position.symbol}: Trailing stop raised to "
                    f"${position.trailing_stop_price:.4f}"
                )
        
        return position
    
    def check_stops(
        self,
        symbol: str,
        current_price: float,
        current_date: Optional[date] = None,
    ) -> StopCheckResult:
        """
        Check all stop conditions for a position.
        
        Args:
            symbol: Stock ticker
            current_price: Current market price
            current_date: Current trading date
            
        Returns:
            StopCheckResult with exit signal and details
        """
        symbol = symbol.upper()
        current_date = current_date or date.today()
        
        if symbol not in self._positions:
            return StopCheckResult(
                symbol=symbol,
                exit_signal=ExitSignal.HOLD,
                stop_type=None,
                reason="Position not registered",
                current_price=current_price,
                stop_price=0,
                pnl_pct=0,
                days_held=0,
                distance_to_stop_pct=0,
                distance_to_target_pct=0,
            )
        
        position = self._update_position_state(
            self._positions[symbol], current_price, current_date
        )
        
        active_stop = position.hard_stop_price
        if position.trailing_active and position.trailing_stop_price > position.hard_stop_price:
            active_stop = position.trailing_stop_price
        
        distance_to_stop = (current_price - active_stop) / active_stop if active_stop > 0 else 0
        target_price = position.entry_price * (1 + self._config.breakout_target_pct)
        distance_to_target = (target_price - current_price) / current_price if current_price > 0 else 0
        
        exit_signal = ExitSignal.HOLD
        stop_type = None
        reason = ""
        
        if current_price <= position.hard_stop_price:
            exit_signal = ExitSignal.EXIT
            stop_type = StopType.HARD
            reason = f"Hard stop hit: ${current_price:.4f} <= ${position.hard_stop_price:.4f} (-{self._config.hard_stop_pct*100:.0f}%)"
        
        elif position.trailing_active and current_price <= position.trailing_stop_price:
            exit_signal = ExitSignal.EXIT
            stop_type = StopType.TRAILING
            reason = f"Trailing stop hit: ${current_price:.4f} <= ${position.trailing_stop_price:.4f} (locked in gains)"
        
        elif (position.days_held >= self._config.time_limit_days 
              and position.current_pnl_pct < self._config.time_stop_min_gain_pct):
            exit_signal = ExitSignal.EXIT
            stop_type = StopType.TIME
            reason = f"Time stop: Day {position.days_held}, gain only {position.current_pnl_pct*100:.1f}% < {self._config.time_stop_min_gain_pct*100:.0f}% threshold"
        
        elif distance_to_stop < 0.03:  # Within 3% of stop
            exit_signal = ExitSignal.WARNING
            reason = f"WARNING: Only {distance_to_stop*100:.1f}% above stop level"
        
        position.exit_signal = exit_signal
        position.stop_type = stop_type
        position.stop_reason = reason
        
        return StopCheckResult(
            symbol=symbol,
            exit_signal=exit_signal,
            stop_type=stop_type,
            reason=reason,
            current_price=current_price,
            stop_price=active_stop,
            pnl_pct=position.current_pnl_pct,
            days_held=position.days_held,
            distance_to_stop_pct=distance_to_stop,
            distance_to_target_pct=distance_to_target,
        )
    
    def check_all_positions(
        self,
        prices: Dict[str, float],
        current_date: Optional[date] = None,
    ) -> List[StopCheckResult]:
        """
        Check stops for all registered positions.
        
        Args:
            prices: Dict mapping symbol to current price
            current_date: Current trading date
            
        Returns:
            List of StopCheckResult for each position
        """
        results = []
        for symbol in self._positions:
            price = prices.get(symbol.upper(), prices.get(symbol.lower(), 0))
            if price > 0:
                result = self.check_stops(symbol, price, current_date)
                results.append(result)
        
        results.sort(key=lambda r: (
            0 if r.exit_signal == ExitSignal.EXIT else (1 if r.exit_signal == ExitSignal.WARNING else 2),
            r.distance_to_stop_pct
        ))
        
        return results
    
    def get_position(self, symbol: str) -> Optional[PositionStop]:
        """Get position stop data by symbol."""
        return self._positions.get(symbol.upper())
    
    def get_all_positions(self) -> List[PositionStop]:
        """Get all registered positions."""
        return list(self._positions.values())
    
    def remove_position(self, symbol: str, reason: str = "closed") -> Optional[PositionStop]:
        """Remove a position after exit."""
        symbol = symbol.upper()
        if symbol in self._positions:
            position = self._positions.pop(symbol)
            self._exit_history.append({
                "symbol": symbol,
                "entry_price": position.entry_price,
                "exit_price": position.current_price,
                "pnl_pct": position.current_pnl_pct,
                "days_held": position.days_held,
                "stop_type": position.stop_type.value if position.stop_type else None,
                "reason": reason,
                "exited_at": datetime.utcnow().isoformat(),
            })
            logger.info(f"[StopLossManager] Removed {symbol}: {reason}")
            return position
        return None
    
    def get_exit_history(self) -> List[Dict]:
        """Get history of exited positions."""
        return self._exit_history.copy()
    
    def get_summary(self) -> Dict:
        """Get summary of all stop-loss status."""
        positions = self.get_all_positions()
        
        exit_signals = sum(1 for p in positions if p.exit_signal == ExitSignal.EXIT)
        warnings = sum(1 for p in positions if p.exit_signal == ExitSignal.WARNING)
        holding = sum(1 for p in positions if p.exit_signal == ExitSignal.HOLD)
        trailing_active = sum(1 for p in positions if p.trailing_active)
        
        return {
            "total_positions": len(positions),
            "exit_signals": exit_signals,
            "warnings": warnings,
            "holding": holding,
            "trailing_stops_active": trailing_active,
            "config": self._config.to_dict(),
            "positions": [p.to_dict() for p in positions],
        }


_stop_loss_manager: Optional[StopLossManager] = None


def get_stop_loss_manager() -> StopLossManager:
    """Get or create the global StopLossManager instance."""
    global _stop_loss_manager
    if _stop_loss_manager is None:
        _stop_loss_manager = StopLossManager()
    return _stop_loss_manager
