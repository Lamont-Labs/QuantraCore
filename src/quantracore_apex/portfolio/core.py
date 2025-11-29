"""
Portfolio Engine for QuantraCore Apex.

Institutional-grade portfolio management with risk-based position sizing,
universe management, and NAV tracking. Research mode only.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

EQUITY_UNIVERSE = [
    "AAPL", "NVDA", "TSLA", "AMD", "SMCI", "ARM", "META", "GOOGL", "MSFT", "AMZN",
    "SPY", "QQQ", "IWM", "TQQQ", "SQQQ"
]

CRYPTO_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"
]

FULL_UNIVERSE = EQUITY_UNIVERSE + CRYPTO_UNIVERSE


@dataclass
class Position:
    """Represents a portfolio position."""
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost * 100


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""
    timestamp: datetime
    cash: float
    positions_value: float
    nav: float
    positions: Dict[str, Position]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash": self.cash,
            "positions_value": self.positions_value,
            "nav": self.nav,
            "position_count": len([p for p in self.positions.values() if p.quantity > 0])
        }


class Portfolio:
    """
    Institutional-grade portfolio engine with risk-based sizing.
    
    Features:
    - Volatility-adjusted position sizing
    - QuantraScore-based allocation
    - Omega directive compliance (no positions on alerts)
    - NAV tracking with full history
    
    Note: Research mode only - all trades are simulated.
    """
    
    def __init__(self, initial_cash: float = 1_000_000.0, universe: Optional[List[str]] = None):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash balance
            universe: List of tradeable symbols
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.universe = universe or FULL_UNIVERSE
        
        self.positions: Dict[str, Position] = {
            symbol: Position(symbol=symbol) for symbol in self.universe
        }
        
        self.history: List[PortfolioSnapshot] = []
        self.trade_log: List[Dict[str, Any]] = []
        
        logger.info(f"Portfolio initialized: ${initial_cash:,.2f} | Universe: {len(self.universe)} symbols")
    
    @property
    def positions_value(self) -> float:
        """Total market value of all positions."""
        return sum(p.market_value for p in self.positions.values())
    
    @property
    def nav(self) -> float:
        """Net asset value."""
        return self.cash + self.positions_value
    
    @property
    def total_return(self) -> float:
        """Total return percentage."""
        return (self.nav - self.initial_cash) / self.initial_cash * 100
    
    def update_price(self, symbol: str, price: float):
        """
        Update current price for a symbol.
        
        Args:
            symbol: Symbol to update
            price: Current price
        """
        if symbol in self.positions:
            self.positions[symbol].current_price = price
            self.positions[symbol].last_update = datetime.now()
    
    def rebalance(
        self,
        signals: Dict[str, Dict[str, Any]],
        max_position_risk: float = 0.02,
        min_score: int = 65
    ):
        """
        Rebalance portfolio based on signals and risk parameters.
        
        Args:
            signals: Dict of symbol -> signal data with quantra_score, atr_pct, omega_alert, close
            max_position_risk: Maximum risk per position (default 2%)
            min_score: Minimum QuantraScore to take position (default 65)
        """
        for symbol, signal in signals.items():
            if symbol not in self.positions:
                continue
            
            score = signal.get('quantra_score', 0)
            omega_alert = signal.get('omega_alert', False)
            close_price = signal.get('close', 0)
            atr_pct = signal.get('atr_pct', 0.02)
            
            self.update_price(symbol, close_price)
            
            if score < min_score or omega_alert or close_price <= 0:
                if self.positions[symbol].quantity > 0:
                    self._close_position(symbol, close_price, "Signal below threshold")
                continue
            
            normalized_score = score / 100
            volatility_adj = max(atr_pct, 0.005)
            target_weight = normalized_score * max_position_risk / volatility_adj
            target_weight = min(target_weight, 0.10)
            
            target_dollars = self.nav * target_weight
            target_quantity = target_dollars / close_price
            
            current_position = self.positions[symbol]
            
            if abs(target_quantity - current_position.quantity) > 0.01:
                self._adjust_position(symbol, target_quantity, close_price, score)
        
        self._record_snapshot()
    
    def _adjust_position(self, symbol: str, target_qty: float, price: float, score: int):
        """Adjust position to target quantity."""
        current_pos = self.positions[symbol]
        delta = target_qty - current_pos.quantity
        
        if delta > 0:
            cost = delta * price
            if cost <= self.cash:
                self.cash -= cost
                
                if current_pos.quantity == 0:
                    current_pos.avg_cost = price
                else:
                    total_cost = (current_pos.quantity * current_pos.avg_cost) + (delta * price)
                    current_pos.avg_cost = total_cost / (current_pos.quantity + delta)
                
                current_pos.quantity = target_qty
                current_pos.current_price = price
                
                self.trade_log.append({
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": delta,
                    "price": price,
                    "score": score
                })
                logger.info(f"BUY {symbol}: +{delta:.4f} @ ${price:.2f} (Score: {score})")
        
        elif delta < 0:
            sell_qty = min(abs(delta), current_pos.quantity)
            proceeds = sell_qty * price
            self.cash += proceeds
            current_pos.quantity -= sell_qty
            
            self.trade_log.append({
                "timestamp": datetime.now(),
                "symbol": symbol,
                "action": "SELL",
                "quantity": sell_qty,
                "price": price,
                "score": score
            })
            logger.info(f"SELL {symbol}: -{sell_qty:.4f} @ ${price:.2f} (Score: {score})")
    
    def _close_position(self, symbol: str, price: float, reason: str):
        """Close entire position."""
        position = self.positions[symbol]
        if position.quantity <= 0:
            return
        
        proceeds = position.quantity * price
        self.cash += proceeds
        
        self.trade_log.append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "action": "CLOSE",
            "quantity": position.quantity,
            "price": price,
            "reason": reason
        })
        
        logger.info(f"CLOSE {symbol}: {position.quantity:.4f} @ ${price:.2f} ({reason})")
        position.quantity = 0
        position.avg_cost = 0
    
    def _record_snapshot(self):
        """Record current portfolio state."""
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=self.cash,
            positions_value=self.positions_value,
            nav=self.nav,
            positions={k: Position(**v.__dict__) for k, v in self.positions.items()}
        )
        self.history.append(snapshot)
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get NAV history as DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=['date', 'nav', 'cash', 'positions_value'])
        
        data = [{
            'date': s.timestamp,
            'nav': s.nav,
            'cash': s.cash,
            'positions_value': s.positions_value
        } for s in self.history]
        
        return pd.DataFrame(data)
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get list of active positions."""
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct
            }
            for p in self.positions.values()
            if p.quantity > 0
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        active = self.get_active_positions()
        
        return {
            "nav": self.nav,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "total_return_pct": self.total_return,
            "active_positions": len(active),
            "universe_size": len(self.universe),
            "trade_count": len(self.trade_log)
        }
    
    def save_equity_curve(self, filepath: str = "equity_curve.csv"):
        """Save equity curve to CSV."""
        df = self.get_equity_curve()
        df.to_csv(filepath, index=False)
        logger.info(f"Equity curve saved to {filepath}")
