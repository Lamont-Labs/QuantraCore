"""
Portfolio Management for QuantraCore Apex.

Provides:
- Position tracking with cost basis
- P&L calculation (realized and unrealized)
- Sector/exposure analysis
- Heat map generation
- Portfolio snapshots for research

All functionality is for research and simulation only.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Position(BaseModel):
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    sector: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis for position."""
        return self.quantity * self.avg_cost
    
    @property
    def is_long(self) -> bool:
        """True if long position."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """True if short position."""
        return self.quantity < 0


class PortfolioSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cash: float
    positions_value: float
    total_equity: float
    total_pnl: float
    total_pnl_pct: float
    num_positions: int
    long_exposure: float
    short_exposure: float
    net_exposure: float
    sector_exposure: Dict[str, float] = Field(default_factory=dict)
    
    compliance_note: str = "Portfolio snapshot for research purposes only"


class Portfolio:
    """
    Portfolio tracker for research and simulation.
    
    Tracks positions, calculates P&L, and provides exposure analysis.
    This is for research/simulation only - not for live trading.
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.snapshots: List[PortfolioSnapshot] = []
        self.sector_map: Dict[str, str] = {}
    
    def set_sector_map(self, sector_map: Dict[str, str]):
        """Set symbol-to-sector mapping for exposure analysis."""
        self.sector_map = sector_map
        for sym, pos in self.positions.items():
            if sym in sector_map:
                pos.sector = sector_map[sym]
    
    def update_position(
        self,
        symbol: str,
        quantity_change: float,
        price: float,
        commission: float = 0.0,
    ) -> Position:
        """
        Update a position after a trade.
        
        Args:
            symbol: Ticker symbol
            quantity_change: Positive for buy, negative for sell
            price: Execution price
            commission: Trading commission
        
        Returns:
            Updated Position object
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                sector=self.sector_map.get(symbol),
            )
        
        pos = self.positions[symbol]
        old_qty = pos.quantity
        new_qty = old_qty + quantity_change
        
        if abs(old_qty) < 0.0001:
            pos.avg_cost = price
        elif (old_qty > 0 and quantity_change > 0) or (old_qty < 0 and quantity_change < 0):
            total_cost = pos.avg_cost * abs(old_qty) + price * abs(quantity_change)
            pos.avg_cost = total_cost / (abs(old_qty) + abs(quantity_change))
        else:
            close_qty = min(abs(quantity_change), abs(old_qty))
            if old_qty > 0:
                realized = close_qty * (price - pos.avg_cost)
            else:
                realized = close_qty * (pos.avg_cost - price)
            
            pos.realized_pnl += realized
            self.realized_pnl += realized
            
            if abs(new_qty) > abs(old_qty) - abs(quantity_change) + 0.0001:
                pos.avg_cost = price
        
        pos.quantity = new_qty
        pos.last_updated = datetime.utcnow()
        
        self.cash -= quantity_change * price + commission
        
        if abs(pos.quantity) < 0.0001:
            del self.positions[symbol]
            return pos
        
        return pos
    
    def update_prices(self, prices: Dict[str, float]):
        """Update market values and unrealized P&L from current prices."""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                pos.market_value = pos.quantity * current_price
                pos.unrealized_pnl = pos.market_value - pos.cost_basis
                
                if abs(pos.cost_basis) > 0.0001:
                    pos.unrealized_pnl_pct = pos.unrealized_pnl / abs(pos.cost_basis) * 100
                else:
                    pos.unrealized_pnl_pct = 0.0
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self.positions.values())
    
    def get_total_equity(self) -> float:
        """Get total portfolio equity (cash + positions)."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.realized_pnl + unrealized
    
    def get_long_exposure(self) -> float:
        """Get total long exposure."""
        return sum(p.market_value for p in self.positions.values() if p.quantity > 0)
    
    def get_short_exposure(self) -> float:
        """Get total short exposure (absolute value)."""
        return abs(sum(p.market_value for p in self.positions.values() if p.quantity < 0))
    
    def get_net_exposure(self) -> float:
        """Get net exposure (long - short)."""
        return sum(p.market_value for p in self.positions.values())
    
    def get_sector_exposure(self) -> Dict[str, float]:
        """Get exposure by sector."""
        sector_exp: Dict[str, float] = {}
        for pos in self.positions.values():
            sector = pos.sector or "Unknown"
            sector_exp[sector] = sector_exp.get(sector, 0.0) + pos.market_value
        return sector_exp
    
    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state."""
        positions_value = sum(p.market_value for p in self.positions.values())
        total_equity = self.cash + positions_value
        total_pnl = total_equity - self.initial_cash
        
        snapshot = PortfolioSnapshot(
            cash=self.cash,
            positions_value=positions_value,
            total_equity=total_equity,
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / self.initial_cash * 100) if self.initial_cash > 0 else 0,
            num_positions=len(self.positions),
            long_exposure=self.get_long_exposure(),
            short_exposure=self.get_short_exposure(),
            net_exposure=self.get_net_exposure(),
            sector_exposure=self.get_sector_exposure(),
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_heat_map(self) -> Dict[str, Dict[str, float]]:
        """
        Generate heat map data for visualization.
        
        Returns:
            Dict with sector -> {symbol: pnl_pct}
        """
        heat_map: Dict[str, Dict[str, float]] = {}
        
        for pos in self.positions.values():
            sector = pos.sector or "Unknown"
            if sector not in heat_map:
                heat_map[sector] = {}
            heat_map[sector][pos.symbol] = pos.unrealized_pnl_pct
        
        return heat_map
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self.realized_pnl = 0.0
        self.snapshots.clear()
