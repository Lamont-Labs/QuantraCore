"""
Risk Engine for QuantraCore Apex Broker Layer.

Deterministic risk checks that must approve every order before execution.
Fail-closed design: if any check fails, order is rejected.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, date

from .config import RiskConfig
from .models import OrderTicket, BrokerPosition, RiskDecision
from .enums import OrderSide, RiskDecisionType


logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Deterministic Risk Engine.
    
    Every order must pass through the risk engine before execution.
    If ANY check fails, the order is REJECTED.
    
    Checks:
    - max_notional_exposure_usd
    - max_position_notional_per_symbol_usd
    - max_positions
    - max_daily_turnover_usd
    - max_order_notional_usd
    - max_leverage
    - block_short_selling
    - block_margin
    - require_positive_equity
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self._config = config or RiskConfig()
        self._daily_turnover: Dict[date, float] = {}
    
    @property
    def config(self) -> RiskConfig:
        return self._config
    
    def check(
        self,
        order: OrderTicket,
        positions: List[BrokerPosition],
        equity: float,
        last_price: Optional[float] = None,
    ) -> RiskDecision:
        """
        Run all risk checks on an order.
        
        Args:
            order: The order to check
            positions: Current positions
            equity: Current account equity
            last_price: Last known price for the symbol (optional)
            
        Returns:
            RiskDecision with APPROVE or REJECT
        """
        checks_passed = []
        checks_failed = []
        
        # Get price estimate
        price = last_price or order.limit_price or 100.0  # Default estimate
        order_notional = order.qty * price
        
        # Check 1: Max order notional
        if order_notional > self._config.max_order_notional_usd:
            checks_failed.append(
                f"Order notional ${order_notional:.2f} exceeds max ${self._config.max_order_notional_usd:.2f}"
            )
        else:
            checks_passed.append("max_order_notional")
        
        # Check 2: Require positive equity
        if self._config.require_positive_equity and equity <= 0:
            checks_failed.append(f"Account equity ${equity:.2f} is not positive")
        else:
            checks_passed.append("positive_equity")
        
        # Check 3: Max positions
        current_position_count = len([p for p in positions if abs(p.qty) > 0])
        symbol_has_position = any(p.symbol.upper() == order.symbol.upper() for p in positions)
        
        if not symbol_has_position and order.side == OrderSide.BUY:
            # Opening new position
            if current_position_count >= self._config.max_positions:
                checks_failed.append(
                    f"Position count {current_position_count} would exceed max {self._config.max_positions}"
                )
            else:
                checks_passed.append("max_positions")
        else:
            checks_passed.append("max_positions")
        
        # Check 4: Max position notional per symbol
        current_symbol_notional = 0
        for pos in positions:
            if pos.symbol.upper() == order.symbol.upper():
                current_symbol_notional = abs(pos.qty * price)
                break
        
        new_symbol_notional = current_symbol_notional
        if order.side == OrderSide.BUY:
            new_symbol_notional += order_notional
        else:
            new_symbol_notional = max(0, new_symbol_notional - order_notional)
        
        if new_symbol_notional > self._config.max_position_notional_per_symbol_usd:
            checks_failed.append(
                f"Position notional ${new_symbol_notional:.2f} would exceed max "
                f"${self._config.max_position_notional_per_symbol_usd:.2f}"
            )
        else:
            checks_passed.append("max_position_notional_per_symbol")
        
        # Check 5: Max total notional exposure
        total_exposure = sum(abs(p.qty * price) for p in positions)
        if order.side == OrderSide.BUY:
            total_exposure += order_notional
        
        if total_exposure > self._config.max_notional_exposure_usd:
            checks_failed.append(
                f"Total exposure ${total_exposure:.2f} would exceed max "
                f"${self._config.max_notional_exposure_usd:.2f}"
            )
        else:
            checks_passed.append("max_notional_exposure")
        
        # Check 6: Max daily turnover
        today = date.today()
        daily_turnover = self._daily_turnover.get(today, 0)
        new_turnover = daily_turnover + order_notional
        
        if new_turnover > self._config.max_daily_turnover_usd:
            checks_failed.append(
                f"Daily turnover ${new_turnover:.2f} would exceed max "
                f"${self._config.max_daily_turnover_usd:.2f}"
            )
        else:
            checks_passed.append("max_daily_turnover")
        
        # Check 7: Max leverage
        if equity > 0:
            new_leverage = total_exposure / equity
            if new_leverage > self._config.max_leverage:
                checks_failed.append(
                    f"Leverage {new_leverage:.2f}x would exceed max {self._config.max_leverage:.2f}x"
                )
            else:
                checks_passed.append("max_leverage")
        else:
            checks_passed.append("max_leverage")
        
        # Check 8: Block short selling
        if self._config.block_short_selling:
            if order.side == OrderSide.SELL:
                # Check if this would create a short position
                current_qty = 0
                for pos in positions:
                    if pos.symbol.upper() == order.symbol.upper():
                        current_qty = pos.qty
                        break
                
                if order.qty > current_qty:
                    checks_failed.append(
                        f"Short selling blocked: selling {order.qty} but only have {current_qty}"
                    )
                else:
                    checks_passed.append("block_short_selling")
            else:
                checks_passed.append("block_short_selling")
        else:
            checks_passed.append("block_short_selling")
        
        # Check 9: Block margin (simplified - just check we're not going negative cash)
        if self._config.block_margin:
            if order.side == OrderSide.BUY:
                # Simplified margin check
                if order_notional > equity * 0.95:  # Leave 5% buffer
                    checks_failed.append(
                        f"Order notional ${order_notional:.2f} too large for equity ${equity:.2f} (margin blocked)"
                    )
                else:
                    checks_passed.append("block_margin")
            else:
                checks_passed.append("block_margin")
        else:
            checks_passed.append("block_margin")
        
        # Build decision
        if checks_failed:
            reason = "; ".join(checks_failed)
            logger.warning(f"[RiskEngine] REJECTED: {order.symbol} - {reason}")
            
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                reason=reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        
        # All checks passed - record turnover
        self._daily_turnover[today] = new_turnover
        
        logger.info(f"[RiskEngine] APPROVED: {order.side.value} {order.qty} {order.symbol}")
        
        return RiskDecision(
            decision=RiskDecisionType.APPROVE,
            reason="All risk checks passed",
            checks_passed=checks_passed,
            checks_failed=[],
        )
    
    def get_daily_turnover(self) -> float:
        """Get today's turnover."""
        return self._daily_turnover.get(date.today(), 0)
    
    def reset_daily_turnover(self):
        """Reset daily turnover tracking."""
        self._daily_turnover.clear()
    
    def calculate_position_size(
        self,
        equity: float,
        last_price: float,
        volatility_pct: float = 2.0,
    ) -> float:
        """
        Calculate recommended position size based on risk parameters.
        
        Args:
            equity: Current account equity
            last_price: Current price of the asset
            volatility_pct: Expected volatility (for ATR-based sizing)
            
        Returns:
            Recommended number of shares
        """
        # Base position value from risk fraction
        base_notional = equity * self._config.per_trade_risk_fraction
        
        # Cap at max order notional
        capped_notional = min(base_notional, self._config.max_order_notional_usd)
        
        # Cap at max per-symbol notional
        capped_notional = min(capped_notional, self._config.max_position_notional_per_symbol_usd)
        
        # Convert to shares
        if last_price > 0:
            shares = capped_notional / last_price
        else:
            shares = 0
        
        return max(0, round(shares, 2))
