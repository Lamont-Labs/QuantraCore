"""
Automatic Swing Trade Executor.

LEGAL NOTICE AND COMPLIANCE:
- This module operates EXCLUSIVELY on Alpaca PAPER trading accounts
- NO REAL MONEY is ever at risk - all trades are simulated
- All outputs are for RESEARCH and EDUCATIONAL purposes only
- This is NOT financial advice and should not be treated as such
- Past performance in paper trading does NOT guarantee future results
- Users are solely responsible for any decisions made based on this system

Scans the universe, picks top setups, and executes simulated trades on Alpaca paper.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import json

from ..broker.adapters.alpaca_adapter import AlpacaPaperAdapter
from ..broker.models import OrderTicket, OrderMetadata
from ..broker.enums import OrderSide, OrderType, TimeInForce
from ..signals.signal_service import ApexSignalService

logger = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    """A trade setup ready for execution."""
    symbol: str
    quantrascore: float
    current_price: float
    entry_price: float
    stop_loss: float
    target: float
    shares: int
    position_value: float
    risk_amount: float
    reward_amount: float
    risk_reward: float
    conviction: str
    regime: str
    timing: str
    runner_prob: float


@dataclass 
class ExecutedTrade:
    """Result of an executed trade."""
    symbol: str
    order_id: str
    side: str
    shares: int
    fill_price: float
    status: str
    timestamp: str
    setup: TradeSetup
    error: Optional[str] = None


class AutoTrader:
    """
    Automatic swing trade executor for PAPER TRADING ONLY.
    
    IMPORTANT LEGAL DISCLAIMERS:
    - Executes ONLY on Alpaca paper trading accounts (simulated, no real money)
    - All signals are structural probability analyses for research purposes
    - This is NOT investment advice and should not be treated as such
    - Past paper trading performance does NOT guarantee future results
    - The system may produce losses even in paper trading
    
    Workflow:
    1. Scan universe for top setups (research analysis)
    2. Rank by QuantraScore (probability metric, not guarantee)
    3. Apply position sizing (simulated positions)
    4. Execute on Alpaca paper (NO REAL MONEY)
    """
    
    def __init__(
        self,
        max_positions: int = 3,
        max_position_pct: float = 0.10,  # 10% of equity per position
        min_quantrascore: float = 60.0,
        log_dir: str = "investor_logs/auto_trades/",
    ):
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct
        self.min_quantrascore = min_quantrascore
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.alpaca = AlpacaPaperAdapter()
        self.signal_service = ApexSignalService()
        
    def get_account_status(self) -> Dict[str, Any]:
        """Get Alpaca account status."""
        if not self.alpaca.is_configured:
            return {"error": "Alpaca not configured"}
        
        try:
            account = self.alpaca.get_account_info()
            positions = self.alpaca.get_positions()
            
            return {
                "equity": float(account.get("equity", 0)),
                "cash": float(account.get("cash", 0)),
                "buying_power": float(account.get("buying_power", 0)),
                "positions_count": len(positions),
                "positions": [
                    {
                        "symbol": p.symbol,
                        "qty": p.qty,
                        "market_value": p.market_value,
                        "unrealized_pl": p.unrealized_pl,
                    }
                    for p in positions
                ],
            }
        except Exception as e:
            return {"error": str(e)}
    
    def scan_for_setups(self, top_n: int = 10) -> List[TradeSetup]:
        """Scan universe and return top trade setups."""
        try:
            signals = self.signal_service.scan_universe()
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return []
        
        qualified = [
            s for s in signals
            if getattr(s, 'quantrascore', 0) >= self.min_quantrascore
        ]
        
        sorted_signals = sorted(
            qualified,
            key=lambda x: getattr(x, 'quantrascore', 0),
            reverse=True
        )[:top_n]
        
        account = self.get_account_status()
        equity = account.get("equity", 100000)
        max_position_value = equity * self.max_position_pct
        
        setups = []
        for sig in sorted_signals:
            price = getattr(sig, 'current_price', 0)
            entry = getattr(sig, 'suggested_entry', price)
            stop = getattr(sig, 'stop_loss', price * 0.98)
            target = getattr(sig, 'target_level_1', price * 1.04)
            
            if price <= 0:
                continue
                
            shares = int(max_position_value / price)
            if shares < 1:
                continue
            
            position_value = shares * price
            risk_per_share = entry - stop
            reward_per_share = target - entry
            
            if risk_per_share <= 0:
                risk_per_share = price * 0.02
            
            risk_amount = shares * risk_per_share
            reward_amount = shares * reward_per_share
            risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0
            
            conviction = getattr(sig, 'conviction_tier', None)
            if conviction and hasattr(conviction, 'value'):
                conviction = conviction.value
            else:
                conviction = str(conviction) if conviction else "low"
            
            timing = getattr(sig, 'timing_bucket', None)
            if timing and hasattr(timing, 'value'):
                timing = timing.value
            else:
                timing = str(timing) if timing else "none"
            
            setups.append(TradeSetup(
                symbol=getattr(sig, 'symbol', ''),
                quantrascore=getattr(sig, 'quantrascore', 0),
                current_price=price,
                entry_price=entry,
                stop_loss=stop,
                target=target,
                shares=shares,
                position_value=position_value,
                risk_amount=risk_amount,
                reward_amount=reward_amount,
                risk_reward=risk_reward,
                conviction=conviction,
                regime=getattr(sig, 'regime_prediction', 'unknown'),
                timing=timing,
                runner_prob=getattr(sig, 'runner_probability', 0),
            ))
        
        return setups
    
    def select_trades(self, setups: List[TradeSetup]) -> List[TradeSetup]:
        """Select top trades based on criteria."""
        current_positions = self.get_account_status().get("positions", [])
        current_symbols = {p["symbol"] for p in current_positions}
        
        available = [s for s in setups if s.symbol not in current_symbols]
        
        return available[:self.max_positions]
    
    def execute_trade(self, setup: TradeSetup) -> ExecutedTrade:
        """Execute a single trade on Alpaca."""
        if not self.alpaca.is_configured:
            return ExecutedTrade(
                symbol=setup.symbol,
                order_id="",
                side="buy",
                shares=setup.shares,
                fill_price=0,
                status="rejected",
                timestamp=datetime.utcnow().isoformat(),
                setup=setup,
                error="Alpaca not configured",
            )
        
        metadata = OrderMetadata(
            quantra_score=setup.quantrascore,
            regime=setup.regime,
            risk_tier="medium",
            protocols_fired=[],
            stop_loss_price=setup.stop_loss,
            take_profit_price=setup.target,
            risk_reward_ratio=setup.risk_reward,
            runner_probability=setup.runner_prob,
            quality_tier=setup.conviction[0].upper() if setup.conviction else "C",
        )
        
        ticket = OrderTicket(
            symbol=setup.symbol,
            qty=setup.shares,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            strategy_id="AutoSwingTrader",
            metadata=metadata,
        )
        
        try:
            result = self.alpaca.place_order(ticket)
            
            executed = ExecutedTrade(
                symbol=setup.symbol,
                order_id=result.order_id,
                side="buy",
                shares=setup.shares,
                fill_price=result.avg_fill_price or setup.current_price,
                status=result.status.value,
                timestamp=result.timestamp_utc or datetime.utcnow().isoformat(),
                setup=setup,
                error=result.error_message,
            )
            
            self._log_trade(executed)
            return executed
            
        except Exception as e:
            logger.error(f"Trade execution failed for {setup.symbol}: {e}")
            return ExecutedTrade(
                symbol=setup.symbol,
                order_id="",
                side="buy",
                shares=setup.shares,
                fill_price=0,
                status="error",
                timestamp=datetime.utcnow().isoformat(),
                setup=setup,
                error=str(e),
            )
    
    def execute_top_swings(self, count: int = 3) -> Dict[str, Any]:
        """
        Main entry point: Scan, select, and execute top swing trades.
        
        Returns dict with account status, selected trades, and execution results.
        """
        logger.info(f"[AutoTrader] Starting auto-swing execution for top {count} setups")
        
        account_before = self.get_account_status()
        if "error" in account_before:
            return {
                "success": False,
                "error": account_before["error"],
                "account": account_before,
            }
        
        setups = self.scan_for_setups(top_n=20)
        logger.info(f"[AutoTrader] Found {len(setups)} qualified setups (QS >= {self.min_quantrascore})")
        
        selected = self.select_trades(setups)[:count]
        logger.info(f"[AutoTrader] Selected {len(selected)} trades for execution")
        
        if not selected:
            return {
                "success": True,
                "message": "No qualified setups found or all slots filled",
                "account": account_before,
                "setups_scanned": len(setups),
                "trades_executed": [],
            }
        
        executed_trades = []
        for setup in selected:
            logger.info(f"[AutoTrader] Executing: BUY {setup.shares} {setup.symbol} @ ~${setup.current_price:.2f}")
            result = self.execute_trade(setup)
            executed_trades.append(result)
            logger.info(f"[AutoTrader] Result: {result.status} - Order ID: {result.order_id}")
        
        account_after = self.get_account_status()
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "account_before": account_before,
            "account_after": account_after,
            "setups_scanned": len(setups),
            "trades_selected": [
                {
                    "symbol": s.symbol,
                    "quantrascore": s.quantrascore,
                    "shares": s.shares,
                    "entry": s.entry_price,
                    "stop": s.stop_loss,
                    "target": s.target,
                    "risk_reward": round(s.risk_reward, 2),
                }
                for s in selected
            ],
            "trades_executed": [
                {
                    "symbol": t.symbol,
                    "order_id": t.order_id,
                    "shares": t.shares,
                    "fill_price": t.fill_price,
                    "status": t.status,
                    "error": t.error,
                }
                for t in executed_trades
            ],
        }
    
    def _log_trade(self, trade: ExecutedTrade):
        """Log executed trade for audit."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"auto_trade_{trade.symbol}_{timestamp}.json"
        
        log_data = {
            "timestamp": trade.timestamp,
            "symbol": trade.symbol,
            "order_id": trade.order_id,
            "side": trade.side,
            "shares": trade.shares,
            "fill_price": trade.fill_price,
            "status": trade.status,
            "error": trade.error,
            "setup": {
                "quantrascore": trade.setup.quantrascore,
                "entry_price": trade.setup.entry_price,
                "stop_loss": trade.setup.stop_loss,
                "target": trade.setup.target,
                "risk_reward": trade.setup.risk_reward,
                "conviction": trade.setup.conviction,
                "regime": trade.setup.regime,
            },
        }
        
        try:
            with open(self.log_dir / filename, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")


_auto_trader: Optional[AutoTrader] = None


def get_auto_trader() -> AutoTrader:
    """Get singleton AutoTrader instance."""
    global _auto_trader
    if _auto_trader is None:
        _auto_trader = AutoTrader()
    return _auto_trader
