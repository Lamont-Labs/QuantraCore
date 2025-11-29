"""
Broker Router for QuantraCore Apex.

Selects and instantiates the appropriate broker adapter based on configuration.
"""

import logging
from typing import List, Optional

from .config import BrokerConfig, load_broker_config
from .enums import ExecutionMode
from .models import OrderTicket, ExecutionResult, BrokerPosition
from .adapters.base_adapter import BrokerAdapter
from .adapters.null_adapter import NullAdapter
from .adapters.paper_sim_adapter import PaperSimAdapter
from .adapters.alpaca_adapter import AlpacaPaperAdapter


logger = logging.getLogger(__name__)


class BrokerRouter:
    """
    Routes orders to the appropriate broker adapter.
    
    Selection logic:
    - RESEARCH mode → NullAdapter (logs only)
    - PAPER mode + Alpaca configured → AlpacaPaperAdapter
    - PAPER mode + no Alpaca → PaperSimAdapter
    - LIVE mode → DISABLED (raises error)
    """
    
    def __init__(self, config: Optional[BrokerConfig] = None, config_path: str = "config/broker.yaml"):
        self._config = config or load_broker_config(config_path)
        self._adapter: Optional[BrokerAdapter] = None
        self._initialize_adapter()
    
    def _initialize_adapter(self):
        """Initialize the appropriate adapter based on configuration."""
        mode = self._config.execution_mode
        
        if mode == ExecutionMode.RESEARCH:
            logger.info("[BrokerRouter] Mode: RESEARCH - using NullAdapter (no execution)")
            self._adapter = NullAdapter()
        
        elif mode == ExecutionMode.PAPER:
            # Check if Alpaca is configured
            alpaca_config = self._config.alpaca_paper
            
            if alpaca_config.is_configured and alpaca_config.enabled:
                logger.info("[BrokerRouter] Mode: PAPER - using AlpacaPaperAdapter")
                self._adapter = AlpacaPaperAdapter(
                    api_key=alpaca_config.api_key,
                    api_secret=alpaca_config.api_secret,
                    base_url=alpaca_config.base_url,
                    log_raw_dir=self._config.logging.broker_raw_dir,
                )
            else:
                logger.info("[BrokerRouter] Mode: PAPER - using PaperSimAdapter (no Alpaca configured)")
                self._adapter = PaperSimAdapter()
        
        elif mode == ExecutionMode.LIVE:
            # LIVE mode is disabled
            logger.error("[BrokerRouter] LIVE mode is DISABLED. Use PAPER mode instead.")
            raise RuntimeError(
                "LIVE trading is DISABLED. This system is for PAPER trading only. "
                "To use paper trading, set execution.mode to PAPER in config/broker.yaml"
            )
        
        else:
            logger.warning(f"[BrokerRouter] Unknown mode: {mode}, defaulting to RESEARCH")
            self._adapter = NullAdapter()
    
    @property
    def adapter(self) -> BrokerAdapter:
        """Get the active broker adapter."""
        if self._adapter is None:
            raise RuntimeError("Broker adapter not initialized")
        return self._adapter
    
    @property
    def mode(self) -> ExecutionMode:
        """Get the current execution mode."""
        return self._config.execution_mode
    
    @property
    def is_paper(self) -> bool:
        """Check if running in paper/simulation mode."""
        return self._adapter is not None and self._adapter.is_paper
    
    @property
    def adapter_name(self) -> str:
        """Get the name of the active adapter."""
        return self._adapter.name if self._adapter else "NONE"
    
    def place_order(self, order: OrderTicket) -> ExecutionResult:
        """
        Route an order to the appropriate adapter.
        
        Args:
            order: OrderTicket to execute
            
        Returns:
            ExecutionResult from the broker
        """
        logger.info(
            f"[BrokerRouter] Routing order: {order.side.value} {order.qty} {order.symbol} "
            f"via {self.adapter_name}"
        )
        return self.adapter.place_order(order)
    
    def cancel_order(self, order_id: str) -> ExecutionResult:
        """Cancel an order."""
        return self.adapter.cancel_order(order_id)
    
    def get_open_orders(self) -> List[ExecutionResult]:
        """Get all open orders."""
        return self.adapter.get_open_orders()
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get all current positions."""
        return self.adapter.get_positions()
    
    def get_position(self, symbol: str) -> BrokerPosition:
        """Get position for a specific symbol."""
        return self.adapter.get_position(symbol)
    
    def get_account_equity(self) -> float:
        """Get current account equity."""
        return self.adapter.get_account_equity()
    
    def get_last_price(self, symbol: str) -> float:
        """Get last known price for a symbol."""
        return self.adapter.get_last_price(symbol)
    
    def sync_state(self):
        """Sync state from broker."""
        self.adapter.sync_state()
    
    def get_status(self) -> dict:
        """Get router status for diagnostics."""
        return {
            "mode": self._config.execution_mode.value,
            "adapter": self.adapter_name,
            "is_paper": self.is_paper,
            "config": self._config.to_dict(),
        }
