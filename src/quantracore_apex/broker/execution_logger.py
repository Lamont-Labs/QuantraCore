"""
Execution Logger for QuantraCore Apex Broker Layer.

Structured logging for all execution activity, risk decisions, and audit records.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import LoggingConfig
from .models import OrderTicket, ExecutionResult, RiskDecision


logger = logging.getLogger(__name__)


class ExecutionLogger:
    """
    Structured logging for execution activity.
    
    Logs:
    - execution_log: All orders and fills
    - audit_log: Detailed risk decisions and account state
    """
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self._config = config or LoggingConfig()
        
        # Create log directories
        self._execution_dir = Path(self._config.execution_log_dir)
        self._audit_dir = Path(self._config.audit_log_dir)
        
        self._execution_dir.mkdir(parents=True, exist_ok=True)
        self._audit_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_log_path(self, base_dir: Path, prefix: str) -> Path:
        """Get log file path for today."""
        today = datetime.utcnow().strftime("%Y%m%d")
        return base_dir / f"{prefix}_{today}.log"
    
    def _append_log(self, path: Path, data: dict):
        """Append a log entry."""
        try:
            with open(path, 'a') as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")
    
    def log_execution(
        self,
        ticket: OrderTicket,
        result: ExecutionResult,
        risk_decision: RiskDecision,
    ):
        """
        Log an execution event.
        
        Args:
            ticket: The order ticket
            result: Execution result from broker
            risk_decision: Risk engine decision
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "EXECUTION",
            "ticket_id": ticket.ticket_id,
            "order_id": result.order_id,
            "symbol": ticket.symbol,
            "side": ticket.side.value,
            "qty": ticket.qty,
            "order_type": ticket.order_type.value,
            "intent": ticket.intent.value,
            "limit_price": ticket.limit_price,
            "stop_price": ticket.stop_price,
            "status": result.status.value,
            "filled_qty": result.filled_qty,
            "avg_fill_price": result.avg_fill_price,
            "broker": result.broker,
            "risk_approved": risk_decision.approved,
            "source_signal_id": ticket.source_signal_id,
            "strategy_id": ticket.strategy_id,
            "metadata": ticket.metadata.to_dict(),
        }
        
        path = self._get_log_path(self._execution_dir, "execution")
        self._append_log(path, log_entry)
        
        logger.info(
            f"[ExecutionLog] {ticket.side.value} {ticket.qty} {ticket.symbol} "
            f"-> {result.status.value} @ {result.avg_fill_price}"
        )
    
    def log_risk_decision(
        self,
        ticket: OrderTicket,
        decision: RiskDecision,
        equity: float,
    ):
        """
        Log a risk decision for audit.
        
        Args:
            ticket: The order ticket being checked
            decision: Risk engine decision
            equity: Current account equity
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "RISK_CHECK",
            "ticket_id": ticket.ticket_id,
            "symbol": ticket.symbol,
            "side": ticket.side.value,
            "qty": ticket.qty,
            "order_notional": ticket.qty * (ticket.limit_price or 100),
            "decision": decision.decision.value,
            "reason": decision.reason,
            "checks_passed": decision.checks_passed,
            "checks_failed": decision.checks_failed,
            "equity_before": equity,
            "source_signal_id": ticket.source_signal_id,
            "ticket_hash": ticket.hash(),
        }
        
        path = self._get_log_path(self._audit_dir, "audit")
        self._append_log(path, log_entry)
    
    def log_signal_received(self, signal_id: str, symbol: str, direction: str, metadata: dict):
        """Log when a signal is received."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "SIGNAL_RECEIVED",
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "metadata": metadata,
        }
        
        path = self._get_log_path(self._execution_dir, "execution")
        self._append_log(path, log_entry)
    
    def log_error(self, error_type: str, message: str, context: dict = None):
        """Log an error event."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "ERROR",
            "error_type": error_type,
            "message": message,
            "context": context or {},
        }
        
        path = self._get_log_path(self._audit_dir, "audit")
        self._append_log(path, log_entry)
        
        logger.error(f"[ExecutionLog] ERROR: {error_type} - {message}")
