"""
HyperLearner Integration Hooks.

Provides easy integration points for all system components
to emit events to the learning system.
"""

from typing import Any, Dict, List, Optional
from functools import wraps
import logging

from ..models import (
    EventCategory,
    EventType,
    LearningPriority,
    OutcomeType,
)
from ..core.hyperlearner import get_hyperlearner


logger = logging.getLogger(__name__)


def learn_from_scan(func):
    """Decorator to learn from ApexEngine scans."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        try:
            hyperlearner = get_hyperlearner()
            
            symbol = kwargs.get("symbol") or (args[1] if len(args) > 1 else None)
            
            if hasattr(result, "quantrascore"):
                context = {
                    "quantrascore": result.quantrascore,
                    "regime": getattr(result, "regime", "unknown"),
                    "risk_tier": getattr(result, "risk_tier", "unknown"),
                    "entropy": getattr(result, "entropy", 0),
                    "suppression": getattr(result, "suppression", 0),
                    "drift": getattr(result, "drift", 0),
                }
                
                if hasattr(result, "protocols_fired"):
                    context["protocols_fired"] = [p.protocol_id for p in result.protocols_fired]
                if hasattr(result, "omega_alerts"):
                    context["omega_triggers"] = result.omega_alerts
                    
                hyperlearner.emit(
                    category=EventCategory.ANALYSIS,
                    event_type=EventType.SCAN_COMPLETED,
                    source="apex_engine",
                    context=context,
                    symbol=symbol,
                    confidence=result.quantrascore / 100,
                    priority=LearningPriority.MEDIUM,
                )
        except Exception as e:
            logger.debug(f"[HyperLearner] Scan hook error: {e}")
            
        return result
    return wrapper


def learn_from_signal(func):
    """Decorator to learn from signal generation."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        try:
            hyperlearner = get_hyperlearner()
            
            if hasattr(result, "symbol"):
                context = {
                    "signal_type": getattr(result, "signal_type", "unknown"),
                    "direction": getattr(result, "direction", "unknown"),
                    "confidence": getattr(result, "confidence", 0),
                    "entry_price": getattr(result, "entry_price", 0),
                }
                
                hyperlearner.emit(
                    category=EventCategory.SIGNAL,
                    event_type=EventType.SIGNAL_GENERATED,
                    source="signal_builder",
                    context=context,
                    symbol=result.symbol,
                    confidence=getattr(result, "confidence", 0),
                    priority=LearningPriority.HIGH,
                )
        except Exception as e:
            logger.debug(f"[HyperLearner] Signal hook error: {e}")
            
        return result
    return wrapper


def learn_from_execution(func):
    """Decorator to learn from trade execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        try:
            hyperlearner = get_hyperlearner()
            
            if hasattr(result, "order_id"):
                context = {
                    "order_id": result.order_id,
                    "side": getattr(result, "side", "unknown"),
                    "quantity": getattr(result, "quantity", 0),
                    "fill_price": getattr(result, "fill_price", 0),
                    "status": getattr(result, "status", "unknown"),
                }
                
                event_type = EventType.TRADE_ENTERED if getattr(result, "is_entry", True) else EventType.TRADE_EXITED
                
                hyperlearner.emit(
                    category=EventCategory.EXECUTION,
                    event_type=event_type,
                    source="execution_engine",
                    context=context,
                    symbol=getattr(result, "symbol", None),
                    priority=LearningPriority.HIGH,
                )
        except Exception as e:
            logger.debug(f"[HyperLearner] Execution hook error: {e}")
            
        return result
    return wrapper


def learn_from_omega(func):
    """Decorator to learn from Omega directive triggers."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        try:
            hyperlearner = get_hyperlearner()
            
            omega_id = kwargs.get("omega_id") or (args[1] if len(args) > 1 else None)
            
            if result:
                context = {
                    "omega_id": omega_id,
                    "triggered": True,
                    "action": getattr(result, "action", "override"),
                    "reason": getattr(result, "reason", "safety_trigger"),
                }
                
                hyperlearner.emit(
                    category=EventCategory.OMEGA,
                    event_type=EventType.OMEGA_TRIGGERED,
                    source="omega_directives",
                    context=context,
                    priority=LearningPriority.CRITICAL,
                )
        except Exception as e:
            logger.debug(f"[HyperLearner] Omega hook error: {e}")
            
        return result
    return wrapper


class LearningContext:
    """Context manager for learning from operations."""
    
    def __init__(
        self,
        category: EventCategory,
        event_type: EventType,
        source: str,
        symbol: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.category = category
        self.event_type = event_type
        self.source = source
        self.symbol = symbol
        self.context = context or {}
        self.event_id: Optional[str] = None
        self._hyperlearner = get_hyperlearner()
        
    def __enter__(self):
        self.event_id = self._hyperlearner.emit(
            category=self.category,
            event_type=self.event_type,
            source=self.source,
            context=self.context,
            symbol=self.symbol,
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._hyperlearner.record_outcome(
                event_id=self.event_id,
                outcome_type=OutcomeType.FAIL,
                details={"error": str(exc_val)},
            )
        return False
        
    def succeed(
        self,
        return_pct: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Mark operation as successful."""
        self._hyperlearner.record_outcome(
            event_id=self.event_id,
            outcome_type=OutcomeType.WIN,
            return_pct=return_pct,
            was_correct=True,
            details=details,
        )
        
    def fail(
        self,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Mark operation as failed."""
        self._hyperlearner.record_outcome(
            event_id=self.event_id,
            outcome_type=OutcomeType.LOSS,
            was_correct=False,
            details={"reason": reason, **(details or {})},
        )


def emit_protocol_result(
    protocol_id: str,
    symbol: str,
    fired: bool,
    confidence: float,
    context: Optional[Dict[str, Any]] = None,
):
    """Emit a protocol firing result."""
    hyperlearner = get_hyperlearner()
    
    event_type = EventType.PROTOCOL_FIRED if fired else EventType.PROTOCOL_SILENT
    
    hyperlearner.emit(
        category=EventCategory.PROTOCOL,
        event_type=event_type,
        source=protocol_id,
        context={
            "protocol_id": protocol_id,
            "fired": fired,
            **(context or {}),
        },
        symbol=symbol,
        confidence=confidence,
        priority=LearningPriority.LOW,
    )


def emit_runner_detection(
    symbol: str,
    detected: bool,
    predicted_probability: float,
    actual_move: Optional[float] = None,
):
    """Emit runner detection result."""
    hyperlearner = get_hyperlearner()
    
    if actual_move is not None:
        was_runner = actual_move > 10
        
        if detected and was_runner:
            event_type = EventType.RUNNER_DETECTED
            outcome = OutcomeType.WIN
        elif not detected and was_runner:
            event_type = EventType.RUNNER_MISSED
            outcome = OutcomeType.LOSS
        else:
            event_type = EventType.PREDICTION_CORRECT if detected == was_runner else EventType.PREDICTION_WRONG
            outcome = OutcomeType.WIN if detected == was_runner else OutcomeType.LOSS
    else:
        event_type = EventType.RUNNER_DETECTED if detected else EventType.PROTOCOL_SILENT
        outcome = OutcomeType.PENDING
        
    event_id = hyperlearner.emit(
        category=EventCategory.PREDICTION,
        event_type=event_type,
        source="monster_runner",
        context={
            "detected": detected,
            "predicted_probability": predicted_probability,
            "actual_move": actual_move,
        },
        symbol=symbol,
        confidence=predicted_probability,
        priority=LearningPriority.HIGH,
    )
    
    if actual_move is not None:
        hyperlearner.record_outcome(
            event_id=event_id,
            outcome_type=outcome,
            return_pct=actual_move,
            was_correct=(outcome == OutcomeType.WIN),
        )


def emit_regime_change(
    symbol: str,
    old_regime: str,
    new_regime: str,
    context: Optional[Dict[str, Any]] = None,
):
    """Emit regime change detection."""
    hyperlearner = get_hyperlearner()
    
    hyperlearner.emit(
        category=EventCategory.ANALYSIS,
        event_type=EventType.REGIME_CHANGE,
        source="regime_detector",
        context={
            "old_regime": old_regime,
            "new_regime": new_regime,
            **(context or {}),
        },
        symbol=symbol,
        priority=LearningPriority.MEDIUM,
    )


def emit_data_anomaly(
    symbol: str,
    anomaly_type: str,
    severity: float,
    details: Optional[Dict[str, Any]] = None,
):
    """Emit data anomaly detection."""
    hyperlearner = get_hyperlearner()
    
    hyperlearner.emit(
        category=EventCategory.DATA,
        event_type=EventType.DATA_ANOMALY,
        source="data_validator",
        context={
            "anomaly_type": anomaly_type,
            "severity": severity,
            **(details or {}),
        },
        symbol=symbol,
        priority=LearningPriority.HIGH if severity > 0.8 else LearningPriority.MEDIUM,
    )
