"""
Swing Trading Strategy.

Uses the EOD model for multi-day holds (2-5 trading days).
This wraps the existing moonshot detection system.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
import uuid

from ..strategy_orchestrator import (
    BaseStrategy, TradingIntent, StrategyPosition,
    StrategyType, SignalDirection, STRATEGY_CONFIGS,
)

logger = logging.getLogger(__name__)


class SwingStrategy(BaseStrategy):
    """
    Swing Trading Strategy using EOD model.
    
    Targets 50%+ gains within 5 trading days.
    Uses the massive_ensemble_v3 model for signal generation.
    """
    
    def __init__(self, config=None):
        if config is None:
            config = STRATEGY_CONFIGS[StrategyType.SWING]
        super().__init__(config)
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the EOD moonshot model."""
        try:
            from ...ml.moonshot_predictor import MoonshotPredictor
            self.predictor = MoonshotPredictor()
            self.model_loaded = True
            logger.info("[SwingStrategy] EOD model loaded")
        except Exception as e:
            logger.warning(f"[SwingStrategy] Could not load model: {e}")
            self.predictor = None
    
    def generate_signals(self, symbols: List[str]) -> List[TradingIntent]:
        """Generate swing trade intents from EOD model."""
        intents = []
        
        if not self.model_loaded or not self.predictor:
            logger.warning("[SwingStrategy] Model not loaded, skipping")
            return intents
        
        try:
            from ...server.ml_scanner import scan_for_runners
            
            candidates = scan_for_runners(
                symbols=symbols,
                top_n=10,
                min_confidence=self.config.min_score_threshold,
            )
            
            for candidate in candidates:
                symbol = candidate.get("symbol", "")
                confidence = candidate.get("confidence", 0)
                price = candidate.get("current_price", 0)
                
                if confidence < self.config.min_score_threshold:
                    continue
                
                if price <= 0:
                    continue
                
                stop_loss = round(price * (1 - self.config.default_stop_loss_pct), 2)
                take_profit = round(price * (1 + self.config.default_take_profit_pct), 2)
                
                intent = TradingIntent(
                    intent_id=f"swing_{uuid.uuid4().hex[:8]}",
                    strategy_type=StrategyType.SWING,
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    score=confidence,
                    entry_price=price,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    reason=f"EOD model confidence: {confidence:.1%}",
                    metadata={
                        "model": "massive_ensemble_v3",
                        "features": candidate.get("top_features", []),
                    },
                )
                intents.append(intent)
                
        except Exception as e:
            logger.error(f"[SwingStrategy] Signal generation error: {e}")
        
        logger.info(f"[SwingStrategy] Generated {len(intents)} intents")
        return intents
    
    def get_exit_signals(self, positions: List[StrategyPosition]) -> List[TradingIntent]:
        """Check for exit conditions on swing positions."""
        exits = []
        
        for pos in positions:
            should_exit, reason = self._check_exit_conditions(pos)
            if should_exit:
                exits.append(TradingIntent(
                    intent_id=f"swing_exit_{uuid.uuid4().hex[:8]}",
                    strategy_type=StrategyType.SWING,
                    symbol=pos.symbol,
                    direction=SignalDirection.EXIT,
                    score=1.0,
                    reason=reason,
                ))
        
        return exits
    
    def _check_exit_conditions(self, pos: StrategyPosition) -> tuple[bool, str]:
        """Check if position should be exited."""
        if pos.current_price <= pos.stop_loss_price:
            return True, f"Stop-loss triggered at ${pos.stop_loss_price}"
        
        if pos.current_price >= pos.take_profit_price:
            return True, f"Take-profit triggered at ${pos.take_profit_price}"
        
        entry_time = datetime.fromisoformat(pos.entry_time)
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        max_hold = self.config.hold_time_hours[1]
        
        if hours_held > max_hold and pos.unrealized_pnl_pct < 0.05:
            return True, f"Time stop: held {hours_held:.0f}h with only {pos.unrealized_pnl_pct:.1%} gain"
        
        return False, ""
