"""
Scalping Strategy.

Quick in/out trades targeting small gains (2-5%) within minutes to hours.
Uses intraday momentum and volatility patterns.
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


class ScalpStrategy(BaseStrategy):
    """
    Scalping Strategy for quick intraday trades.
    
    Targets 2-5% gains within minutes to hours.
    Uses volume spikes and momentum bursts.
    """
    
    def __init__(self, config=None):
        if config is None:
            config = STRATEGY_CONFIGS[StrategyType.SCALP]
        super().__init__(config)
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the intraday model for scalping signals."""
        try:
            from src.quantracore_apex.ml.intraday_predictor import IntradayMoonshotPredictor
            self.predictor = IntradayMoonshotPredictor()
            self.model_loaded = True
            logger.info("[ScalpStrategy] Intraday model loaded")
        except Exception as e:
            logger.warning(f"[ScalpStrategy] Could not load model: {e}")
            self.predictor = None
    
    def generate_signals(self, symbols: List[str]) -> List[TradingIntent]:
        """Generate scalp intents from intraday patterns."""
        intents = []
        
        if not self.model_loaded or not self.predictor:
            logger.warning("[ScalpStrategy] Model not loaded, skipping")
            return intents
        
        try:
            from src.quantracore_apex.trading.universe_scanner import get_universe_scanner
            
            scanner = get_universe_scanner()
            scalp_universe = scanner.get_strategy_universe('scalp', max_symbols=50)
            scan_symbols = scalp_universe if scalp_universe else symbols[:50]
            
            logger.info(f"[ScalpStrategy] Scanning {len(scan_symbols)} high-liquidity symbols")
            
            for symbol in scan_symbols:
                try:
                    prediction = self.predictor.predict_single(symbol)
                    if prediction is None:
                        continue
                    
                    confidence = prediction.get("confidence", 0)
                    if confidence < self.config.min_score_threshold:
                        continue
                    
                    price = prediction.get("current_price", 0)
                    if price <= 0:
                        continue
                    
                    is_scalp_setup = self._validate_scalp_setup(prediction)
                    if not is_scalp_setup:
                        continue
                    
                    stop_loss = round(price * (1 - self.config.default_stop_loss_pct), 2)
                    take_profit = round(price * (1 + self.config.default_take_profit_pct), 2)
                    
                    intent = TradingIntent(
                        intent_id=f"scalp_{uuid.uuid4().hex[:8]}",
                        strategy_type=StrategyType.SCALP,
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        score=confidence,
                        entry_price=price,
                        stop_loss_price=stop_loss,
                        take_profit_price=take_profit,
                        reason=f"Scalp setup: {confidence:.1%} confidence",
                        metadata={
                            "momentum_burst": prediction.get("momentum_score", 0),
                            "volume_spike": prediction.get("volume_ratio", 1.0),
                        },
                    )
                    intents.append(intent)
                    
                except Exception as e:
                    logger.debug(f"[ScalpStrategy] Skip {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"[ScalpStrategy] Signal generation error: {e}")
        
        logger.info(f"[ScalpStrategy] Generated {len(intents)} intents")
        return intents
    
    def _validate_scalp_setup(self, prediction: Dict[str, Any]) -> bool:
        """
        Validate if prediction qualifies as a scalp setup.
        
        Scalps need:
        - Tight spread (low slippage risk)
        - Sufficient volume (easy in/out)
        - Clear momentum direction
        """
        volume_ratio = prediction.get("volume_ratio", 1.0)
        if volume_ratio < 1.5:
            return False
        
        momentum = prediction.get("momentum_score", 0)
        if momentum < 0.5:
            return False
        
        return True
    
    def get_exit_signals(self, positions: List[StrategyPosition]) -> List[TradingIntent]:
        """Check for exit conditions on scalp positions."""
        exits = []
        
        for pos in positions:
            should_exit, reason = self._check_exit_conditions(pos)
            if should_exit:
                exits.append(TradingIntent(
                    intent_id=f"scalp_exit_{uuid.uuid4().hex[:8]}",
                    strategy_type=StrategyType.SCALP,
                    symbol=pos.symbol,
                    direction=SignalDirection.EXIT,
                    score=1.0,
                    reason=reason,
                ))
        
        return exits
    
    def _check_exit_conditions(self, pos: StrategyPosition) -> tuple[bool, str]:
        """
        Check scalp exit conditions.
        
        Scalps exit quickly - either at target or stop.
        """
        if pos.current_price <= pos.stop_loss_price:
            return True, f"Stop-loss at ${pos.stop_loss_price}"
        
        if pos.current_price >= pos.take_profit_price:
            return True, f"Target hit at ${pos.take_profit_price}"
        
        entry_time = datetime.fromisoformat(pos.entry_time)
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        
        if hours_held > 2 and pos.unrealized_pnl_pct < 0.01:
            return True, "Time stop: 2h with no progress"
        
        if hours_held > 4:
            return True, "Max hold time exceeded for scalp"
        
        return False, ""
