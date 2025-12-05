"""
Momentum Strategy.

Captures medium-term momentum moves (hours to days).
Bridges the gap between scalping and swing trading.
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


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy for trend-following trades.
    
    Targets 10-20% gains over 4-48 hours.
    Combines EOD and intraday signals for momentum confirmation.
    """
    
    def __init__(self, config=None):
        if config is None:
            config = STRATEGY_CONFIGS[StrategyType.MOMENTUM]
        super().__init__(config)
        self.model_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load both EOD and intraday models for momentum detection."""
        try:
            from ...ml.moonshot_predictor import MoonshotPredictor
            from ...ml.intraday_predictor import IntradayPredictor
            self.eod_predictor = MoonshotPredictor()
            self.intraday_predictor = IntradayPredictor()
            self.model_loaded = True
            logger.info("[MomentumStrategy] Models loaded")
        except Exception as e:
            logger.warning(f"[MomentumStrategy] Could not load models: {e}")
            self.eod_predictor = None
            self.intraday_predictor = None
    
    def generate_signals(self, symbols: List[str]) -> List[TradingIntent]:
        """Generate momentum intents from combined model analysis."""
        intents = []
        
        if not self.model_loaded:
            logger.warning("[MomentumStrategy] Models not loaded, skipping")
            return intents
        
        try:
            from ...server.ml_scanner import combined_moonshot_scan
            
            candidates = combined_moonshot_scan(
                symbols=symbols[:30],
                min_combined_confidence=self.config.min_score_threshold,
                eod_weight=0.5,
                intraday_weight=0.5,
            )
            
            for candidate in candidates.get("candidates", [])[:5]:
                symbol = candidate.get("symbol", "")
                combined_score = candidate.get("combined_score", 0)
                eod_conf = candidate.get("eod_confidence", 0)
                intraday_conf = candidate.get("intraday_confidence", 0)
                price = candidate.get("current_price", 0)
                
                if combined_score < self.config.min_score_threshold:
                    continue
                
                if price <= 0:
                    continue
                
                is_momentum = self._validate_momentum(eod_conf, intraday_conf)
                if not is_momentum:
                    continue
                
                stop_loss = round(price * (1 - self.config.default_stop_loss_pct), 2)
                take_profit = round(price * (1 + self.config.default_take_profit_pct), 2)
                
                intent = TradingIntent(
                    intent_id=f"momentum_{uuid.uuid4().hex[:8]}",
                    strategy_type=StrategyType.MOMENTUM,
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    score=combined_score,
                    entry_price=price,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    reason=f"Momentum: EOD {eod_conf:.0%} + Intraday {intraday_conf:.0%}",
                    metadata={
                        "eod_confidence": eod_conf,
                        "intraday_confidence": intraday_conf,
                        "signal_alignment": "both" if eod_conf > 0.5 and intraday_conf > 0.5 else "partial",
                    },
                )
                intents.append(intent)
                
        except Exception as e:
            logger.error(f"[MomentumStrategy] Signal generation error: {e}")
        
        logger.info(f"[MomentumStrategy] Generated {len(intents)} intents")
        return intents
    
    def _validate_momentum(self, eod_conf: float, intraday_conf: float) -> bool:
        """
        Validate momentum alignment.
        
        True momentum requires confirmation from both timeframes.
        """
        if eod_conf < 0.4:
            return False
        
        if intraday_conf < 0.4:
            return False
        
        avg_conf = (eod_conf + intraday_conf) / 2
        if avg_conf < 0.5:
            return False
        
        return True
    
    def get_exit_signals(self, positions: List[StrategyPosition]) -> List[TradingIntent]:
        """Check for exit conditions on momentum positions."""
        exits = []
        
        for pos in positions:
            should_exit, reason = self._check_exit_conditions(pos)
            if should_exit:
                exits.append(TradingIntent(
                    intent_id=f"momentum_exit_{uuid.uuid4().hex[:8]}",
                    strategy_type=StrategyType.MOMENTUM,
                    symbol=pos.symbol,
                    direction=SignalDirection.EXIT,
                    score=1.0,
                    reason=reason,
                ))
        
        return exits
    
    def _check_exit_conditions(self, pos: StrategyPosition) -> tuple[bool, str]:
        """
        Check momentum exit conditions.
        
        Momentum trades use trailing stops after initial gain.
        """
        if pos.current_price <= pos.stop_loss_price:
            return True, f"Stop-loss at ${pos.stop_loss_price}"
        
        if pos.current_price >= pos.take_profit_price:
            return True, f"Target hit at ${pos.take_profit_price}"
        
        if pos.unrealized_pnl_pct >= 0.10:
            trailing_stop = pos.current_price * 0.95
            if pos.current_price < trailing_stop:
                return True, f"Trailing stop with {pos.unrealized_pnl_pct:.0%} gain"
        
        entry_time = datetime.fromisoformat(pos.entry_time)
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        
        if hours_held > 48 and pos.unrealized_pnl_pct < 0.03:
            return True, "Time stop: 48h with <3% gain"
        
        return False, ""
