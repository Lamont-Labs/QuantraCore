"""
MonsterRunner Strategy.

Detects stocks ready for extreme moves (50%+ gains).
Uses existing MonsterRunner protocols and mega_runners models.
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


class MonsterRunnerStrategy(BaseStrategy):
    """
    MonsterRunner Strategy for extreme move detection.
    
    Targets 100%+ gains on high-conviction setups.
    Uses specialized models for detecting "moonshots" and "superruns".
    """
    
    def __init__(self, config=None):
        if config is None:
            config = STRATEGY_CONFIGS[StrategyType.MONSTER_RUNNER]
        super().__init__(config)
        self.model_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load MonsterRunner detection models."""
        try:
            from ...ml.moonshot_predictor import MoonshotPredictor
            self.predictor = MoonshotPredictor()
            self.model_loaded = True
            logger.info("[MonsterRunnerStrategy] Models loaded")
        except Exception as e:
            logger.warning(f"[MonsterRunnerStrategy] Could not load models: {e}")
            self.predictor = None
    
    def generate_signals(self, symbols: List[str]) -> List[TradingIntent]:
        """Generate MonsterRunner intents for extreme move candidates."""
        intents = []
        
        if not self.model_loaded or not self.predictor:
            logger.warning("[MonsterRunnerStrategy] Models not loaded, skipping")
            return intents
        
        try:
            from ...server.ml_scanner import scan_for_runners
            
            candidates = scan_for_runners(
                symbols=symbols,
                top_n=5,
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
                
                is_monster = self._classify_as_monster(candidate)
                if not is_monster:
                    continue
                
                stop_loss = round(price * (1 - self.config.default_stop_loss_pct), 2)
                take_profit = round(price * (1 + self.config.default_take_profit_pct), 2)
                
                intent = TradingIntent(
                    intent_id=f"monster_{uuid.uuid4().hex[:8]}",
                    strategy_type=StrategyType.MONSTER_RUNNER,
                    symbol=symbol,
                    direction=SignalDirection.LONG,
                    score=confidence,
                    entry_price=price,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    reason=f"MonsterRunner signal: {confidence:.1%} confidence",
                    metadata={
                        "monster_type": "superrun",
                        "features": candidate.get("top_features", []),
                        "volume_surge": candidate.get("volume_ratio", 1.0),
                    },
                )
                intents.append(intent)
                
        except Exception as e:
            logger.error(f"[MonsterRunnerStrategy] Signal generation error: {e}")
        
        logger.info(f"[MonsterRunnerStrategy] Generated {len(intents)} intents")
        return intents
    
    def _classify_as_monster(self, candidate: Dict[str, Any]) -> bool:
        """
        Determine if a candidate qualifies as a MonsterRunner.
        
        MonsterRunners have specific characteristics:
        - High volume surge (3x+ normal)
        - Strong momentum
        - Breaking key resistance levels
        - High institutional interest signals
        """
        confidence = candidate.get("confidence", 0)
        if confidence < 0.85:
            return False
        
        volume_ratio = candidate.get("volume_ratio", 1.0)
        if volume_ratio < 2.0:
            return False
        
        momentum_score = candidate.get("momentum_score", 0)
        if momentum_score < 0.6:
            return False
        
        return True
    
    def get_exit_signals(self, positions: List[StrategyPosition]) -> List[TradingIntent]:
        """Check for exit conditions on MonsterRunner positions."""
        exits = []
        
        for pos in positions:
            should_exit, reason = self._check_exit_conditions(pos)
            if should_exit:
                exits.append(TradingIntent(
                    intent_id=f"monster_exit_{uuid.uuid4().hex[:8]}",
                    strategy_type=StrategyType.MONSTER_RUNNER,
                    symbol=pos.symbol,
                    direction=SignalDirection.EXIT,
                    score=1.0,
                    reason=reason,
                ))
        
        return exits
    
    def _check_exit_conditions(self, pos: StrategyPosition) -> tuple[bool, str]:
        """
        Check MonsterRunner exit conditions.
        
        MonsterRunners use wider stops and higher targets.
        """
        if pos.current_price <= pos.stop_loss_price:
            return True, f"Stop-loss triggered at ${pos.stop_loss_price}"
        
        if pos.current_price >= pos.take_profit_price:
            return True, f"Monster target hit at ${pos.take_profit_price}"
        
        if pos.unrealized_pnl_pct >= 0.50:
            trailing_stop = pos.current_price * 0.85
            if pos.current_price < trailing_stop:
                return True, f"Trailing stop triggered with {pos.unrealized_pnl_pct:.0%} gain"
        
        return False, ""
