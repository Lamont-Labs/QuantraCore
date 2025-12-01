"""
ApexSignal Service - Actionable Signal Generation for Manual Trading.

Generates ranked trading signals from ApexCore v3 predictions with:
- Timing predictions (when to enter)
- Entry/exit levels
- Confidence and conviction tiers
- Compliance notes
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class Direction(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class ConvictionTier(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    AVOID = "avoid"


class TimingUrgency(Enum):
    IMMEDIATE = "immediate"
    VERY_SOON = "very_soon"
    SOON = "soon"
    LATE = "late"
    NONE = "none"


@dataclass
class ApexSignalRecord:
    """Individual trading signal with all actionable information."""
    
    symbol: str
    timestamp: datetime
    model_version: str = "apexcore_v3"
    
    direction: Direction = Direction.NEUTRAL
    quantrascore: float = 0.5
    quantrascore_calibrated: float = 0.5
    
    runner_probability: float = 0.0
    quality_prediction: str = "fair"
    avoid_probability: float = 0.0
    regime_prediction: str = "chop"
    
    timing_bucket: TimingUrgency = TimingUrgency.NONE
    timing_confidence: float = 0.0
    bars_to_move_estimate: int = 11
    move_direction: int = 0
    
    expected_runup_pct: float = 0.0
    runup_confidence: float = 0.0
    predicted_top_price: float = 0.0
    
    current_price: float = 0.0
    suggested_entry: float = 0.0
    stop_loss: float = 0.0
    target_level_1: float = 0.0
    target_level_2: float = 0.0
    target_level_3: float = 0.0
    
    risk_reward_ratio: float = 0.0
    position_size_pct: float = 1.0
    
    conviction_tier: ConvictionTier = ConvictionTier.LOW
    priority_score: float = 0.0
    signal_strength: float = 0.0
    
    uncertainty_lower: float = 0.0
    uncertainty_upper: float = 1.0
    
    timing_guidance: str = ""
    action_window: str = ""
    compliance_note: str = "Structural analysis only - not trading advice"
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "direction": self.direction.value,
            "quantrascore": round(self.quantrascore, 4),
            "quantrascore_calibrated": round(self.quantrascore_calibrated, 4),
            "runner_probability": round(self.runner_probability, 4),
            "quality_prediction": self.quality_prediction,
            "avoid_probability": round(self.avoid_probability, 4),
            "regime_prediction": self.regime_prediction,
            "timing_bucket": self.timing_bucket.value,
            "timing_confidence": round(self.timing_confidence, 4),
            "bars_to_move_estimate": self.bars_to_move_estimate,
            "move_direction": self.move_direction,
            "expected_runup_pct": round(self.expected_runup_pct, 4),
            "runup_confidence": round(self.runup_confidence, 4),
            "predicted_top_price": round(self.predicted_top_price, 2),
            "current_price": round(self.current_price, 2),
            "suggested_entry": round(self.suggested_entry, 2),
            "stop_loss": round(self.stop_loss, 2),
            "target_level_1": round(self.target_level_1, 2),
            "target_level_2": round(self.target_level_2, 2),
            "target_level_3": round(self.target_level_3, 2),
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "position_size_pct": round(self.position_size_pct, 2),
            "conviction_tier": self.conviction_tier.value,
            "priority_score": round(self.priority_score, 4),
            "signal_strength": round(self.signal_strength, 4),
            "uncertainty_lower": round(self.uncertainty_lower, 4),
            "uncertainty_upper": round(self.uncertainty_upper, 4),
            "timing_guidance": self.timing_guidance,
            "action_window": self.action_window,
            "compliance_note": self.compliance_note,
        }


@dataclass
class SignalServiceConfig:
    """Configuration for signal generation service."""
    
    max_signals_stored: int = 1000
    signal_ttl_hours: int = 24
    min_quantrascore: float = 0.55
    min_runner_probability: float = 0.3
    max_avoid_probability: float = 0.7
    min_timing_confidence: float = 0.25
    default_stop_atr_multiplier: float = 2.0
    default_target_rr_1: float = 2.0
    default_target_rr_2: float = 3.0
    default_target_rr_3: float = 5.0
    slippage_percent: float = 0.001


class SignalPersistence:
    """Persists signals to disk for recovery and historical access."""
    
    def __init__(self, storage_dir: str = "signals"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def save_signals(self, signals: List[ApexSignalRecord]) -> str:
        """Save batch of signals to JSONL file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"signals_{timestamp}.jsonl"
        filepath = self.storage_dir / filename
        
        with self._lock:
            with open(filepath, 'w') as f:
                for signal in signals:
                    f.write(json.dumps(signal.to_dict()) + "\n")
        
        logger.info(f"Saved {len(signals)} signals to {filepath}")
        return str(filepath)
    
    def load_recent_signals(self, hours: int = 24) -> List[Dict]:
        """Load signals from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        signals = []
        
        with self._lock:
            for filepath in sorted(self.storage_dir.glob("signals_*.jsonl"), reverse=True):
                try:
                    file_time = datetime.strptime(filepath.stem.split("_", 1)[1], "%Y%m%d_%H%M%S")
                    if file_time < cutoff:
                        break
                    
                    with open(filepath) as f:
                        for line in f:
                            try:
                                signals.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.debug(f"Error loading {filepath}: {e}")
                    continue
        
        return signals
    
    def cleanup_old_signals(self, hours: int = 24):
        """Remove signal files older than N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            for filepath in self.storage_dir.glob("signals_*.jsonl"):
                try:
                    file_time = datetime.strptime(filepath.stem.split("_", 1)[1], "%Y%m%d_%H%M%S")
                    if file_time < cutoff:
                        filepath.unlink()
                        logger.debug(f"Removed old signal file: {filepath}")
                except Exception as e:
                    logger.debug(f"Error removing {filepath}: {e}")


class ApexSignalService:
    """
    Generates actionable trading signals from ApexCore v3 predictions.
    
    Features:
    - Bulk inference across symbol universe
    - Ranking and prioritization
    - Timing guidance for manual trading
    - Signal persistence
    """
    
    def __init__(
        self,
        config: Optional[SignalServiceConfig] = None,
        model_dir: str = "models/apexcore_v3/big"
    ):
        self.config = config or SignalServiceConfig()
        self.model_dir = Path(model_dir)
        self.persistence = SignalPersistence()
        
        self._signals: deque = deque(maxlen=self.config.max_signals_stored)
        self._last_scan_time: Optional[datetime] = None
        self._scan_in_progress = False
        self._lock = threading.Lock()
        
        self._model = None
    
    def _get_model(self):
        """Lazy load the ApexCore v3 model."""
        if self._model is None:
            try:
                from ..prediction.apexcore_v3 import ApexCoreV3Model
                self._model = ApexCoreV3Model.load(str(self.model_dir))
                logger.info(f"Loaded ApexCore v3 model from {self.model_dir}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        return self._model
    
    def _fetch_latest_data(self, symbol: str) -> Optional[Dict]:
        """Fetch latest market data for a symbol."""
        try:
            from ..apexlab.unified_trainer import AlpacaFetcher, WindowGenerator
            
            fetcher = AlpacaFetcher()
            if not fetcher.is_available():
                return None
            
            end_date = datetime.now() - timedelta(minutes=20)
            start_date = end_date - timedelta(days=10)
            
            bars = fetcher.fetch(symbol, start_date, end_date)
            if not bars or len(bars) < 50:
                return None
            
            window_gen = WindowGenerator(window_size=100, step_size=100)
            windows = list(window_gen.generate(bars, symbol, future_bars=0))
            
            if not windows:
                return None
            
            window = windows[-1][0]
            
            return {
                "symbol": symbol,
                "window": window,
                "current_price": bars[-1].close,
                "atr": self._calculate_atr(bars[-20:]),
                "volume_avg": np.mean([b.volume for b in bars[-20:]]),
            }
            
        except Exception as e:
            logger.debug(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_atr(self, bars) -> float:
        """Calculate Average True Range."""
        if len(bars) < 2:
            return 0.0
        
        trs = []
        for i in range(1, len(bars)):
            high_low = bars[i].high - bars[i].low
            high_prev_close = abs(bars[i].high - bars[i-1].close)
            low_prev_close = abs(bars[i].low - bars[i-1].close)
            trs.append(max(high_low, high_prev_close, low_prev_close))
        
        return float(np.mean(trs)) if trs else 0.0
    
    def _compute_entry_exit_levels(
        self,
        current_price: float,
        atr: float,
        direction: Direction
    ) -> Tuple[float, float, float, float, float]:
        """Compute entry, stop-loss, and target levels."""
        slippage = current_price * self.config.slippage_percent
        stop_distance = atr * self.config.default_stop_atr_multiplier
        
        if direction == Direction.LONG:
            suggested_entry = current_price + slippage
            stop_loss = current_price - stop_distance
            risk = suggested_entry - stop_loss
            
            target_1 = suggested_entry + (risk * self.config.default_target_rr_1)
            target_2 = suggested_entry + (risk * self.config.default_target_rr_2)
            target_3 = suggested_entry + (risk * self.config.default_target_rr_3)
            
        elif direction == Direction.SHORT:
            suggested_entry = current_price - slippage
            stop_loss = current_price + stop_distance
            risk = stop_loss - suggested_entry
            
            target_1 = suggested_entry - (risk * self.config.default_target_rr_1)
            target_2 = suggested_entry - (risk * self.config.default_target_rr_2)
            target_3 = suggested_entry - (risk * self.config.default_target_rr_3)
            
        else:
            suggested_entry = current_price
            stop_loss = current_price
            target_1 = target_2 = target_3 = current_price
        
        return suggested_entry, stop_loss, target_1, target_2, target_3
    
    def _compute_conviction_tier(
        self,
        quantrascore: float,
        runner_prob: float,
        avoid_prob: float,
        timing_confidence: float
    ) -> ConvictionTier:
        """Determine conviction tier based on signal quality."""
        if avoid_prob > 0.7:
            return ConvictionTier.AVOID
        
        if quantrascore >= 0.7 and runner_prob >= 0.6 and timing_confidence >= 0.5:
            return ConvictionTier.HIGH
        
        if quantrascore >= 0.6 and runner_prob >= 0.4 and timing_confidence >= 0.35:
            return ConvictionTier.MEDIUM
        
        return ConvictionTier.LOW
    
    def _compute_priority_score(
        self,
        quantrascore: float,
        runner_prob: float,
        avoid_prob: float,
        timing_bucket: str,
        timing_confidence: float
    ) -> float:
        """Compute composite priority score for ranking signals."""
        timing_urgency_weights = {
            "immediate": 1.0,
            "very_soon": 0.8,
            "soon": 0.6,
            "late": 0.4,
            "none": 0.1,
        }
        
        urgency_weight = timing_urgency_weights.get(timing_bucket, 0.1)
        
        base_score = (
            quantrascore * 0.35 +
            runner_prob * 0.25 +
            (1 - avoid_prob) * 0.15 +
            urgency_weight * 0.15 +
            timing_confidence * 0.10
        )
        
        if timing_bucket == "immediate" and timing_confidence > 0.6:
            base_score *= 1.2
        if avoid_prob > 0.5:
            base_score *= 0.5
        
        return min(base_score, 1.0)
    
    def _get_timing_guidance(self, timing_bucket: str, bars_to_move: int) -> Tuple[str, str]:
        """Convert timing prediction to actionable guidance."""
        bar_duration_minutes = 15
        
        bucket_guidance = {
            "immediate": ("ENTER NOW - Move expected within next bar", f"Next {bar_duration_minutes} minutes"),
            "very_soon": ("PREPARE ENTRY - Move expected in 2-3 bars", f"Next {2*bar_duration_minutes}-{3*bar_duration_minutes} minutes"),
            "soon": ("MONITOR CLOSELY - Move expected in 4-6 bars", f"Next {4*bar_duration_minutes}-{6*bar_duration_minutes} minutes"),
            "late": ("ADD TO WATCHLIST - Move expected in 7-10 bars", f"Next {7*bar_duration_minutes}-{10*bar_duration_minutes} minutes"),
            "none": ("NO TIMING SIGNAL - Move timing uncertain", "Unknown"),
        }
        
        guidance, window = bucket_guidance.get(timing_bucket, ("WAIT", "Unknown"))
        
        if bars_to_move <= 10:
            minutes = bars_to_move * bar_duration_minutes
            window = f"~{minutes} minutes ({bars_to_move} bars)"
        
        return guidance, window
    
    def generate_signal(self, symbol: str) -> Optional[ApexSignalRecord]:
        """Generate a single signal for a symbol."""
        data = self._fetch_latest_data(symbol)
        if not data:
            return None
        
        try:
            model = self._get_model()
            
            input_data = {
                "bars": data["window"].bars,
                "symbol": symbol,
            }
            
            prediction = model.predict(input_data)
            
            qs = prediction.quantrascore_calibrated
            runner_prob = prediction.runner_probability
            avoid_prob = prediction.avoid_trade_probability
            quality = prediction.quality_tier_pred
            regime = prediction.regime_pred
            timing_bucket = prediction.timing_bucket
            timing_conf = prediction.timing_confidence
            bars_to_move = prediction.bars_to_move_estimate
            move_dir = prediction.move_direction
            
            expected_runup_pct = getattr(prediction, 'expected_runup_pct', 0.0) or 0.0
            runup_confidence = getattr(prediction, 'runup_confidence', 0.0) or 0.0
            expected_runup_pct = float(expected_runup_pct) if expected_runup_pct else 0.0
            runup_confidence = float(runup_confidence) if runup_confidence else 0.0
            predicted_top = data["current_price"] * (1 + expected_runup_pct) if expected_runup_pct > 0 else data["current_price"]
            
            if qs >= 0.5 and move_dir >= 0:
                direction = Direction.LONG
            elif qs < 0.5 and move_dir <= 0:
                direction = Direction.SHORT
            else:
                direction = Direction.NEUTRAL
            
            entry, stop, t1, t2, t3 = self._compute_entry_exit_levels(
                data["current_price"],
                data["atr"],
                direction
            )
            
            risk = abs(entry - stop)
            reward = abs(t1 - entry)
            rr_ratio = reward / risk if risk > 0 else 0.0
            
            conviction = self._compute_conviction_tier(qs, runner_prob, avoid_prob, timing_conf)
            priority = self._compute_priority_score(qs, runner_prob, avoid_prob, timing_bucket, timing_conf)
            
            timing_guidance, action_window = self._get_timing_guidance(timing_bucket, bars_to_move)
            
            signal = ApexSignalRecord(
                symbol=symbol,
                timestamp=datetime.now(),
                direction=direction,
                quantrascore=prediction.quantrascore_pred,
                quantrascore_calibrated=qs,
                runner_probability=runner_prob,
                quality_prediction=quality,
                avoid_probability=avoid_prob,
                regime_prediction=regime,
                timing_bucket=TimingUrgency(timing_bucket),
                timing_confidence=timing_conf,
                bars_to_move_estimate=bars_to_move,
                move_direction=move_dir,
                expected_runup_pct=expected_runup_pct,
                runup_confidence=runup_confidence,
                predicted_top_price=predicted_top,
                current_price=data["current_price"],
                suggested_entry=entry,
                stop_loss=stop,
                target_level_1=t1,
                target_level_2=t2,
                target_level_3=t3,
                risk_reward_ratio=rr_ratio,
                conviction_tier=conviction,
                priority_score=priority,
                signal_strength=qs * runner_prob * (1 - avoid_prob),
                timing_guidance=timing_guidance,
                action_window=action_window,
                uncertainty_lower=prediction.uncertainty_lower,
                uncertainty_upper=prediction.uncertainty_upper,
            )
            
            return signal
            
        except Exception as e:
            logger.debug(f"Error generating signal for {symbol}: {e}")
            return None
    
    def scan_universe(
        self,
        symbols: Optional[List[str]] = None,
        top_n: int = 20,
        min_conviction: str = "low"
    ) -> List[ApexSignalRecord]:
        """Scan entire universe and return top ranked signals."""
        if symbols is None:
            from ..apexlab.continuous_learning import EXTENDED_UNIVERSE
            symbols = EXTENDED_UNIVERSE
        
        self._scan_in_progress = True
        signals = []
        
        try:
            for symbol in symbols[:100]:
                signal = self.generate_signal(symbol)
                if signal:
                    if self._passes_filters(signal, min_conviction):
                        signals.append(signal)
            
            signals.sort(key=lambda s: s.priority_score, reverse=True)
            top_signals = signals[:top_n]
            
            with self._lock:
                for signal in top_signals:
                    self._signals.append(signal)
                self._last_scan_time = datetime.now()
            
            if top_signals:
                self.persistence.save_signals(top_signals)
            
            return top_signals
            
        finally:
            self._scan_in_progress = False
    
    def _passes_filters(self, signal: ApexSignalRecord, min_conviction: str) -> bool:
        """Check if signal passes quality filters."""
        conviction_order = {"avoid": 0, "low": 1, "medium": 2, "high": 3}
        min_conv_level = conviction_order.get(min_conviction, 1)
        signal_conv_level = conviction_order.get(signal.conviction_tier.value, 1)
        
        if signal_conv_level < min_conv_level:
            return False
        
        if signal.quantrascore_calibrated < self.config.min_quantrascore:
            return False
        
        if signal.avoid_probability > self.config.max_avoid_probability:
            return False
        
        return True
    
    def get_live_signals(
        self,
        top_n: int = 20,
        direction_filter: Optional[str] = None,
        timing_filter: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """Get current live signals with optional filters."""
        with self._lock:
            signals = list(self._signals)
        
        if direction_filter:
            signals = [s for s in signals if s.direction.value == direction_filter]
        
        if timing_filter:
            signals = [s for s in signals if s.timing_bucket.value == timing_filter]
        
        if min_confidence > 0:
            signals = [s for s in signals if s.timing_confidence >= min_confidence]
        
        cutoff = datetime.now() - timedelta(hours=self.config.signal_ttl_hours)
        signals = [s for s in signals if s.timestamp > cutoff]
        
        signals.sort(key=lambda s: s.priority_score, reverse=True)
        
        return [s.to_dict() for s in signals[:top_n]]
    
    def get_status(self) -> Dict:
        """Get service status."""
        return {
            "signals_cached": len(self._signals),
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "scan_in_progress": self._scan_in_progress,
            "model_loaded": self._model is not None,
            "config": {
                "max_signals_stored": self.config.max_signals_stored,
                "signal_ttl_hours": self.config.signal_ttl_hours,
                "min_quantrascore": self.config.min_quantrascore,
            },
        }


_signal_service: Optional[ApexSignalService] = None


def get_signal_service() -> ApexSignalService:
    """Get or create the global signal service instance."""
    global _signal_service
    if _signal_service is None:
        _signal_service = ApexSignalService()
    return _signal_service
