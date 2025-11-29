"""
MonsterRunner Engine for QuantraCore Apex.

Detects rare extreme-move precursor patterns.
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from enum import Enum

from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits


class RunnerState(str, Enum):
    IDLE = "idle"
    FORMING = "forming"
    PRIMED = "primed"


class RareEventClass(str, Enum):
    NONE = "none"
    PHASE_COMPRESSION = "phase_compression"
    VOLUME_IGNITION = "volume_ignition"
    RANGE_FLIP = "range_flip"
    ENTROPY_COLLAPSE = "entropy_collapse"
    SECTOR_CASCADE = "sector_cascade"


@dataclass
class MonsterRunnerOutput:
    """Output from MonsterRunner analysis."""
    runner_probability: float
    runner_state: RunnerState
    rare_event_class: RareEventClass
    compression_trace: float
    entropy_floor: float
    volume_pulse: float
    range_contraction: float
    primed_confidence: float
    details: Dict[str, Any]


class MonsterRunnerEngine:
    """
    Detects rare extreme-move patterns.
    
    This is a structural detection engine, NOT a trade signal generator.
    All outputs are framed as structural probabilities.
    """
    
    def __init__(self, log_dir: str = "logs/proof_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.compression_threshold = 0.6
        self.volume_threshold = 2.5
        self.entropy_threshold = 0.3
    
    def analyze(
        self,
        window: OhlcvWindow,
        microtraits: Optional[Microtraits] = None
    ) -> MonsterRunnerOutput:
        """
        Analyze window for rare event precursors.
        """
        bars = window.bars
        
        if microtraits is None:
            from src.quantracore_apex.core.microtraits import compute_microtraits
            microtraits = compute_microtraits(window)
        
        compression_trace = self._compute_compression_trace(bars)
        
        entropy_floor = self._compute_entropy_floor(bars)
        
        volume_pulse = self._compute_volume_pulse(bars)
        
        range_contraction = self._compute_range_contraction(bars)
        
        rare_event_class, primed_confidence = self._classify_rare_event(
            compression_trace=compression_trace,
            entropy_floor=entropy_floor,
            volume_pulse=volume_pulse,
            range_contraction=range_contraction,
            microtraits=microtraits
        )
        
        runner_state = self._determine_state(primed_confidence)
        
        runner_probability = self._compute_runner_probability(
            compression_trace=compression_trace,
            entropy_floor=entropy_floor,
            volume_pulse=volume_pulse,
            primed_confidence=primed_confidence
        )
        
        output = MonsterRunnerOutput(
            runner_probability=runner_probability,
            runner_state=runner_state,
            rare_event_class=rare_event_class,
            compression_trace=compression_trace,
            entropy_floor=entropy_floor,
            volume_pulse=volume_pulse,
            range_contraction=range_contraction,
            primed_confidence=primed_confidence,
            details={
                "symbol": window.symbol,
                "window_hash": window.get_hash(),
                "compression_threshold": self.compression_threshold,
                "volume_threshold": self.volume_threshold,
                "compliance_note": "Structural probability only - not a trade signal"
            }
        )
        
        if runner_state in [RunnerState.FORMING, RunnerState.PRIMED]:
            self._log_candidate(output, window)
        
        return output
    
    def _compute_compression_trace(self, bars) -> float:
        """Compute multi-bar compression trace."""
        if len(bars) < 20:
            return 0.0
        
        ranges = np.array([b.range for b in bars])
        
        recent = ranges[-5:]
        historical = ranges[-20:-5]
        
        if np.mean(historical) == 0:
            return 0.0
        
        compression = 1 - (np.mean(recent) / np.mean(historical))
        
        narrowing_count = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        narrowing_bonus = narrowing_count / len(recent) * 0.2
        
        return float(max(0, min(1, compression + narrowing_bonus)))
    
    def _compute_entropy_floor(self, bars) -> float:
        """Compute entropy floor (minimum entropy state)."""
        if len(bars) < 20:
            return 0.5
        
        closes = np.array([b.close for b in bars])
        returns = np.diff(closes) / closes[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) < 10:
            return 0.5
        
        recent_std = np.std(returns[-10:])
        historical_std = np.std(returns[:-10]) if len(returns) > 10 else recent_std
        
        if historical_std == 0:
            return 0.5
        
        entropy_ratio = recent_std / historical_std
        
        return float(max(0.1, min(1.0, entropy_ratio)))
    
    def _compute_volume_pulse(self, bars) -> float:
        """Compute volume pulse (sudden volume increase)."""
        if len(bars) < 20:
            return 0.0
        
        volumes = np.array([b.volume for b in bars])
        
        recent_vol = volumes[-3:]
        avg_vol = np.mean(volumes[-20:-3])
        
        if avg_vol == 0:
            return 0.0
        
        pulse = np.max(recent_vol) / avg_vol
        
        return float(max(0, min(5, pulse)))
    
    def _compute_range_contraction(self, bars) -> float:
        """Compute range contraction pattern."""
        if len(bars) < 10:
            return 0.0
        
        [b.range for b in bars[-10:]]
        
        inside_bars = 0
        for i in range(1, len(bars) - 1):
            if bars[i].high <= bars[i-1].high and bars[i].low >= bars[i-1].low:
                inside_bars += 1
        
        contraction_score = inside_bars / (len(bars) - 1) if len(bars) > 1 else 0
        
        return float(contraction_score)
    
    def _classify_rare_event(
        self,
        compression_trace: float,
        entropy_floor: float,
        volume_pulse: float,
        range_contraction: float,
        microtraits: Microtraits
    ) -> tuple:
        """Classify the type of rare event precursor."""
        
        if compression_trace > 0.7 and microtraits.compression_score > 0.6:
            return RareEventClass.PHASE_COMPRESSION, compression_trace
        
        if volume_pulse > self.volume_threshold and compression_trace > 0.4:
            return RareEventClass.VOLUME_IGNITION, min(1.0, volume_pulse / 5)
        
        if range_contraction > 0.5 and microtraits.volatility_ratio < 0.6:
            return RareEventClass.RANGE_FLIP, range_contraction
        
        if entropy_floor < self.entropy_threshold and microtraits.noise_score < 0.3:
            return RareEventClass.ENTROPY_COLLAPSE, 1 - entropy_floor
        
        return RareEventClass.NONE, 0.0
    
    def _determine_state(self, primed_confidence: float) -> RunnerState:
        """Determine runner state based on confidence."""
        if primed_confidence > 0.7:
            return RunnerState.PRIMED
        elif primed_confidence > 0.4:
            return RunnerState.FORMING
        else:
            return RunnerState.IDLE
    
    def _compute_runner_probability(
        self,
        compression_trace: float,
        entropy_floor: float,
        volume_pulse: float,
        primed_confidence: float
    ) -> float:
        """Compute overall runner probability."""
        base_prob = primed_confidence * 0.4
        
        compression_boost = compression_trace * 0.25
        
        entropy_boost = (1 - entropy_floor) * 0.15 if entropy_floor < 0.5 else 0
        
        volume_boost = min(0.2, volume_pulse / 10) if volume_pulse > 1.5 else 0
        
        probability = base_prob + compression_boost + entropy_boost + volume_boost
        
        return float(max(0, min(1, probability)))
    
    def _log_candidate(self, output: MonsterRunnerOutput, window: OhlcvWindow) -> None:
        """Log candidate rare event to proof logs."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": window.symbol,
            "window_hash": window.get_hash(),
            "runner_state": output.runner_state.value,
            "rare_event_class": output.rare_event_class.value,
            "runner_probability": output.runner_probability,
            "metrics": {
                "compression_trace": output.compression_trace,
                "entropy_floor": output.entropy_floor,
                "volume_pulse": output.volume_pulse,
                "range_contraction": output.range_contraction,
                "primed_confidence": output.primed_confidence,
            },
            "compliance_note": "Structural detection only - not a trade signal"
        }
        
        filename = f"monster_runner_{window.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=2)


MonsterRunner = MonsterRunnerEngine
