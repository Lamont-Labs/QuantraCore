"""
MR02 — Volume Anomaly Detector

Detects unusual volume patterns that often precede large moves.
Analyzes volume spikes, accumulation/distribution, and climax patterns.

This is Stage 1 deterministic implementation using heuristic rules.

Version: 8.1
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR02Result:
    """Result of MR02 Volume Anomaly detection."""
    protocol_id: str = "MR02"
    fired: bool = False
    volume_anomaly_score: float = 0.0
    spike_magnitude: float = 0.0
    accumulation_signal: float = 0.0
    distribution_signal: float = 0.0
    climax_detected: bool = False
    dry_up_detected: bool = False
    confidence: float = 0.0
    notes: str = ""


def run_MR02(bars: List[OhlcvBar], lookback: int = 20) -> MR02Result:
    """
    Execute MR02 Volume Anomaly Detector.
    
    Identifies unusual volume patterns that may precede
    explosive moves. Uses deterministic analysis only.
    
    Args:
        bars: OHLCV price bars (minimum 30 required)
        lookback: Period for volume baseline
        
    Returns:
        MR02Result with volume anomaly metrics
        
    Protocol Logic:
    1. Calculate volume spike magnitude vs baseline
    2. Detect accumulation patterns (high volume + up closes)
    3. Detect distribution patterns (high volume + down closes)
    4. Identify volume climax (extreme spikes)
    5. Identify dry-up (abnormally low volume)
    """
    if len(bars) < 30:
        return MR02Result(notes="Insufficient data (need 30+ bars)")
    
    volumes = np.array([b.volume for b in bars], dtype=float)
    closes = np.array([b.close for b in bars])
    opens = np.array([b.open for b in bars])
    
    if np.all(volumes == 0):
        return MR02Result(notes="No volume data available")
    
    avg_volume = float(np.mean(volumes[-lookback:]))
    std_volume = float(np.std(volumes[-lookback:])) + 1e-10
    
    current_volume = volumes[-1]
    spike_magnitude = (current_volume - avg_volume) / std_volume
    
    up_bars = closes > opens
    down_bars = closes < opens
    
    recent_up_volume = float(np.sum(volumes[-lookback:] * up_bars[-lookback:]))
    recent_down_volume = float(np.sum(volumes[-lookback:] * down_bars[-lookback:]))
    total_recent_volume = recent_up_volume + recent_down_volume + 1e-10
    
    accumulation_signal = recent_up_volume / total_recent_volume
    distribution_signal = recent_down_volume / total_recent_volume
    
    volume_threshold_climax = avg_volume + (3 * std_volume)
    climax_detected = current_volume > volume_threshold_climax
    
    volume_threshold_dryup = avg_volume - (1.5 * std_volume)
    recent_avg = float(np.mean(volumes[-5:]))
    dry_up_detected = recent_avg < max(volume_threshold_dryup, avg_volume * 0.3)
    
    volume_anomaly_score = 0.0
    
    if climax_detected:
        volume_anomaly_score += 0.4
    if dry_up_detected:
        volume_anomaly_score += 0.3
    
    if abs(accumulation_signal - distribution_signal) > 0.3:
        volume_anomaly_score += 0.2
    
    if spike_magnitude > 2.0:
        volume_anomaly_score += min(spike_magnitude * 0.05, 0.1)
    
    volume_anomaly_score = float(np.clip(volume_anomaly_score, 0.0, 1.0))
    
    fired = volume_anomaly_score >= 0.5
    
    confidence = min(volume_anomaly_score * 100, 85.0)
    
    notes_parts = []
    if climax_detected:
        notes_parts.append("Volume climax")
    if dry_up_detected:
        notes_parts.append("Volume dry-up")
    if accumulation_signal > 0.65:
        notes_parts.append("Accumulation bias")
    if distribution_signal > 0.65:
        notes_parts.append("Distribution bias")
    
    return MR02Result(
        fired=fired,
        volume_anomaly_score=round(volume_anomaly_score, 4),
        spike_magnitude=round(float(spike_magnitude), 4),
        accumulation_signal=round(accumulation_signal, 4),
        distribution_signal=round(distribution_signal, 4),
        climax_detected=climax_detected,
        dry_up_detected=dry_up_detected,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No significant volume anomaly",
    )
