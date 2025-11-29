"""
MonsterRunner Fuse Score Calculator for QuantraCore Apex.

Fuses multiple MonsterRunner signals with volatility, liquidity,
and small-cap characteristics into a unified runner probability score.

Version: 9.0-A
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np

from src.quantracore_apex.core.schemas import OhlcvBar


@dataclass
class FuseScoreResult:
    """Result of MonsterRunner fuse score calculation."""
    fuse_score: float = 0.0
    mr_bias: float = 0.0
    volatility_expansion: float = 0.0
    liquidity_score: float = 0.0
    float_pressure: float = 0.0
    trend_strength: float = 0.0
    
    risk_flags: List[str] = field(default_factory=list)
    runner_candidate: bool = False
    runner_tier: str = "none"
    
    components: dict = field(default_factory=dict)
    notes: str = ""


def calculate_volatility_expansion(
    bars: List[OhlcvBar],
    lookback: int = 20
) -> float:
    """
    Calculate volatility expansion relative to historical.
    
    Args:
        bars: OHLCV price bars
        lookback: Period for calculation
        
    Returns:
        Volatility expansion ratio (>1 = expanding)
    """
    if len(bars) < lookback * 2:
        return 1.0
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    
    recent_atr = float(np.mean(tr[-lookback:]))
    historical_atr = float(np.mean(tr[:-lookback]))
    
    if historical_atr == 0:
        return 1.0
    
    return recent_atr / historical_atr


def calculate_liquidity_score(
    bars: List[OhlcvBar],
    lookback: int = 20
) -> float:
    """
    Calculate liquidity score based on volume patterns.
    
    Args:
        bars: OHLCV price bars
        lookback: Period for calculation
        
    Returns:
        Liquidity score (0-100)
    """
    if len(bars) < lookback * 2:
        return 50.0
    
    volumes = np.array([b.volume for b in bars])
    
    recent_avg = float(np.mean(volumes[-lookback:]))
    historical_avg = float(np.mean(volumes[:-lookback]))
    
    if historical_avg == 0:
        return 50.0
    
    rel_volume = recent_avg / historical_avg
    
    score = min(rel_volume * 30, 100)
    
    return float(score)


def calculate_float_pressure(
    float_millions: float,
    avg_volume: float,
    price: float
) -> float:
    """
    Calculate float pressure (how much of float trades daily).
    
    Higher pressure = more likely to see explosive moves.
    
    Args:
        float_millions: Float in millions of shares
        avg_volume: Average daily volume
        price: Current price
        
    Returns:
        Float pressure score (0-100)
    """
    if float_millions <= 0 or avg_volume <= 0:
        return 0.0
    
    float_shares = float_millions * 1_000_000
    
    daily_turnover = avg_volume / float_shares
    
    if daily_turnover >= 0.5:
        pressure = 100.0
    elif daily_turnover >= 0.2:
        pressure = 70 + (daily_turnover - 0.2) * 100
    elif daily_turnover >= 0.1:
        pressure = 40 + (daily_turnover - 0.1) * 300
    elif daily_turnover >= 0.05:
        pressure = 20 + (daily_turnover - 0.05) * 400
    else:
        pressure = daily_turnover * 400
    
    return min(float(pressure), 100.0)


def calculate_trend_strength(
    bars: List[OhlcvBar],
    lookback: int = 20
) -> float:
    """
    Calculate trend strength score.
    
    Args:
        bars: OHLCV price bars
        lookback: Period for calculation
        
    Returns:
        Trend strength (-100 to +100, positive = bullish)
    """
    if len(bars) < lookback:
        return 0.0
    
    closes = np.array([b.close for b in bars[-lookback:]])
    
    x = np.arange(len(closes))
    np.polyfit(x, closes, 1)[0]
    
    pct_change = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0
    
    strength = pct_change * 2
    
    return float(np.clip(strength, -100, 100))


def get_mr_protocol_bias(engine_result: Any) -> float:
    """
    Extract MonsterRunner bias from engine result.
    
    Args:
        engine_result: ApexResult from engine
        
    Returns:
        MR bias score (0-100)
    """
    if engine_result is None:
        return 0.0
    
    mr_score = 0.0
    mr_count = 0
    
    if hasattr(engine_result, 'monster_runner_state'):
        mr_state = engine_result.monster_runner_state
        if mr_state:
            if hasattr(mr_state, 'compression_score'):
                mr_score += mr_state.compression_score * 100
                mr_count += 1
            if hasattr(mr_state, 'gap_fade_score'):
                mr_score += mr_state.gap_fade_score * 100
                mr_count += 1
            if hasattr(mr_state, 'breakout_score'):
                mr_score += mr_state.breakout_score * 100
                mr_count += 1
    
    if hasattr(engine_result, 'protocols_fired'):
        for protocol in engine_result.protocols_fired:
            if protocol.startswith('MR'):
                mr_score += 20
                mr_count += 1
    
    if mr_count > 0:
        return min(mr_score / mr_count, 100.0)
    
    return 0.0


def calculate_mr_fuse_score(
    bars: List[OhlcvBar],
    symbol_info: Optional[Any] = None,
    engine_result: Optional[Any] = None,
) -> FuseScoreResult:
    """
    Calculate the unified MonsterRunner fuse score.
    
    Combines:
    - MR protocol bias
    - Volatility expansion
    - Liquidity conditions
    - Float pressure
    - Trend strength
    
    Args:
        bars: OHLCV price bars
        symbol_info: SymbolInfo from universe
        engine_result: ApexResult from engine
        
    Returns:
        FuseScoreResult with fuse score and components
    """
    result = FuseScoreResult()
    
    if len(bars) < 20:
        result.notes = "Insufficient data for fuse calculation"
        return result
    
    result.volatility_expansion = calculate_volatility_expansion(bars)
    
    result.liquidity_score = calculate_liquidity_score(bars)
    
    result.trend_strength = calculate_trend_strength(bars)
    
    float_millions = 0.0
    if symbol_info:
        float_millions = getattr(symbol_info, 'float_millions', 0.0)
    
    if float_millions > 0 and len(bars) > 0:
        recent_volumes = [b.volume for b in bars[-20:]]
        avg_volume = float(np.mean(recent_volumes)) if recent_volumes else 0
        price = bars[-1].close
        result.float_pressure = calculate_float_pressure(float_millions, avg_volume, price)
    
    result.mr_bias = get_mr_protocol_bias(engine_result)
    
    weights = {
        'mr_bias': 0.30,
        'volatility': 0.25,
        'liquidity': 0.15,
        'float_pressure': 0.20,
        'trend': 0.10,
    }
    
    vol_score = min(result.volatility_expansion * 40, 100)
    trend_score = (result.trend_strength + 100) / 2
    
    fuse = (
        weights['mr_bias'] * result.mr_bias +
        weights['volatility'] * vol_score +
        weights['liquidity'] * result.liquidity_score +
        weights['float_pressure'] * result.float_pressure +
        weights['trend'] * trend_score
    )
    
    result.fuse_score = float(np.clip(fuse, 0, 100))
    
    if symbol_info:
        bucket = getattr(symbol_info, 'market_cap_bucket', 'unknown')
        if bucket in ['nano', 'penny']:
            result.risk_flags.append('extreme_smallcap')
        elif bucket in ['small', 'micro']:
            result.risk_flags.append('smallcap_volatility')
    
    if result.volatility_expansion > 2.0:
        result.risk_flags.append('high_volatility_expansion')
    
    if result.float_pressure > 70:
        result.risk_flags.append('high_float_turnover')
    
    if float_millions > 0 and float_millions < 10:
        result.risk_flags.append('very_low_float')
    
    if len(bars) > 0:
        recent_ranges = [(b.high - b.low) / b.close * 100 for b in bars[-5:] if b.close > 0]
        if recent_ranges and np.mean(recent_ranges) > 10:
            result.risk_flags.append('high_gap_risk')
    
    if result.fuse_score >= 75:
        result.runner_tier = 'primed'
        result.runner_candidate = True
    elif result.fuse_score >= 60:
        result.runner_tier = 'warming'
        result.runner_candidate = True
    elif result.fuse_score >= 45:
        result.runner_tier = 'watching'
        result.runner_candidate = False
    else:
        result.runner_tier = 'none'
        result.runner_candidate = False
    
    result.components = {
        'mr_bias': round(result.mr_bias, 2),
        'volatility_expansion': round(result.volatility_expansion, 4),
        'liquidity_score': round(result.liquidity_score, 2),
        'float_pressure': round(result.float_pressure, 2),
        'trend_strength': round(result.trend_strength, 2),
    }
    
    if result.runner_candidate:
        result.notes = f"Runner candidate ({result.runner_tier}): fuse={result.fuse_score:.1f}"
    else:
        result.notes = f"Not a runner candidate: fuse={result.fuse_score:.1f}"
    
    return result


def quick_fuse_check(bars: List[OhlcvBar]) -> float:
    """
    Quick fuse score check without full analysis.
    
    Args:
        bars: OHLCV price bars
        
    Returns:
        Approximate fuse score (0-100)
    """
    result = calculate_mr_fuse_score(bars)
    return result.fuse_score
