"""
Chaos Scenario Library for MarketSimulator.

Each scenario defines a specific extreme market condition with
realistic price dynamics, volume patterns, and timing characteristics.
"""

import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class ScenarioType(str, Enum):
    FLASH_CRASH = "flash_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    GAP_EVENT = "gap_event"
    LIQUIDITY_VOID = "liquidity_void"
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"
    SQUEEZE = "squeeze"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    BLACK_SWAN = "black_swan"
    NORMAL = "normal"
    CHOPPY = "choppy"


@dataclass
class ScenarioParameters:
    intensity: float = 1.0
    duration_bars: int = 100
    recovery_rate: float = 0.5
    randomness: float = 0.2
    extra: Dict[str, Any] = field(default_factory=dict)


class ChaosScenario(ABC):
    """Base class for all chaos scenarios."""
    
    def __init__(
        self,
        scenario_type: ScenarioType,
        name: str,
        description: str,
        default_params: Optional[ScenarioParameters] = None,
    ):
        self.scenario_type = scenario_type
        self.name = name
        self.description = description
        self.params = default_params or ScenarioParameters()
    
    @abstractmethod
    def generate_price_path(
        self,
        initial_price: float,
        num_bars: int,
        params: Optional[ScenarioParameters] = None,
    ) -> List[Dict[str, float]]:
        """
        Generate OHLCV price path for this scenario.
        
        Returns list of dicts with: open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def get_risk_multiplier(self) -> float:
        """
        Return risk multiplier for this scenario.
        Higher = more dangerous, used for position sizing.
        """
        pass
    
    def _add_noise(self, value: float, noise_pct: float = 0.02) -> float:
        """Add random noise to a value."""
        return value * (1 + random.uniform(-noise_pct, noise_pct))
    
    def _generate_volume(
        self, 
        base_volume: float, 
        volatility_factor: float = 1.0,
        panic_factor: float = 1.0,
    ) -> float:
        """Generate realistic volume with volatility correlation."""
        vol_boost = 1 + (volatility_factor - 1) * 2
        panic_boost = panic_factor ** 1.5
        noise = random.uniform(0.7, 1.3)
        return base_volume * vol_boost * panic_boost * noise


class FlashCrashScenario(ChaosScenario):
    """
    Simulates rapid market crash followed by recovery.
    
    Characteristics:
    - Sharp 10-30% drop in minutes
    - Volume explosion (3-10x normal)
    - V-shaped or L-shaped recovery
    - Possible dead cat bounces
    """
    
    def __init__(self):
        super().__init__(
            scenario_type=ScenarioType.FLASH_CRASH,
            name="Flash Crash",
            description="Rapid 10-30% drop with potential recovery",
            default_params=ScenarioParameters(
                intensity=1.0,
                duration_bars=50,
                recovery_rate=0.6,
                extra={"drop_pct": 0.15, "recovery_type": "v_shape"}
            )
        )
    
    def generate_price_path(
        self,
        initial_price: float,
        num_bars: int,
        params: Optional[ScenarioParameters] = None,
    ) -> List[Dict[str, float]]:
        p = params or self.params
        intensity = p.intensity
        drop_pct = p.extra.get("drop_pct", 0.15) * intensity
        recovery_type = p.extra.get("recovery_type", "v_shape")
        
        crash_phase = int(num_bars * 0.15)
        recovery_phase = int(num_bars * 0.5)
        consolidation_phase = num_bars - crash_phase - recovery_phase
        
        prices = []
        price = initial_price
        base_volume = 1_000_000
        
        for i in range(crash_phase):
            progress = i / crash_phase
            drop_rate = (1 - math.exp(-4 * progress)) * drop_pct
            target = initial_price * (1 - drop_rate)
            
            volatility = 0.02 + 0.05 * progress * intensity
            price = self._add_noise(target, volatility)
            
            high = price * (1 + random.uniform(0.005, 0.02))
            low = price * (1 - random.uniform(0.02, 0.08) * (1 + progress))
            open_price = prices[-1]["close"] if prices else initial_price
            
            volume = self._generate_volume(
                base_volume, 
                volatility_factor=1 + 3 * progress,
                panic_factor=1 + 5 * progress
            )
            
            prices.append({
                "open": open_price,
                "high": max(open_price, high, price),
                "low": min(open_price, low, price),
                "close": price,
                "volume": volume,
                "phase": "crash",
            })
        
        bottom_price = price
        recovery_target = initial_price * p.recovery_rate
        
        for i in range(recovery_phase):
            progress = i / recovery_phase
            
            if recovery_type == "v_shape":
                recovery = progress ** 0.7
            elif recovery_type == "l_shape":
                recovery = progress ** 2
            else:
                recovery = progress * (1 + 0.3 * math.sin(progress * 8))
            
            target = bottom_price + (recovery_target - bottom_price) * recovery
            volatility = 0.015 * (1 + (1 - progress) * intensity)
            price = self._add_noise(target, volatility)
            
            open_price = prices[-1]["close"]
            high = price * (1 + random.uniform(0.01, 0.04))
            low = price * (1 - random.uniform(0.01, 0.03))
            
            volume = self._generate_volume(
                base_volume,
                volatility_factor=1 + (1 - progress) * 2,
                panic_factor=1 + (1 - progress)
            )
            
            prices.append({
                "open": open_price,
                "high": max(open_price, high, price),
                "low": min(open_price, low, price),
                "close": price,
                "volume": volume,
                "phase": "recovery",
            })
        
        for i in range(consolidation_phase):
            drift = random.uniform(-0.005, 0.005)
            price = price * (1 + drift)
            
            open_price = prices[-1]["close"]
            high = price * (1 + random.uniform(0.005, 0.015))
            low = price * (1 - random.uniform(0.005, 0.015))
            
            volume = self._generate_volume(base_volume, volatility_factor=0.8)
            
            prices.append({
                "open": open_price,
                "high": max(open_price, high, price),
                "low": min(open_price, low, price),
                "close": price,
                "volume": volume,
                "phase": "consolidation",
            })
        
        return prices
    
    def get_risk_multiplier(self) -> float:
        return 3.0


class VolatilitySpikeScenario(ChaosScenario):
    """
    Simulates VIX explosion scenario.
    
    Characteristics:
    - Volatility expands 3-5x in short period
    - Wide intraday ranges
    - Whipsaw price action
    - Elevated volume throughout
    """
    
    def __init__(self):
        super().__init__(
            scenario_type=ScenarioType.VOLATILITY_SPIKE,
            name="Volatility Spike",
            description="VIX-style volatility explosion (3-5x normal)",
            default_params=ScenarioParameters(
                intensity=1.0,
                duration_bars=80,
                extra={"vol_multiplier": 4.0, "direction_bias": 0.0}
            )
        )
    
    def generate_price_path(
        self,
        initial_price: float,
        num_bars: int,
        params: Optional[ScenarioParameters] = None,
    ) -> List[Dict[str, float]]:
        p = params or self.params
        vol_mult = p.extra.get("vol_multiplier", 4.0) * p.intensity
        bias = p.extra.get("direction_bias", 0.0)
        
        prices = []
        price = initial_price
        base_volume = 1_000_000
        base_volatility = 0.01
        
        for i in range(num_bars):
            progress = i / num_bars
            vol_curve = math.sin(progress * math.pi) ** 0.5
            current_vol = base_volatility * (1 + (vol_mult - 1) * vol_curve)
            
            direction = random.choice([-1, 1])
            if random.random() < 0.3:
                direction *= -1
            
            move = direction * current_vol * random.uniform(0.5, 1.5)
            move += bias * 0.002
            price = price * (1 + move)
            
            open_price = prices[-1]["close"] if prices else initial_price
            
            range_mult = 1 + vol_curve * 3
            high = max(open_price, price) * (1 + random.uniform(0.005, 0.02) * range_mult)
            low = min(open_price, price) * (1 - random.uniform(0.005, 0.02) * range_mult)
            
            volume = self._generate_volume(
                base_volume,
                volatility_factor=1 + vol_curve * 3
            )
            
            prices.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
                "phase": "volatility_spike",
                "implied_vol": current_vol * 100,
            })
        
        return prices
    
    def get_risk_multiplier(self) -> float:
        return 2.5


class GapEventScenario(ChaosScenario):
    """
    Simulates overnight gap from news/earnings.
    
    Characteristics:
    - Large gap up or down (5-20%)
    - Initial volatility spike
    - Gap fill or continuation pattern
    - Pre-gap and post-gap behavior differs
    """
    
    def __init__(self):
        super().__init__(
            scenario_type=ScenarioType.GAP_EVENT,
            name="Gap Event",
            description="Overnight gap from news/earnings (5-20%)",
            default_params=ScenarioParameters(
                intensity=1.0,
                duration_bars=60,
                extra={
                    "gap_pct": 0.12,
                    "gap_direction": "random",
                    "fill_probability": 0.4,
                }
            )
        )
    
    def generate_price_path(
        self,
        initial_price: float,
        num_bars: int,
        params: Optional[ScenarioParameters] = None,
    ) -> List[Dict[str, float]]:
        p = params or self.params
        gap_pct = p.extra.get("gap_pct", 0.12) * p.intensity
        gap_dir = p.extra.get("gap_direction", "random")
        fill_prob = p.extra.get("fill_probability", 0.4)
        
        if gap_dir == "random":
            gap_dir = random.choice(["up", "down"])
        
        gap_mult = (1 + gap_pct) if gap_dir == "up" else (1 - gap_pct)
        will_fill = random.random() < fill_prob
        
        pre_gap_bars = int(num_bars * 0.2)
        post_gap_bars = num_bars - pre_gap_bars
        
        prices = []
        price = initial_price
        base_volume = 1_000_000
        
        for i in range(pre_gap_bars):
            drift = random.uniform(-0.005, 0.005)
            price = price * (1 + drift)
            
            open_price = prices[-1]["close"] if prices else initial_price
            high = price * (1 + random.uniform(0.003, 0.01))
            low = price * (1 - random.uniform(0.003, 0.01))
            
            volume = self._generate_volume(base_volume)
            
            prices.append({
                "open": open_price,
                "high": max(open_price, high, price),
                "low": min(open_price, low, price),
                "close": price,
                "volume": volume,
                "phase": "pre_gap",
            })
        
        pre_gap_close = price
        gap_price = pre_gap_close * gap_mult
        price = gap_price
        
        gap_open = gap_price
        first_bar_vol = 0.03 * p.intensity
        
        if gap_dir == "up":
            high = gap_price * (1 + random.uniform(0.01, 0.04))
            low = gap_price * (1 - random.uniform(0.01, first_bar_vol))
        else:
            high = gap_price * (1 + random.uniform(0.01, first_bar_vol))
            low = gap_price * (1 - random.uniform(0.01, 0.04))
        
        price = random.uniform(low, high)
        volume = self._generate_volume(base_volume, volatility_factor=4, panic_factor=3)
        
        prices.append({
            "open": gap_open,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
            "phase": "gap",
            "gap_pct": gap_pct * 100 * (1 if gap_dir == "up" else -1),
        })
        
        fill_target = pre_gap_close if will_fill else None
        
        for i in range(1, post_gap_bars):
            progress = i / post_gap_bars
            
            if will_fill and progress < 0.6:
                fill_progress = progress / 0.6
                target = gap_price + (fill_target - gap_price) * fill_progress * 0.8
                volatility = 0.015 * (1 + (1 - progress) * 2)
            else:
                drift = 0.003 if gap_dir == "up" else -0.003
                if will_fill:
                    drift *= -0.5
                target = price * (1 + drift + random.uniform(-0.01, 0.01))
                volatility = 0.01
            
            price = self._add_noise(target, volatility)
            open_price = prices[-1]["close"]
            high = max(open_price, price) * (1 + random.uniform(0.003, 0.015))
            low = min(open_price, price) * (1 - random.uniform(0.003, 0.015))
            
            volume_factor = 1 + (1 - progress) * 2
            volume = self._generate_volume(base_volume, volatility_factor=volume_factor)
            
            prices.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
                "phase": "post_gap",
            })
        
        return prices
    
    def get_risk_multiplier(self) -> float:
        return 2.0


class LiquidityVoidScenario(ChaosScenario):
    """
    Simulates thin order book / liquidity crisis.
    
    Characteristics:
    - Wide bid-ask spreads
    - Erratic price jumps
    - Volume dries up then spikes
    - Price can move significantly on small orders
    """
    
    def __init__(self):
        super().__init__(
            scenario_type=ScenarioType.LIQUIDITY_VOID,
            name="Liquidity Void",
            description="Thin order book with wide spreads and erratic moves",
            default_params=ScenarioParameters(
                intensity=1.0,
                duration_bars=70,
                extra={"spread_multiplier": 5.0, "volume_drought": 0.2}
            )
        )
    
    def generate_price_path(
        self,
        initial_price: float,
        num_bars: int,
        params: Optional[ScenarioParameters] = None,
    ) -> List[Dict[str, float]]:
        p = params or self.params
        spread_mult = p.extra.get("spread_multiplier", 5.0) * p.intensity
        vol_drought = p.extra.get("volume_drought", 0.2)
        
        prices = []
        price = initial_price
        base_volume = 1_000_000
        
        for i in range(num_bars):
            progress = i / num_bars
            liquidity_curve = math.sin(progress * math.pi) ** 0.7
            
            current_spread = 0.001 * spread_mult * (1 + liquidity_curve * 2)
            
            if random.random() < 0.15 * p.intensity:
                jump = random.choice([-1, 1]) * random.uniform(0.02, 0.05) * p.intensity
                price = price * (1 + jump)
            else:
                drift = random.uniform(-current_spread, current_spread)
                price = price * (1 + drift)
            
            open_price = prices[-1]["close"] if prices else initial_price
            
            high = max(open_price, price) * (1 + current_spread * random.uniform(0.5, 1.5))
            low = min(open_price, price) * (1 - current_spread * random.uniform(0.5, 1.5))
            
            volume_factor = vol_drought + (1 - vol_drought) * (1 - liquidity_curve)
            if random.random() < 0.1:
                volume_factor *= 3
            
            volume = self._generate_volume(base_volume) * volume_factor
            
            prices.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
                "phase": "liquidity_void",
                "spread": current_spread * 100,
            })
        
        return prices
    
    def get_risk_multiplier(self) -> float:
        return 2.8


class MomentumExhaustionScenario(ChaosScenario):
    """
    Simulates parabolic run-up followed by collapse.
    
    Characteristics:
    - Exponential price increase
    - Climactic volume at top
    - Sharp reversal
    - Dead cat bounces
    """
    
    def __init__(self):
        super().__init__(
            scenario_type=ScenarioType.MOMENTUM_EXHAUSTION,
            name="Momentum Exhaustion",
            description="Parabolic run-up then collapse",
            default_params=ScenarioParameters(
                intensity=1.0,
                duration_bars=100,
                extra={"run_up_pct": 0.50, "collapse_pct": 0.40}
            )
        )
    
    def generate_price_path(
        self,
        initial_price: float,
        num_bars: int,
        params: Optional[ScenarioParameters] = None,
    ) -> List[Dict[str, float]]:
        p = params or self.params
        run_up = p.extra.get("run_up_pct", 0.50) * p.intensity
        collapse = p.extra.get("collapse_pct", 0.40) * p.intensity
        
        run_up_bars = int(num_bars * 0.5)
        collapse_bars = int(num_bars * 0.3)
        bounce_bars = num_bars - run_up_bars - collapse_bars
        
        prices = []
        price = initial_price
        base_volume = 1_000_000
        
        for i in range(run_up_bars):
            progress = i / run_up_bars
            exp_factor = math.exp(progress * 2) - 1
            target_gain = run_up * (exp_factor / (math.e ** 2 - 1))
            
            target = initial_price * (1 + target_gain)
            volatility = 0.01 + 0.02 * progress
            price = self._add_noise(target, volatility)
            
            open_price = prices[-1]["close"] if prices else initial_price
            high = price * (1 + random.uniform(0.005, 0.03))
            low = min(open_price, price) * (1 - random.uniform(0.003, 0.015))
            
            volume = self._generate_volume(
                base_volume,
                volatility_factor=1 + progress * 3,
                panic_factor=1 + progress ** 2 * 2
            )
            
            prices.append({
                "open": open_price,
                "high": max(open_price, high),
                "low": low,
                "close": price,
                "volume": volume,
                "phase": "run_up",
            })
        
        peak_price = price
        
        for i in range(collapse_bars):
            progress = i / collapse_bars
            drop_curve = 1 - math.exp(-3 * progress)
            target_drop = collapse * drop_curve
            
            target = peak_price * (1 - target_drop)
            volatility = 0.02 + 0.03 * (1 - progress)
            price = self._add_noise(target, volatility)
            
            open_price = prices[-1]["close"]
            high = max(open_price, price) * (1 + random.uniform(0.005, 0.02))
            low = price * (1 - random.uniform(0.01, 0.04))
            
            volume = self._generate_volume(
                base_volume,
                volatility_factor=2 + (1 - progress) * 3,
                panic_factor=2 + (1 - progress) * 4
            )
            
            prices.append({
                "open": open_price,
                "high": high,
                "low": min(open_price, low),
                "close": price,
                "volume": volume,
                "phase": "collapse",
            })
        
        bottom_price = price
        
        for i in range(bounce_bars):
            progress = i / bounce_bars
            
            if i % 15 < 8:
                bounce = 0.05 * math.sin(progress * 4)
                target = bottom_price * (1 + bounce)
            else:
                target = bottom_price * (1 + random.uniform(-0.02, 0.02))
            
            price = self._add_noise(target, 0.015)
            
            open_price = prices[-1]["close"]
            high = max(open_price, price) * (1 + random.uniform(0.005, 0.02))
            low = min(open_price, price) * (1 - random.uniform(0.005, 0.02))
            
            volume = self._generate_volume(base_volume, volatility_factor=1.5)
            
            prices.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
                "phase": "dead_cat_bounce",
            })
        
        return prices
    
    def get_risk_multiplier(self) -> float:
        return 2.2


class SqueezeScenario(ChaosScenario):
    """
    Simulates short squeeze or gamma squeeze.
    
    Characteristics:
    - Rapid upward price acceleration
    - Volume explosion
    - Multiple legs up with brief consolidations
    - Eventually runs out of fuel
    """
    
    def __init__(self):
        super().__init__(
            scenario_type=ScenarioType.SQUEEZE,
            name="Squeeze",
            description="Short/gamma squeeze with multiple legs up",
            default_params=ScenarioParameters(
                intensity=1.0,
                duration_bars=60,
                extra={"squeeze_magnitude": 2.0, "num_legs": 3}
            )
        )
    
    def generate_price_path(
        self,
        initial_price: float,
        num_bars: int,
        params: Optional[ScenarioParameters] = None,
    ) -> List[Dict[str, float]]:
        p = params or self.params
        magnitude = p.extra.get("squeeze_magnitude", 2.0) * p.intensity
        num_legs = p.extra.get("num_legs", 3)
        
        leg_bars = num_bars // num_legs
        prices = []
        price = initial_price
        base_volume = 1_000_000
        
        for leg in range(num_legs):
            leg_gain = (magnitude - 1) / num_legs * (1 - leg * 0.2)
            squeeze_bars = int(leg_bars * 0.6)
            consolidation_bars = leg_bars - squeeze_bars
            
            leg_start_price = price
            
            for i in range(squeeze_bars):
                progress = i / squeeze_bars
                acceleration = math.exp(progress * 2) / math.e ** 2
                target_gain = leg_gain * acceleration
                
                target = leg_start_price * (1 + target_gain)
                volatility = 0.015 + 0.025 * progress
                price = self._add_noise(target, volatility)
                
                open_price = prices[-1]["close"] if prices else initial_price
                high = price * (1 + random.uniform(0.01, 0.05))
                low = min(open_price, price) * (1 - random.uniform(0.005, 0.02))
                
                volume = self._generate_volume(
                    base_volume,
                    volatility_factor=2 + progress * 4,
                    panic_factor=1 + progress * 3
                )
                
                prices.append({
                    "open": open_price,
                    "high": max(open_price, high),
                    "low": low,
                    "close": price,
                    "volume": volume,
                    "phase": f"squeeze_leg_{leg + 1}",
                })
            
            consolidation_base = price
            
            for i in range(consolidation_bars):
                drift = random.uniform(-0.02, 0.01)
                price = consolidation_base * (1 + drift + random.uniform(-0.01, 0.01))
                
                open_price = prices[-1]["close"]
                high = max(open_price, price) * (1 + random.uniform(0.005, 0.015))
                low = min(open_price, price) * (1 - random.uniform(0.005, 0.015))
                
                volume = self._generate_volume(base_volume, volatility_factor=1.5)
                
                prices.append({
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": price,
                    "volume": volume,
                    "phase": "consolidation",
                })
        
        return prices
    
    def get_risk_multiplier(self) -> float:
        return 2.5


class CorrelationBreakdownScenario(ChaosScenario):
    """
    Simulates correlation breakdown between normally correlated assets.
    
    For single-asset simulation, this manifests as erratic behavior
    where normal technical patterns fail.
    """
    
    def __init__(self):
        super().__init__(
            scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
            name="Correlation Breakdown",
            description="Technical patterns fail, erratic behavior",
            default_params=ScenarioParameters(
                intensity=1.0,
                duration_bars=80,
                extra={"pattern_failure_rate": 0.7}
            )
        )
    
    def generate_price_path(
        self,
        initial_price: float,
        num_bars: int,
        params: Optional[ScenarioParameters] = None,
    ) -> List[Dict[str, float]]:
        p = params or self.params
        failure_rate = p.extra.get("pattern_failure_rate", 0.7) * p.intensity
        
        prices = []
        price = initial_price
        base_volume = 1_000_000
        
        trend = random.choice([-1, 1])
        trend_strength = random.uniform(0.001, 0.003)
        
        for i in range(num_bars):
            if random.random() < failure_rate * 0.1:
                trend *= -1
                trend_strength = random.uniform(0.001, 0.003)
            
            if random.random() < failure_rate * 0.3:
                move = random.choice([-1, 1]) * random.uniform(0.01, 0.03)
            else:
                move = trend * trend_strength + random.uniform(-0.01, 0.01)
            
            price = price * (1 + move)
            
            open_price = prices[-1]["close"] if prices else initial_price
            volatility = 0.01 + random.uniform(0, 0.02) * p.intensity
            high = max(open_price, price) * (1 + volatility)
            low = min(open_price, price) * (1 - volatility)
            
            volume = self._generate_volume(
                base_volume,
                volatility_factor=1 + abs(move) * 50
            )
            
            prices.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
                "phase": "correlation_breakdown",
            })
        
        return prices
    
    def get_risk_multiplier(self) -> float:
        return 2.0


class BlackSwanScenario(ChaosScenario):
    """
    Simulates extreme multi-sigma events (COVID crash, 2008, etc.).
    
    Characteristics:
    - Unprecedented moves (5-10+ sigma)
    - Normal risk models completely fail
    - Cascading failures
    - Can last days to weeks
    """
    
    def __init__(self):
        super().__init__(
            scenario_type=ScenarioType.BLACK_SWAN,
            name="Black Swan",
            description="Multi-sigma catastrophic event (COVID, 2008-style)",
            default_params=ScenarioParameters(
                intensity=1.0,
                duration_bars=150,
                recovery_rate=0.4,
                extra={
                    "crash_pct": 0.35,
                    "num_waves": 3,
                    "capitulation_bar": 0.7,
                }
            )
        )
    
    def generate_price_path(
        self,
        initial_price: float,
        num_bars: int,
        params: Optional[ScenarioParameters] = None,
    ) -> List[Dict[str, float]]:
        p = params or self.params
        crash_pct = p.extra.get("crash_pct", 0.35) * p.intensity
        num_waves = p.extra.get("num_waves", 3)
        capitulation_pct = p.extra.get("capitulation_bar", 0.7)
        
        crash_bars = int(num_bars * 0.6)
        recovery_bars = num_bars - crash_bars
        wave_bars = crash_bars // num_waves
        
        prices = []
        price = initial_price
        base_volume = 1_000_000
        
        for wave in range(num_waves):
            wave_drop = crash_pct / num_waves * (1 + wave * 0.3)
            wave_start = price
            
            for i in range(wave_bars):
                progress = i / wave_bars
                
                if wave == num_waves - 1 and progress > 0.6:
                    drop_curve = math.exp((progress - 0.6) * 5) / math.e ** 2
                else:
                    drop_curve = progress ** 1.5
                
                target_drop = wave_drop * drop_curve
                target = wave_start * (1 - target_drop)
                
                volatility = 0.02 + 0.04 * progress * p.intensity
                if wave == num_waves - 1 and progress > 0.8:
                    volatility *= 2
                
                price = self._add_noise(target, volatility)
                
                open_price = prices[-1]["close"] if prices else initial_price
                
                high = max(open_price, price) * (1 + random.uniform(0.01, 0.03))
                low = price * (1 - random.uniform(0.02, 0.06) * (1 + progress))
                
                is_capitulation = (wave == num_waves - 1 and progress > 0.8)
                panic = 3 + wave * 2 + (5 if is_capitulation else 0)
                
                volume = self._generate_volume(
                    base_volume,
                    volatility_factor=2 + progress * 3,
                    panic_factor=panic
                )
                
                prices.append({
                    "open": open_price,
                    "high": high,
                    "low": min(open_price, low),
                    "close": price,
                    "volume": volume,
                    "phase": f"crash_wave_{wave + 1}",
                    "is_capitulation": is_capitulation,
                })
            
            if wave < num_waves - 1:
                relief_bars = int(wave_bars * 0.3)
                relief_target = price * 1.08
                
                for i in range(relief_bars):
                    progress = i / relief_bars
                    target = price + (relief_target - price) * progress * 0.5
                    price = self._add_noise(target, 0.015)
                    
                    open_price = prices[-1]["close"]
                    high = max(open_price, price) * (1 + random.uniform(0.01, 0.03))
                    low = min(open_price, price) * (1 - random.uniform(0.01, 0.02))
                    
                    volume = self._generate_volume(base_volume, volatility_factor=2)
                    
                    prices.append({
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": price,
                        "volume": volume,
                        "phase": "relief_rally",
                    })
        
        bottom_price = price
        recovery_target = initial_price * p.recovery_rate
        
        for i in range(recovery_bars):
            progress = i / recovery_bars
            recovery_curve = 1 - math.exp(-2 * progress)
            
            if random.random() < 0.15:
                setback = random.uniform(0.02, 0.05)
                target = price * (1 - setback)
            else:
                target = bottom_price + (recovery_target - bottom_price) * recovery_curve
            
            price = self._add_noise(target, 0.015)
            
            open_price = prices[-1]["close"]
            high = max(open_price, price) * (1 + random.uniform(0.008, 0.02))
            low = min(open_price, price) * (1 - random.uniform(0.008, 0.02))
            
            volume = self._generate_volume(
                base_volume,
                volatility_factor=1.5 + (1 - progress)
            )
            
            prices.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
                "phase": "recovery",
            })
        
        return prices
    
    def get_risk_multiplier(self) -> float:
        return 5.0


def get_scenario_by_type(scenario_type: ScenarioType) -> ChaosScenario:
    """Get scenario instance by type."""
    scenarios = {
        ScenarioType.FLASH_CRASH: FlashCrashScenario,
        ScenarioType.VOLATILITY_SPIKE: VolatilitySpikeScenario,
        ScenarioType.GAP_EVENT: GapEventScenario,
        ScenarioType.LIQUIDITY_VOID: LiquidityVoidScenario,
        ScenarioType.MOMENTUM_EXHAUSTION: MomentumExhaustionScenario,
        ScenarioType.SQUEEZE: SqueezeScenario,
        ScenarioType.CORRELATION_BREAKDOWN: CorrelationBreakdownScenario,
        ScenarioType.BLACK_SWAN: BlackSwanScenario,
    }
    
    if scenario_type not in scenarios:
        raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    return scenarios[scenario_type]()


def get_all_scenarios() -> List[ChaosScenario]:
    """Get instances of all chaos scenarios."""
    return [
        FlashCrashScenario(),
        VolatilitySpikeScenario(),
        GapEventScenario(),
        LiquidityVoidScenario(),
        MomentumExhaustionScenario(),
        SqueezeScenario(),
        CorrelationBreakdownScenario(),
        BlackSwanScenario(),
    ]


__all__ = [
    "ScenarioType",
    "ScenarioParameters",
    "ChaosScenario",
    "FlashCrashScenario",
    "VolatilitySpikeScenario",
    "GapEventScenario",
    "LiquidityVoidScenario",
    "MomentumExhaustionScenario",
    "SqueezeScenario",
    "CorrelationBreakdownScenario",
    "BlackSwanScenario",
    "get_scenario_by_type",
    "get_all_scenarios",
]
