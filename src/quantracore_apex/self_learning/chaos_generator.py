"""
Chaos Training Data Generator.

Generates labeled training samples from MarketSimulator chaos scenarios.
Each scenario run produces samples that feed into ApexLab for model training.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

from src.quantracore_apex.simulator.simulator import MarketSimulator, SimulationResult
from src.quantracore_apex.simulator.scenarios import ScenarioType
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow

logger = logging.getLogger(__name__)


@dataclass
class ChaosSample:
    """Training sample from chaos simulation."""
    sample_id: str
    symbol: str
    timestamp: str
    source: str = "chaos_simulator"
    scenario_type: str = ""
    intensity: float = 1.0
    quantra_score: float = 50.0
    risk_tier: str = "medium"
    regime: str = "unknown"
    entropy_state: str = "stable"
    suppression_state: str = "none"
    drift_state: str = "none"
    protocol_flags: List[str] = field(default_factory=list)
    omega_flags: List[str] = field(default_factory=list)
    max_runup_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    end_return_pct: float = 0.0
    total_volume: float = 0.0
    quality_tier: str = "C"
    hit_runner_threshold: int = 0
    hit_monster_runner_threshold: int = 0
    avoid_trade: int = 0
    risk_multiplier: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChaosTrainingGenerator:
    """
    Generates training data from chaos simulations.
    
    Flow:
    1. Run MarketSimulator scenarios
    2. Feed bars to ApexEngine
    3. Capture predictions + outcomes
    4. Label samples with quality/runner/avoid flags
    5. Save to ApexLab training data
    """
    
    SCENARIOS = [
        ScenarioType.FLASH_CRASH,
        ScenarioType.VOLATILITY_SPIKE,
        ScenarioType.GAP_EVENT,
        ScenarioType.LIQUIDITY_VOID,
        ScenarioType.MOMENTUM_EXHAUSTION,
        ScenarioType.SQUEEZE,
        ScenarioType.CORRELATION_BREAKDOWN,
        ScenarioType.BLACK_SWAN,
    ]
    
    SYMBOLS = ["SIM_AAPL", "SIM_NVDA", "SIM_TSLA", "SIM_SPY", "SIM_QQQ", 
               "SIM_AMZN", "SIM_GOOG", "SIM_META", "SIM_MSFT", "SIM_AMD"]
    
    INTENSITIES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    def __init__(
        self,
        output_path: str = "data/apexlab/chaos_simulation_samples.json",
        random_seed: Optional[int] = None
    ):
        self.output_path = Path(output_path)
        self.simulator = MarketSimulator(random_seed=random_seed)
        self.engine = ApexEngine(enable_logging=False)
        self._sample_count = 0
    
    def generate_batch(
        self,
        runs_per_combo: int = 2,
        symbols: Optional[List[str]] = None,
        scenarios: Optional[List[ScenarioType]] = None,
        intensities: Optional[List[float]] = None,
    ) -> List[ChaosSample]:
        """
        Generate a batch of training samples from chaos simulations.
        
        Args:
            runs_per_combo: Runs per scenario/symbol/intensity combination
            symbols: Override default symbols
            scenarios: Override default scenarios
            intensities: Override default intensities
            
        Returns:
            List of ChaosSample objects
        """
        symbols = symbols or self.SYMBOLS
        scenarios = scenarios or self.SCENARIOS
        intensities = intensities or self.INTENSITIES
        
        samples = []
        total_expected = len(scenarios) * len(symbols) * len(intensities) * runs_per_combo
        
        logger.info(f"Starting chaos generation: {total_expected} expected samples")
        
        for scenario in scenarios:
            for symbol in symbols:
                for intensity in intensities:
                    for run_idx in range(runs_per_combo):
                        sample = self._generate_single(
                            scenario, symbol, intensity, run_idx
                        )
                        if sample:
                            samples.append(sample)
                            self._sample_count += 1
                            
                            if self._sample_count % 100 == 0:
                                logger.info(f"Generated {self._sample_count} samples...")
        
        logger.info(f"Completed: {len(samples)} samples generated")
        return samples
    
    def _generate_single(
        self,
        scenario: ScenarioType,
        symbol: str,
        intensity: float,
        run_idx: int
    ) -> Optional[ChaosSample]:
        """Generate a single training sample."""
        seed = hash((scenario.value, symbol, intensity, run_idx)) % (2**31)
        
        try:
            initial_price = 100.0 + random.uniform(-20, 80)
            
            result = self.simulator.run_scenario(
                scenario_type=scenario,
                symbol=symbol,
                initial_price=initial_price,
                num_bars=100,
                intensity=intensity,
            )
            
            bars = [
                OhlcvBar(
                    timestamp=b.timestamp,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume
                )
                for b in result.bars
            ]
            
            if len(bars) < 50:
                return None
            
            apex_result = self.engine.run_scan(bars, symbol, seed=seed)
            
            quality_tier = self._compute_quality_tier(apex_result.quantrascore, result)
            hit_runner = 1 if result.max_runup_pct > 10 else 0
            hit_monster = 1 if result.max_runup_pct > 30 else 0
            avoid_trade = self._compute_avoid_trade(apex_result, result)
            
            sample_id = f"chaos_{scenario.value}_{symbol}_{intensity}_{run_idx}_{seed}"
            
            protocol_flags = []
            for p in apex_result.protocol_results:
                if hasattr(p, 'protocol_id') and hasattr(p, 'fired') and p.fired:
                    protocol_flags.append(p.protocol_id)
            
            omega_flags = []
            if isinstance(apex_result.omega_overrides, dict):
                omega_flags = [k for k, v in apex_result.omega_overrides.items() if v]
            
            return ChaosSample(
                sample_id=sample_id,
                symbol=symbol,
                timestamp=datetime.utcnow().isoformat(),
                source="chaos_simulator",
                scenario_type=scenario.value,
                intensity=intensity,
                quantra_score=apex_result.quantrascore,
                risk_tier=apex_result.risk_tier.value,
                regime=apex_result.regime.value,
                entropy_state=apex_result.entropy_state.value,
                suppression_state=apex_result.suppression_state.value,
                drift_state=apex_result.drift_state.value,
                protocol_flags=protocol_flags,
                omega_flags=omega_flags,
                max_runup_pct=result.max_runup_pct,
                max_drawdown_pct=result.max_drawdown_pct,
                end_return_pct=(result.end_price - result.start_price) / result.start_price * 100,
                total_volume=result.total_volume,
                quality_tier=quality_tier,
                hit_runner_threshold=hit_runner,
                hit_monster_runner_threshold=hit_monster,
                avoid_trade=avoid_trade,
                risk_multiplier=result.risk_multiplier,
            )
            
        except Exception as e:
            logger.warning(f"Error generating sample for {scenario.value}/{symbol}: {e}")
            return None
    
    def _compute_quality_tier(self, quantra_score: float, result: SimulationResult) -> str:
        """Compute quality tier from score and outcome."""
        profit_factor = result.max_runup_pct / max(result.max_drawdown_pct, 0.1)
        
        if quantra_score >= 75 and profit_factor > 2.0:
            return "A+"
        elif quantra_score >= 70 and profit_factor > 1.5:
            return "A"
        elif quantra_score >= 50 and profit_factor > 1.0:
            return "B"
        elif quantra_score >= 30:
            return "C"
        else:
            return "D"
    
    def _compute_avoid_trade(self, apex_result, result: SimulationResult) -> int:
        """Determine if trade should have been avoided."""
        if result.max_drawdown_pct > 15:
            return 1
        if apex_result.risk_tier.value == "extreme":
            return 1
        if isinstance(apex_result.omega_overrides, dict):
            if apex_result.omega_overrides.get("hard_lock", False):
                return 1
            if apex_result.omega_overrides.get("compliance_override", False):
                return 1
        if result.end_price < result.start_price * 0.85:
            return 1
        return 0
    
    def save_samples(self, samples: List[ChaosSample]) -> int:
        """
        Save samples to JSON file, merging with existing.
        
        Returns:
            Total sample count after save
        """
        existing_samples = []
        
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                existing_samples = json.load(f)
            logger.info(f"Found {len(existing_samples)} existing samples")
        
        new_sample_dicts = [s.to_dict() for s in samples]
        all_samples = existing_samples + new_sample_dicts
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(all_samples, f, indent=2, default=str)
        
        logger.info(f"Saved {len(all_samples)} total samples to {self.output_path}")
        return len(all_samples)
    
    def get_sample_stats(self) -> Dict[str, Any]:
        """Get statistics about generated samples."""
        if not self.output_path.exists():
            return {"total": 0, "by_scenario": {}, "by_quality": {}}
        
        with open(self.output_path, "r") as f:
            samples = json.load(f)
        
        by_scenario = {}
        by_quality = {}
        by_avoid = {"avoid": 0, "trade": 0}
        
        for s in samples:
            scenario = s.get("scenario_type", "unknown")
            by_scenario[scenario] = by_scenario.get(scenario, 0) + 1
            
            quality = s.get("quality_tier", "C")
            by_quality[quality] = by_quality.get(quality, 0) + 1
            
            if s.get("avoid_trade", 0) == 1:
                by_avoid["avoid"] += 1
            else:
                by_avoid["trade"] += 1
        
        return {
            "total": len(samples),
            "by_scenario": by_scenario,
            "by_quality": by_quality,
            "by_avoid": by_avoid,
        }
