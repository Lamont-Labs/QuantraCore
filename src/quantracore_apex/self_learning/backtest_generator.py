"""
Backtest Training Data Generator.

Generates labeled training samples from historical data backtesting.
Processes historical OHLCV data through ApexEngine and labels outcomes.
"""

import json
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter

logger = logging.getLogger(__name__)


@dataclass
class BacktestSample:
    """Training sample from historical backtest."""
    sample_id: str
    symbol: str
    timestamp: str
    source: str = "backtest"
    timeframe: str = "1d"
    quantra_score: float = 50.0
    risk_tier: str = "medium"
    regime: str = "unknown"
    entropy_state: str = "stable"
    suppression_state: str = "none"
    drift_state: str = "none"
    protocol_flags: List[str] = field(default_factory=list)
    omega_flags: List[str] = field(default_factory=list)
    ret_1d: float = 0.0
    ret_3d: float = 0.0
    ret_5d: float = 0.0
    ret_10d: float = 0.0
    max_runup_5d: float = 0.0
    max_drawdown_5d: float = 0.0
    quality_tier: str = "C"
    hit_runner_threshold: int = 0
    hit_monster_runner_threshold: int = 0
    avoid_trade: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BacktestTrainingGenerator:
    """
    Generates training data from historical backtesting.
    
    Flow:
    1. Load historical OHLCV data
    2. Slide 100-bar windows across history
    3. Run ApexEngine on each window
    4. Compute future returns (1d, 3d, 5d, 10d)
    5. Label with quality tier, runner flags, avoid flags
    6. Save to ApexLab training data
    """
    
    SYMBOLS = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "AMZN", "GOOG", "META", "MSFT", "AMD",
               "NFLX", "CRM", "PYPL", "INTC", "CSCO", "ORCL", "IBM", "ADBE", "NOW", "SNOW"]
    
    def __init__(
        self,
        output_path: str = "data/apexlab/backtest_samples.json",
        use_synthetic: bool = True
    ):
        self.output_path = Path(output_path)
        self.engine = ApexEngine(enable_logging=False)
        self.use_synthetic = use_synthetic
        
        self.data_adapter = SyntheticAdapter()
        
        self._sample_count = 0
    
    def generate_batch(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        window_size: int = 100,
        step_size: int = 5,
        future_bars: int = 10,
    ) -> List[BacktestSample]:
        """
        Generate training samples from historical backtest.
        
        Args:
            symbols: Symbols to process
            start_date: Start date for historical data
            end_date: End date for historical data
            window_size: Size of analysis window
            step_size: Bars to step between windows
            future_bars: Number of future bars for labeling
            
        Returns:
            List of BacktestSample objects
        """
        symbols = symbols or self.SYMBOLS
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now() - timedelta(days=future_bars + 1)
        
        samples = []
        
        logger.info(f"Starting backtest generation for {len(symbols)} symbols")
        
        for symbol in symbols:
            symbol_samples = self._process_symbol(
                symbol, start_date, end_date, window_size, step_size, future_bars
            )
            samples.extend(symbol_samples)
            logger.info(f"Generated {len(symbol_samples)} samples for {symbol}")
        
        logger.info(f"Completed: {len(samples)} total samples")
        return samples
    
    def _process_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        window_size: int,
        step_size: int,
        future_bars: int,
    ) -> List[BacktestSample]:
        """Process a single symbol."""
        samples = []
        
        try:
            extended_end = end_date + timedelta(days=future_bars * 2)
            bars = self.data_adapter.fetch_ohlcv(
                symbol=symbol,
                start=start_date,
                end=extended_end,
                timeframe="1d"
            )
            
            if len(bars) < window_size + future_bars:
                logger.warning(f"Insufficient data for {symbol}")
                return []
            
            for i in range(0, len(bars) - window_size - future_bars, step_size):
                window_bars = bars[i:i + window_size]
                future_bars_data = bars[i + window_size:i + window_size + future_bars]
                
                sample = self._create_sample(symbol, window_bars, future_bars_data, i)
                if sample:
                    samples.append(sample)
                    self._sample_count += 1
                    
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")
        
        return samples
    
    def _create_sample(
        self,
        symbol: str,
        window_bars: List[OhlcvBar],
        future_bars: List[OhlcvBar],
        index: int
    ) -> Optional[BacktestSample]:
        """Create a single backtest sample."""
        try:
            apex_result = self.engine.run_scan(window_bars, symbol, seed=42 + index)
            
            entry_price = window_bars[-1].close
            future_closes = [b.close for b in future_bars]
            
            if len(future_closes) < 5:
                return None
            
            ret_1d = (future_closes[0] - entry_price) / entry_price * 100
            ret_3d = (future_closes[min(2, len(future_closes)-1)] - entry_price) / entry_price * 100
            ret_5d = (future_closes[min(4, len(future_closes)-1)] - entry_price) / entry_price * 100
            ret_10d = (future_closes[-1] - entry_price) / entry_price * 100
            
            max_future_price = max(future_closes[:5])
            min_future_price = min(future_closes[:5])
            
            max_runup_5d = (max_future_price - entry_price) / entry_price * 100
            max_drawdown_5d = (entry_price - min_future_price) / entry_price * 100
            
            quality_tier = self._compute_quality_tier(max_runup_5d, max_drawdown_5d, ret_5d)
            hit_runner = 1 if max_runup_5d > 10 else 0
            hit_monster = 1 if max_runup_5d > 30 else 0
            avoid_trade = self._compute_avoid_trade(apex_result, max_drawdown_5d)
            
            protocol_flags = []
            for p in apex_result.protocol_results:
                if hasattr(p, 'protocol_id') and hasattr(p, 'fired') and p.fired:
                    protocol_flags.append(p.protocol_id)
            
            omega_flags = []
            if isinstance(apex_result.omega_overrides, dict):
                omega_flags = [k for k, v in apex_result.omega_overrides.items() if v]
            
            sample_id = f"backtest_{symbol}_{index}_{window_bars[-1].timestamp.isoformat()}"
            
            return BacktestSample(
                sample_id=sample_id,
                symbol=symbol,
                timestamp=window_bars[-1].timestamp.isoformat(),
                source="backtest",
                timeframe="1d",
                quantra_score=apex_result.quantrascore,
                risk_tier=apex_result.risk_tier.value,
                regime=apex_result.regime.value,
                entropy_state=apex_result.entropy_state.value,
                suppression_state=apex_result.suppression_state.value,
                drift_state=apex_result.drift_state.value,
                protocol_flags=protocol_flags,
                omega_flags=omega_flags,
                ret_1d=ret_1d,
                ret_3d=ret_3d,
                ret_5d=ret_5d,
                ret_10d=ret_10d,
                max_runup_5d=max_runup_5d,
                max_drawdown_5d=max_drawdown_5d,
                quality_tier=quality_tier,
                hit_runner_threshold=hit_runner,
                hit_monster_runner_threshold=hit_monster,
                avoid_trade=avoid_trade,
            )
            
        except Exception as e:
            logger.warning(f"Error creating sample: {e}")
            return None
    
    def _compute_quality_tier(self, max_runup: float, max_drawdown: float, ret_5d: float) -> str:
        """Compute quality tier from outcomes."""
        profit_factor = max_runup / max(max_drawdown, 0.1)
        
        if ret_5d > 5 and profit_factor > 2.0:
            return "A+"
        elif ret_5d > 2 and profit_factor > 1.5:
            return "A"
        elif ret_5d > 0 and profit_factor > 1.0:
            return "B"
        elif ret_5d > -2:
            return "C"
        else:
            return "D"
    
    def _compute_avoid_trade(self, apex_result, max_drawdown: float) -> int:
        """Determine if trade should have been avoided."""
        if max_drawdown > 10:
            return 1
        if apex_result.risk_tier.value == "extreme":
            return 1
        if apex_result.risk_tier.value == "high" and max_drawdown > 5:
            return 1
        if isinstance(apex_result.omega_overrides, dict):
            if apex_result.omega_overrides.get("hard_lock", False):
                return 1
            if apex_result.omega_overrides.get("compliance_override", False):
                return 1
        return 0
    
    def save_samples(self, samples: List[BacktestSample]) -> int:
        """Save samples to JSON file, merging with existing."""
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
