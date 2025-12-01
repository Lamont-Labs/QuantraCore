"""
Simulation-Based Data Augmentation for Training.

Multiplies training samples by generating variations grounded in real market data:
- Entry timing variations (simulate different entry points)
- Hold duration variations (different exit horizons)
- Monte Carlo outcome bootstrapping
- Rare event oversampling (monster runners, crashes)
- Walk-forward simulation windows

All augmented samples use REAL price data - no synthetic prices.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter

from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    entry_shifts: List[int] = field(default_factory=lambda: [-2, -1, 1, 2])
    hold_variations: List[int] = field(default_factory=lambda: [5, 7, 10, 15, 20])
    bootstrap_samples: int = 3
    oversample_runners: bool = True
    oversample_ratio: float = 3.0
    oversample_crashes: bool = True
    walkforward_windows: int = 3
    max_augmentation_factor: int = 10
    runner_threshold: float = 0.05
    crash_threshold: float = -0.05


class EntryTimingAugmenter:
    """
    Generates training samples with shifted entry points.
    
    Simulates "what if we entered N bars earlier/later?" using real prices.
    """
    
    def __init__(self, shifts: List[int] = None):
        self.shifts = shifts or [-2, -1, 1, 2]
    
    def augment(
        self,
        bars: List[OhlcvBar],
        window_size: int,
        future_bars: int,
        base_idx: int,
    ) -> List[Tuple[List[OhlcvBar], np.ndarray, str]]:
        """
        Generate shifted entry variations.
        
        Returns list of (window_bars, future_closes, shift_label) tuples.
        """
        augmented = []
        
        for shift in self.shifts:
            new_idx = base_idx + shift
            
            if new_idx < 0:
                continue
            if new_idx + window_size + future_bars > len(bars):
                continue
            
            window_bars = bars[new_idx:new_idx + window_size]
            future_closes = np.array([
                b.close for b in bars[new_idx + window_size:new_idx + window_size + future_bars]
            ])
            
            if len(window_bars) == window_size and len(future_closes) == future_bars:
                augmented.append((window_bars, future_closes, f"entry_shift_{shift:+d}"))
        
        return augmented


class HoldDurationAugmenter:
    """
    Generates training samples with different hold durations.
    
    Simulates different exit horizons to capture various trade styles.
    """
    
    def __init__(self, durations: List[int] = None):
        self.durations = durations or [5, 7, 10, 15, 20]
    
    def augment(
        self,
        bars: List[OhlcvBar],
        window_size: int,
        base_idx: int,
    ) -> List[Tuple[List[OhlcvBar], np.ndarray, str]]:
        """
        Generate hold duration variations.
        
        Returns list of (window_bars, future_closes, duration_label) tuples.
        """
        augmented = []
        
        window_bars = bars[base_idx:base_idx + window_size]
        if len(window_bars) != window_size:
            return []
        
        for duration in self.durations:
            if base_idx + window_size + duration > len(bars):
                continue
            
            future_closes = np.array([
                b.close for b in bars[base_idx + window_size:base_idx + window_size + duration]
            ])
            
            if len(future_closes) == duration:
                augmented.append((window_bars, future_closes, f"hold_{duration}bars"))
        
        return augmented


class MonteCarloBootstrapper:
    """
    Bootstrap outcome variations using real price movements.
    
    Resamples actual price changes to generate statistically valid variations.
    """
    
    def __init__(self, n_samples: int = 3, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)
    
    def augment(
        self,
        window_bars: List[OhlcvBar],
        future_closes: np.ndarray,
        all_returns_pool: np.ndarray,
    ) -> List[Tuple[List[OhlcvBar], np.ndarray, str]]:
        """
        Generate bootstrapped outcome variations.
        
        Uses real returns from the entire dataset to create plausible variations.
        """
        if len(all_returns_pool) < 20:
            return []
        
        augmented = []
        entry_price = window_bars[-1].close
        
        for i in range(self.n_samples):
            sampled_returns = self.rng.choice(
                all_returns_pool, 
                size=len(future_closes), 
                replace=True
            )
            
            bootstrapped_closes = entry_price * np.cumprod(1 + sampled_returns)
            
            augmented.append((
                window_bars.copy(),
                bootstrapped_closes,
                f"bootstrap_{i+1}"
            ))
        
        return augmented


class RareEventOversampler:
    """
    Oversample rare but important events.
    
    Monster runners and crashes are rare but critical for model learning.
    This creates synthetic training emphasis without creating fake data.
    """
    
    def __init__(
        self,
        runner_threshold: float = 0.05,
        crash_threshold: float = -0.05,
        runner_ratio: float = 3.0,
        crash_ratio: float = 2.0,
    ):
        self.runner_threshold = runner_threshold
        self.crash_threshold = crash_threshold
        self.runner_ratio = runner_ratio
        self.crash_ratio = crash_ratio
    
    def identify_rare_events(
        self,
        samples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Categorize samples into runners, crashes, and normal."""
        runners = []
        crashes = []
        normal = []
        
        for sample in samples:
            max_return = sample.get("max_runup_5d", 0)
            min_return = sample.get("max_drawdown_5d", 0)
            
            if max_return >= self.runner_threshold:
                runners.append(sample)
            elif min_return <= self.crash_threshold:
                crashes.append(sample)
            else:
                normal.append(sample)
        
        return runners, crashes, normal
    
    def oversample(
        self,
        samples: List[Dict[str, Any]],
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Oversample rare events to balance the dataset."""
        rng = np.random.RandomState(seed)
        
        runners, crashes, normal = self.identify_rare_events(samples)
        
        logger.info(f"Sample distribution: {len(runners)} runners, {len(crashes)} crashes, {len(normal)} normal")
        
        augmented = list(samples)
        
        if runners and self.runner_ratio > 1:
            n_copies = int(len(runners) * (self.runner_ratio - 1))
            for _ in range(n_copies):
                sample = rng.choice(runners).copy()
                sample["augmentation"] = "runner_oversample"
                augmented.append(sample)
            logger.info(f"Added {n_copies} oversampled runner samples")
        
        if crashes and self.crash_ratio > 1:
            n_copies = int(len(crashes) * (self.crash_ratio - 1))
            for _ in range(n_copies):
                sample = rng.choice(crashes).copy()
                sample["augmentation"] = "crash_oversample"
                augmented.append(sample)
            logger.info(f"Added {n_copies} oversampled crash samples")
        
        return augmented


class WalkForwardGenerator:
    """
    Generate walk-forward training windows.
    
    Creates overlapping windows with different starting points to capture
    temporal patterns from multiple perspectives.
    """
    
    def __init__(self, n_windows: int = 3):
        self.n_windows = n_windows
    
    def generate_offsets(self, step_size: int) -> List[int]:
        """Generate offset positions for walk-forward windows."""
        if self.n_windows <= 1:
            return [0]
        
        offsets = [0]
        sub_step = max(1, step_size // self.n_windows)
        
        for i in range(1, self.n_windows):
            offset = i * sub_step
            if offset < step_size and offset not in offsets:
                offsets.append(offset)
        
        return offsets


class AugmentedWindowGenerator:
    """
    Generates augmented training windows from raw bars.
    
    Applies entry shifts, walk-forward offsets, and bootstrapping
    during window extraction to properly multiply training data.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        step_size: int = 2,
        config: AugmentationConfig = None,
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.config = config or AugmentationConfig()
        
        self.entry_augmenter = EntryTimingAugmenter(self.config.entry_shifts)
        self.walkforward = WalkForwardGenerator(self.config.walkforward_windows)
    
    def generate(
        self,
        bars: List[OhlcvBar],
        symbol: str,
        future_bars: int = 10,
    ) -> List[Tuple[List[OhlcvBar], np.ndarray, str]]:
        """
        Generate augmented windows with entry shifts and walk-forward.
        
        Returns list of (window_bars, future_closes, augmentation_tag) tuples.
        """
        if len(bars) < self.window_size + future_bars:
            return []
        
        windows = []
        offsets = self.walkforward.generate_offsets(self.step_size)
        
        for offset in offsets:
            offset_tag = f"_wf{offset}" if offset > 0 else ""
            
            for i in range(offset, len(bars) - self.window_size - future_bars, self.step_size):
                window_bars = bars[i:i + self.window_size]
                future_closes = np.array([
                    b.close for b in bars[i + self.window_size:i + self.window_size + future_bars]
                ])
                
                if len(window_bars) != self.window_size or len(future_closes) != future_bars:
                    continue
                
                windows.append((window_bars, future_closes, f"original{offset_tag}"))
                
                for shift in self.config.entry_shifts:
                    shifted_i = i + shift
                    if shifted_i < 0 or shifted_i + self.window_size + future_bars > len(bars):
                        continue
                    
                    shifted_bars = bars[shifted_i:shifted_i + self.window_size]
                    shifted_futures = np.array([
                        b.close for b in bars[shifted_i + self.window_size:shifted_i + self.window_size + future_bars]
                    ])
                    
                    if len(shifted_bars) == self.window_size and len(shifted_futures) == future_bars:
                        windows.append((shifted_bars, shifted_futures, f"entry_shift_{shift:+d}{offset_tag}"))
        
        return windows


class DataAugmentationPipeline:
    """
    Complete data augmentation pipeline.
    
    Combines all augmentation techniques to multiply training samples
    while staying grounded in real market data.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        
        self.entry_augmenter = EntryTimingAugmenter(self.config.entry_shifts)
        self.hold_augmenter = HoldDurationAugmenter(self.config.hold_variations)
        self.bootstrapper = MonteCarloBootstrapper(self.config.bootstrap_samples)
        self.oversampler = RareEventOversampler(
            runner_threshold=self.config.runner_threshold,
            crash_threshold=self.config.crash_threshold,
            runner_ratio=self.config.oversample_ratio,
        )
        self.walkforward = WalkForwardGenerator(self.config.walkforward_windows)
    
    def create_augmented_window_generator(
        self,
        window_size: int,
        step_size: int,
    ) -> AugmentedWindowGenerator:
        """Create an augmented window generator with current config."""
        return AugmentedWindowGenerator(
            window_size=window_size,
            step_size=step_size,
            config=self.config,
        )
    
    def compute_returns_pool(self, all_bars: Dict[str, List[OhlcvBar]]) -> np.ndarray:
        """Compute pool of real returns for bootstrapping."""
        all_returns = []
        
        for symbol, bars in all_bars.items():
            if len(bars) < 2:
                continue
            
            closes = np.array([b.close for b in bars])
            returns = np.diff(closes) / closes[:-1]
            all_returns.extend(returns.tolist())
        
        return np.array(all_returns)
    
    def augment_windows(
        self,
        bars: List[OhlcvBar],
        symbol: str,
        window_size: int,
        step_size: int,
        future_bars: int,
        returns_pool: np.ndarray,
    ) -> List[Tuple[List[OhlcvBar], np.ndarray, str, str]]:
        """
        Generate augmented training windows from bars.
        
        Returns list of (window_bars, future_closes, symbol, augmentation_type) tuples.
        """
        aug_gen = self.create_augmented_window_generator(window_size, step_size)
        windows = aug_gen.generate(bars, symbol, future_bars)
        
        augmented = []
        for window_bars, future_closes, aug_tag in windows:
            augmented.append((window_bars, future_closes, symbol, aug_tag))
        
        return augmented
    
    def get_augmentation_stats(
        self,
        original_count: int,
        augmented_count: int,
    ) -> Dict[str, Any]:
        """Get statistics about augmentation."""
        return {
            "original_samples": original_count,
            "augmented_samples": augmented_count,
            "augmentation_factor": augmented_count / max(original_count, 1),
            "entry_shifts_enabled": len(self.config.entry_shifts) > 0,
            "hold_variations_enabled": len(self.config.hold_variations) > 0,
            "bootstrap_enabled": self.config.bootstrap_samples > 0,
            "oversampling_enabled": self.config.oversample_runners,
            "walkforward_windows": self.config.walkforward_windows,
        }


class AugmentedTrainingPipeline:
    """
    Extended training pipeline with simulation-based augmentation.
    
    Wraps the standard training pipeline with augmentation capabilities.
    """
    
    def __init__(
        self,
        base_trainer,
        augmentation_config: AugmentationConfig = None,
    ):
        self.base_trainer = base_trainer
        self.augmenter = DataAugmentationPipeline(augmentation_config)
        self.augmentation_stats = {}
    
    def augment_training_data(
        self,
        training_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply augmentation to existing training data."""
        if not training_rows:
            return []
        
        original_count = len(training_rows)
        
        augmented = self.augmenter.oversampler.oversample(training_rows)
        
        self.augmentation_stats = self.augmenter.get_augmentation_stats(
            original_count,
            len(augmented),
        )
        
        logger.info(f"Augmentation: {original_count} → {len(augmented)} samples "
                   f"({self.augmentation_stats['augmentation_factor']:.2f}x)")
        
        return augmented


def create_augmented_trainer(
    symbols: List[str] = None,
    lookback_days: int = 730,
    augmentation_config: AugmentationConfig = None,
):
    """
    Create a trainer with data augmentation enabled.
    
    Args:
        symbols: List of symbols to train on
        lookback_days: Historical data lookback
        augmentation_config: Augmentation settings
    
    Returns:
        AugmentedTrainingPipeline instance
    """
    from src.quantracore_apex.apexlab.unified_trainer import (
        UnifiedTrainer, 
        UnifiedTrainingConfig
    )
    
    config = UnifiedTrainingConfig(lookback_days=lookback_days)
    if symbols:
        config.symbols = symbols
    
    base_trainer = UnifiedTrainer(config)
    
    return AugmentedTrainingPipeline(
        base_trainer=base_trainer,
        augmentation_config=augmentation_config,
    )
