"""
Alpaca Live Data Training Pipeline.

Fetches real market data from Alpaca and trains ApexCore models
on actual market patterns with proper outcome labels.

Benefits over Polygon:
- 200 req/min vs 5 req/min (40x faster)
- Already connected via paper trading credentials
- Primary data source for the system
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar, ApexContext
from src.quantracore_apex.apexlab.features import FeatureExtractor
from src.quantracore_apex.data_layer.adapters.alpaca_data_adapter import AlpacaDataAdapter
from src.quantracore_apex.prediction.apexcore_v3 import ApexCoreV3Model

logger = logging.getLogger(__name__)


@dataclass
class AlpacaTrainingConfig:
    """Configuration for Alpaca-based training."""
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "AMD", "NFLX", "SPY",
        "QQQ", "IWM", "DIA", "XLF", "XLE",
        "INTC", "CRM", "ORCL", "ADBE", "PYPL",
        "V", "MA", "JPM", "BAC", "WFC",
        "PFE", "JNJ", "UNH", "MRK", "ABBV",
    ])
    lookback_days: int = 365
    window_size: int = 100
    step_size: int = 5
    future_bars: int = 10
    runner_threshold: float = 0.05
    model_output_dir: str = "models/apexcore_v3"


class WindowGenerator:
    """Generates training windows from bar data."""
    
    def __init__(self, window_size: int = 100, step_size: int = 5):
        self.window_size = window_size
        self.step_size = step_size
    
    def generate(
        self,
        bars: List[OhlcvBar],
        symbol: str,
        future_bars: int = 10,
    ) -> List[Tuple[OhlcvWindow, np.ndarray]]:
        """Generate windows with future price data for labeling."""
        if len(bars) < self.window_size + future_bars:
            return []
        
        windows_with_futures = []
        
        for i in range(0, len(bars) - self.window_size - future_bars, self.step_size):
            window_bars = bars[i:i + self.window_size]
            future_closes = np.array([
                b.close for b in bars[i + self.window_size:i + self.window_size + future_bars]
            ])
            
            window = OhlcvWindow(
                symbol=symbol,
                bars=window_bars,
                timeframe="1d",
            )
            
            windows_with_futures.append((window, future_closes))
        
        return windows_with_futures


class OutcomeLabelGenerator:
    """Generates training labels from actual future price movements."""
    
    def __init__(self, runner_threshold: float = 0.05):
        self.runner_threshold = runner_threshold
    
    def generate_labels(
        self,
        entry_price: float,
        future_closes: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate outcome-based labels from what actually happened.
        
        This creates REAL labels based on actual price movements,
        not synthetic or engine-predicted values.
        """
        if len(future_closes) == 0:
            return self._default_labels()
        
        returns = (future_closes - entry_price) / entry_price
        
        final_return = returns[-1]
        max_return = np.max(returns)
        min_return = np.min(returns)
        max_drawdown = -min_return if min_return < 0 else 0
        
        if final_return > 0.02:
            regime_label = "trending_up"
        elif final_return < -0.02:
            regime_label = "trending_down"
        else:
            regime_label = "range_bound"
        
        if max_drawdown > 0.08 or final_return < -0.05:
            avoid_trade = 1
        else:
            avoid_trade = 0
        
        if max_return >= 0.10:
            quality_tier = "A+"
        elif max_return >= 0.05:
            quality_tier = "A"
        elif max_return >= 0.02:
            quality_tier = "B"
        elif max_return >= 0:
            quality_tier = "C"
        else:
            quality_tier = "D"
        
        reward_risk_ratio = max_return / (max_drawdown + 0.001)
        base_score = 50 + (final_return * 500)
        risk_penalty = max_drawdown * 300
        quantra_score = np.clip(base_score - risk_penalty + (reward_risk_ratio * 5), 0, 100)
        
        if final_return < -0.03 or max_drawdown > 0.08:
            quantra_score = min(quantra_score, 30)
        
        hit_runner = 1 if max_return >= self.runner_threshold else 0
        
        return {
            "quantra_score": float(quantra_score),
            "hit_runner_threshold": hit_runner,
            "future_quality_tier": quality_tier,
            "avoid_trade": avoid_trade,
            "regime_label": regime_label,
            "final_return": float(final_return),
            "max_return": float(max_return),
            "max_drawdown": float(max_drawdown),
            "ret_1d": float(returns[0]) if len(returns) > 0 else 0.0,
            "ret_3d": float(returns[2]) if len(returns) > 2 else 0.0,
            "ret_5d": float(returns[4]) if len(returns) > 4 else 0.0,
            "max_runup_5d": float(np.max(returns[:5])) if len(returns) >= 5 else float(max_return),
            "max_drawdown_5d": float(np.min(returns[:5])) if len(returns) >= 5 else float(min_return),
        }
    
    def _default_labels(self) -> Dict[str, Any]:
        return {
            "quantra_score": 50.0,
            "hit_runner_threshold": 0,
            "future_quality_tier": "C",
            "avoid_trade": 0,
            "regime_label": "range_bound",
            "final_return": 0.0,
            "max_return": 0.0,
            "max_drawdown": 0.0,
            "ret_1d": 0.0,
            "ret_3d": 0.0,
            "ret_5d": 0.0,
            "max_runup_5d": 0.0,
            "max_drawdown_5d": 0.0,
        }


class AlpacaLiveTrainer:
    """
    Complete training pipeline using real Alpaca market data.
    
    This trainer:
    1. Fetches historical OHLCV from Alpaca (40x faster than Polygon)
    2. Creates training windows with actual future outcomes
    3. Generates labels based on what really happened
    4. Trains ApexCore models on real market patterns
    """
    
    def __init__(self, config: Optional[AlpacaTrainingConfig] = None):
        self.config = config or AlpacaTrainingConfig()
        self.adapter = AlpacaDataAdapter()
        self.window_gen = WindowGenerator(
            window_size=self.config.window_size,
            step_size=self.config.step_size,
        )
        self.label_gen = OutcomeLabelGenerator(
            runner_threshold=self.config.runner_threshold,
        )
        self.feature_extractor = FeatureExtractor()
        self.engine = ApexEngine(enable_logging=False)
        
        self.training_rows: List[Dict[str, Any]] = []
        self.symbols_processed: List[str] = []
        self.total_bars_fetched = 0
    
    def fetch_all_data(self) -> Dict[str, List[OhlcvBar]]:
        """Fetch historical data for all symbols using Alpaca."""
        if not self.adapter.is_available():
            raise RuntimeError("Alpaca adapter not available - check API credentials")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.lookback_days)
        
        all_data = {}
        for i, symbol in enumerate(self.config.symbols):
            logger.info(f"[{i+1}/{len(self.config.symbols)}] Fetching {symbol}...")
            
            try:
                bars = self.adapter.fetch_ohlcv(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    timeframe="1d"
                )
                
                if bars:
                    all_data[symbol] = bars
                    self.total_bars_fetched += len(bars)
                    logger.info(f"  Got {len(bars)} bars for {symbol}")
                else:
                    logger.warning(f"  No data for {symbol}")
                    
            except Exception as e:
                logger.error(f"  Error fetching {symbol}: {e}")
                continue
        
        return all_data
    
    def generate_training_samples(
        self,
        all_data: Dict[str, List[OhlcvBar]],
    ) -> int:
        """Generate training samples from fetched data."""
        samples_generated = 0
        
        for symbol, bars in all_data.items():
            windows_with_futures = self.window_gen.generate(
                bars=bars,
                symbol=symbol,
                future_bars=self.config.future_bars,
            )
            
            for window, future_closes in windows_with_futures:
                entry_price = window.bars[-1].close
                
                outcome_labels = self.label_gen.generate_labels(
                    entry_price=entry_price,
                    future_closes=future_closes,
                )
                
                try:
                    context = ApexContext(seed=42, compliance_mode=True)
                    apex_result = self.engine.run(window, context)
                    
                    features = self.feature_extractor.extract(window)
                    
                    row = {
                        "symbol": symbol,
                        "timestamp": window.bars[-1].timestamp.isoformat(),
                        "entry_price": entry_price,
                        "features": features.tolist() if hasattr(features, 'tolist') else list(features),
                        "quantra_score": outcome_labels["quantra_score"],
                        "hit_runner_threshold": outcome_labels["hit_runner_threshold"],
                        "future_quality_tier": outcome_labels["future_quality_tier"],
                        "avoid_trade": outcome_labels["avoid_trade"],
                        "regime_label": outcome_labels["regime_label"],
                        "ret_1d": outcome_labels["ret_1d"],
                        "ret_3d": outcome_labels["ret_3d"],
                        "ret_5d": outcome_labels["ret_5d"],
                        "max_runup_5d": outcome_labels["max_runup_5d"],
                        "max_drawdown_5d": outcome_labels["max_drawdown_5d"],
                        "engine_score": apex_result.quantrascore,
                        "engine_regime": apex_result.regime.value,
                        "vix_level": 20.0,
                        "vix_percentile": 50.0,
                        "sector_momentum": 0.0,
                        "market_breadth": 0.5,
                    }
                    
                    self.training_rows.append(row)
                    samples_generated += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing window for {symbol}: {e}")
                    continue
            
            self.symbols_processed.append(symbol)
            logger.info(f"  Generated {samples_generated} samples so far ({symbol} done)")
        
        return samples_generated
    
    def train_model(self, model_size: str = "big") -> Dict[str, Any]:
        """Train ApexCore V3 model on the collected samples."""
        if not self.training_rows:
            raise ValueError("No training samples available - run generate_training_samples first")
        
        logger.info(f"Training ApexCore V3 on {len(self.training_rows)} real market samples...")
        
        model = ApexCoreV3Model(model_size=model_size)
        metrics = model.fit(self.training_rows)
        
        model_dir = Path(self.config.model_output_dir) / model_size
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir))
        
        manifest = {
            "version": "3.0.0",
            "model_size": model_size,
            "trained_at": datetime.utcnow().isoformat(),
            "training_samples": len(self.training_rows),
            "symbols_trained": len(self.symbols_processed),
            "symbols": self.symbols_processed,
            "total_bars_fetched": self.total_bars_fetched,
            "data_source": "alpaca",
            "lookback_days": self.config.lookback_days,
            "metrics": metrics,
            "training_note": "Trained on real Alpaca market data with actual outcome labels",
        }
        
        import json
        manifest_path = model_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
        logger.info(f"Training metrics: {metrics}")
        
        return manifest
    
    def run_full_pipeline(self, model_size: str = "big") -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("=" * 60)
        logger.info("ALPACA LIVE DATA TRAINING PIPELINE")
        logger.info("=" * 60)
        
        logger.info(f"Config: {len(self.config.symbols)} symbols, {self.config.lookback_days} days lookback")
        
        logger.info("\n[1/3] Fetching historical data from Alpaca...")
        all_data = self.fetch_all_data()
        logger.info(f"Fetched data for {len(all_data)} symbols, {self.total_bars_fetched} total bars")
        
        logger.info("\n[2/3] Generating training samples with real outcomes...")
        n_samples = self.generate_training_samples(all_data)
        logger.info(f"Generated {n_samples} training samples")
        
        logger.info("\n[3/3] Training ApexCore V3 model...")
        manifest = self.train_model(model_size)
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Samples: {manifest['training_samples']}")
        logger.info(f"  Symbols: {manifest['symbols_trained']}")
        logger.info(f"  Data source: Alpaca (real market data)")
        logger.info("=" * 60)
        
        return manifest


def run_alpaca_training(
    symbols: Optional[List[str]] = None,
    lookback_days: int = 365,
    model_size: str = "big",
) -> Dict[str, Any]:
    """
    Convenience function to run Alpaca-based training.
    
    Args:
        symbols: List of symbols to train on (default: 30 major stocks)
        lookback_days: Days of historical data (default: 365)
        model_size: Model variant ("big" or "mini")
        
    Returns:
        Training manifest with metrics
    """
    config = AlpacaTrainingConfig(
        lookback_days=lookback_days,
    )
    if symbols:
        config.symbols = symbols
    
    trainer = AlpacaLiveTrainer(config)
    return trainer.run_full_pipeline(model_size)
