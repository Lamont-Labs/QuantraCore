"""
Polygon Live Data Training Pipeline.

Fetches real market data from Polygon.io and trains ApexCore models
on actual market patterns with proper outcome labels.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import requests

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar, ApexContext
from src.quantracore_apex.apexlab.features import FeatureExtractor
from src.quantracore_apex.apexcore.models import ApexCoreFull

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for live data training."""
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "AMD", "NFLX", "SPY",
        "QQQ", "IWM", "DIA", "XLF", "XLE"
    ])
    lookback_days: int = 365
    window_size: int = 100
    step_size: int = 5
    future_bars: int = 10
    runner_threshold: float = 0.05
    model_output_dir: str = "data/models"
    

class PolygonDataFetcher:
    """Fetches historical OHLCV data from Polygon.io."""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not set")
    
    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        max_retries: int = 3,
    ) -> List[OhlcvBar]:
        """Fetch daily OHLCV bars from Polygon with retry logic."""
        import time
        
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=30)
                
                if resp.status_code == 429:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Rate limited on {symbol}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                if "results" not in data or not data["results"]:
                    logger.warning(f"No data for {symbol}")
                    return []
                
                bars = []
                for r in data["results"]:
                    bar = OhlcvBar(
                        timestamp=datetime.fromtimestamp(r["t"] / 1000),
                        open=float(r["o"]),
                        high=float(r["h"]),
                        low=float(r["l"]),
                        close=float(r["c"]),
                        volume=float(r["v"]),
                    )
                    bars.append(bar)
                
                return bars
                
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429 and attempt < max_retries - 1:
                    continue
                logger.error(f"HTTP error fetching {symbol}: {e}")
                return []
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                wait_time = (attempt + 1) * 2
                if attempt < max_retries - 1:
                    logger.warning(f"Network error on {symbol}, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Network error fetching {symbol} after {max_retries} attempts: {e}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error fetching {symbol}: {e}")
                return []
        
        return []


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
        """
        Generate windows with future price data for labeling.
        
        Returns:
            List of (window, future_closes) tuples
        """
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
    """Generates training labels from future price movements."""
    
    def __init__(self, runner_threshold: float = 0.05):
        self.runner_threshold = runner_threshold
    
    def generate_labels(
        self,
        entry_price: float,
        future_closes: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate outcome-based labels from future prices.
        
        Labels generated:
        - regime_class: Direction based on final return
        - risk_class: Based on max drawdown
        - quantrascore_target: Based on risk-adjusted return
        - is_runner: Whether it hit runner threshold
        """
        if len(future_closes) == 0:
            return self._default_labels()
        
        returns = (future_closes - entry_price) / entry_price
        
        final_return = returns[-1]
        max_return = np.max(returns)
        min_return = np.min(returns)
        max_drawdown = -min_return if min_return < 0 else 0
        
        if final_return > 0.02:
            regime_class = 0
        elif final_return < -0.02:
            regime_class = 1
        else:
            regime_class = 2
        
        if max_drawdown < 0.02:
            risk_class = 0
        elif max_drawdown < 0.05:
            risk_class = 1
        else:
            risk_class = 2
        
        reward_risk_ratio = max_return / (max_drawdown + 0.001)
        base_score = 50 + (final_return * 500)
        risk_penalty = max_drawdown * 300
        quantrascore_target = np.clip(base_score - risk_penalty, 0, 100)
        
        if final_return < -0.03 or max_drawdown > 0.08:
            quantrascore_target = min(quantrascore_target, 30)
        
        is_runner = max_return >= self.runner_threshold
        
        return {
            "regime_class": regime_class,
            "risk_class": risk_class,
            "quantrascore_target": float(quantrascore_target),
            "is_runner": int(is_runner),
            "final_return": float(final_return),
            "max_return": float(max_return),
            "max_drawdown": float(max_drawdown),
        }
    
    def _default_labels(self) -> Dict[str, Any]:
        return {
            "regime_class": 2,
            "risk_class": 1,
            "quantrascore_target": 50.0,
            "is_runner": 0,
            "final_return": 0.0,
            "max_return": 0.0,
            "max_drawdown": 0.0,
        }


class LiveDataTrainer:
    """
    Complete training pipeline using real Polygon data.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.fetcher = PolygonDataFetcher()
        self.window_gen = WindowGenerator(
            window_size=self.config.window_size,
            step_size=self.config.step_size,
        )
        self.label_gen = OutcomeLabelGenerator(
            runner_threshold=self.config.runner_threshold,
        )
        self.feature_extractor = FeatureExtractor()
        self.engine = ApexEngine(enable_logging=False)
        
        self.model = None
        
        self.training_data = {
            "features": [],
            "regime_labels": [],
            "risk_labels": [],
            "score_labels": [],
            "metadata": [],
        }
    
    def fetch_all_data(self) -> Dict[str, List[OhlcvBar]]:
        """Fetch historical data for all symbols with rate limiting."""
        import time
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.lookback_days)
        
        all_data = {}
        for i, symbol in enumerate(self.config.symbols):
            logger.info(f"Fetching {symbol}...")
            bars = self.fetcher.fetch_daily_bars(symbol, start_date, end_date)
            if bars:
                all_data[symbol] = bars
                logger.info(f"  Got {len(bars)} bars")
            else:
                logger.warning(f"  No data for {symbol}")
            
            if i < len(self.config.symbols) - 1:
                time.sleep(0.25)
        
        return all_data
    
    def generate_training_samples(
        self,
        all_data: Dict[str, List[OhlcvBar]],
    ) -> int:
        """Generate training samples from all symbols."""
        total_samples = 0
        
        for symbol, bars in all_data.items():
            windows_with_futures = self.window_gen.generate(
                bars, symbol, self.config.future_bars
            )
            
            for window, future_closes in windows_with_futures:
                try:
                    features = self.feature_extractor.extract(window)
                    
                    entry_price = window.bars[-1].close
                    labels = self.label_gen.generate_labels(entry_price, future_closes)
                    
                    self.training_data["features"].append(features)
                    self.training_data["regime_labels"].append(labels["regime_class"])
                    self.training_data["risk_labels"].append(labels["risk_class"])
                    self.training_data["score_labels"].append(labels["quantrascore_target"])
                    self.training_data["metadata"].append({
                        "symbol": symbol,
                        "timestamp": window.bars[-1].timestamp.isoformat(),
                        "entry_price": entry_price,
                        "final_return": labels["final_return"],
                        "is_runner": labels["is_runner"],
                    })
                    
                    total_samples += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing window for {symbol}: {e}")
                    continue
        
        return total_samples
    
    def train_model(self) -> Dict[str, Any]:
        """Train the ApexCore model on collected data."""
        if not self.training_data["features"]:
            return {"error": "No training data collected"}
        
        X = np.array(self.training_data["features"])
        y_regime = np.array(self.training_data["regime_labels"])
        y_risk = np.array(self.training_data["risk_labels"])
        y_score = np.array(self.training_data["score_labels"])
        
        logger.info(f"Training on {len(X)} samples...")
        
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler
        
        use_early_stopping = len(X) >= 100
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        regime_classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=use_early_stopping,
            validation_fraction=0.1 if use_early_stopping else 0.0,
        )
        regime_classifier.fit(X_scaled, y_regime)
        
        score_regressor = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=use_early_stopping,
            validation_fraction=0.1 if use_early_stopping else 0.0,
        )
        score_regressor.fit(X_scaled, y_score)
        
        risk_classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=use_early_stopping,
            validation_fraction=0.1 if use_early_stopping else 0.0,
        )
        risk_classifier.fit(X_scaled, y_risk)
        
        self.model = ApexCoreFull()
        self.model.scaler = scaler
        self.model.regime_classifier = regime_classifier
        self.model.score_regressor = score_regressor
        self.model.risk_classifier = risk_classifier
        self.model.is_trained = True
        
        regime_acc = regime_classifier.score(X_scaled, y_regime)
        risk_acc = risk_classifier.score(X_scaled, y_risk)
        score_preds = score_regressor.predict(X_scaled)
        score_mae = float(np.mean(np.abs(score_preds - y_score)))
        
        metrics = {
            "regime_accuracy": round(regime_acc, 4),
            "risk_accuracy": round(risk_acc, 4),
            "score_mae": round(score_mae, 4),
            "samples": len(X),
        }
        
        output_dir = Path(self.config.model_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f"apexcore_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.model.save(str(model_path))
        
        latest_path = output_dir / "apexcore_latest.pkl"
        self.model.save(str(latest_path))
        
        regime_distribution = {
            "bullish": int(np.sum(y_regime == 0)),
            "bearish": int(np.sum(y_regime == 1)),
            "neutral": int(np.sum(y_regime == 2)),
        }
        
        risk_distribution = {
            "low": int(np.sum(y_risk == 0)),
            "medium": int(np.sum(y_risk == 1)),
            "high": int(np.sum(y_risk == 2)),
        }
        
        runner_count = sum(1 for m in self.training_data["metadata"] if m["is_runner"])
        
        return {
            "samples": len(X),
            "symbols": len(set(m["symbol"] for m in self.training_data["metadata"])),
            "metrics": metrics,
            "model_path": str(model_path),
            "latest_path": str(latest_path),
            "regime_distribution": regime_distribution,
            "risk_distribution": risk_distribution,
            "runner_rate": runner_count / len(X) if X.size > 0 else 0,
            "score_stats": {
                "mean": float(np.mean(y_score)),
                "std": float(np.std(y_score)),
                "min": float(np.min(y_score)),
                "max": float(np.max(y_score)),
            },
        }
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline:
        1. Fetch data from Polygon
        2. Generate windows and labels
        3. Train model
        4. Save and return results
        """
        logger.info("=" * 60)
        logger.info("APEXCORE LIVE DATA TRAINING PIPELINE")
        logger.info("=" * 60)
        
        logger.info(f"Configuration:")
        logger.info(f"  Symbols: {len(self.config.symbols)}")
        logger.info(f"  Lookback: {self.config.lookback_days} days")
        logger.info(f"  Window size: {self.config.window_size} bars")
        logger.info(f"  Future bars: {self.config.future_bars}")
        logger.info("")
        
        logger.info("Step 1: Fetching data from Polygon.io...")
        all_data = self.fetch_all_data()
        total_bars = sum(len(bars) for bars in all_data.values())
        logger.info(f"  Fetched {total_bars} total bars from {len(all_data)} symbols")
        logger.info("")
        
        logger.info("Step 2: Generating training samples...")
        n_samples = self.generate_training_samples(all_data)
        logger.info(f"  Generated {n_samples} training samples")
        logger.info("")
        
        logger.info("Step 3: Training ApexCore model...")
        results = self.train_model()
        logger.info(f"  Training complete!")
        logger.info(f"  Regime accuracy: {results['metrics']['regime_accuracy']:.1%}")
        logger.info(f"  Risk accuracy: {results['metrics']['risk_accuracy']:.1%}")
        logger.info(f"  Score MAE: {results['metrics']['score_mae']:.2f}")
        logger.info("")
        
        logger.info("Step 4: Model saved!")
        logger.info(f"  Path: {results['model_path']}")
        logger.info("")
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        
        return results
    
    def get_trained_model(self) -> ApexCoreFull:
        """Get the trained model instance."""
        return self.model


def train_from_polygon(
    symbols: Optional[List[str]] = None,
    lookback_days: int = 365,
) -> Dict[str, Any]:
    """
    Convenience function to train ApexCore on real Polygon data.
    
    Args:
        symbols: List of symbols to train on (default: top 15 stocks + ETFs)
        lookback_days: Number of days of history to fetch
        
    Returns:
        Training results dictionary
    """
    config = TrainingConfig(lookback_days=lookback_days)
    if symbols:
        config.symbols = symbols
    
    trainer = LiveDataTrainer(config)
    return trainer.run_full_pipeline()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    results = train_from_polygon()
    
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Samples: {results['samples']}")
    print(f"Symbols: {results['symbols']}")
    print(f"Regime Accuracy: {results['metrics']['regime_accuracy']:.1%}")
    print(f"Risk Accuracy: {results['metrics']['risk_accuracy']:.1%}")
    print(f"Score MAE: {results['metrics']['score_mae']:.2f}")
    print(f"Runner Rate: {results['runner_rate']:.1%}")
    print(f"Model saved to: {results['model_path']}")
