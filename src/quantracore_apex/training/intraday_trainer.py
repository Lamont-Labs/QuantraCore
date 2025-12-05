"""
Intraday Model Training Pipeline for Moonshot Detection

Trains on 1-minute bar patterns to detect stocks ready for massive moves.
Uses the SPY 2008-2021 dataset (2M+ bars, 207k windows) to learn microstructure
patterns that precede breakouts.

Key insight: Intraday patterns (volume surges, price compression, momentum shifts)
often precede multi-day breakouts. Training on 1-min data captures these signals
that daily data misses.
"""

import logging
import os
import pickle
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib

from src.quantracore_apex.data.intraday_pipeline import IntradayTrainingPipeline
from src.quantracore_apex.data.intraday_features import IntradayFeatureExtractor
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow

logger = logging.getLogger(__name__)


@dataclass
class IntradayTrainingConfig:
    """Configuration for intraday model training."""
    window_size: int = 100
    step_size: int = 10
    
    label_horizon_bars: int = 390
    moonshot_threshold: float = 0.02
    min_gain_threshold: float = 0.005
    
    test_size: float = 0.2
    random_state: int = 42
    
    n_estimators: int = 200
    max_depth: int = 8
    min_samples_leaf: int = 50
    learning_rate: float = 0.05
    
    model_dir: str = "models"
    model_name: str = "intraday_moonshot_v1"


@dataclass
class TrainingResult:
    """Results from model training."""
    model: Any
    scaler: StandardScaler
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    config: IntradayTrainingConfig
    training_samples: int
    positive_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class IntradayLabelGenerator:
    """
    Generates labels for moonshot detection from intraday data.
    
    Labels are based on forward returns over a configurable horizon.
    For moonshot detection, we look for significant gains (2%+ intraday
    or larger moves over multiple sessions).
    """
    
    def __init__(
        self,
        horizon_bars: int = 390,
        moonshot_threshold: float = 0.02,
        min_gain_threshold: float = 0.005,
    ):
        self.horizon_bars = horizon_bars
        self.moonshot_threshold = moonshot_threshold
        self.min_gain_threshold = min_gain_threshold
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        window_size: int = 100,
        step_size: int = 10,
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        Generate labels for each window in the data.
        
        Returns:
            labels: Binary labels (1 = moonshot candidate, 0 = not)
            forward_returns: Actual forward returns for each window
            window_indices: Starting index of each window
        """
        labels = []
        forward_returns = []
        window_indices = []
        
        closes = df['close'].values
        n_bars = len(closes)
        
        max_start = n_bars - window_size - self.horizon_bars
        
        for i in range(0, max_start, step_size):
            window_end = i + window_size
            forward_end = min(window_end + self.horizon_bars, n_bars - 1)
            
            entry_price = closes[window_end - 1]
            
            future_prices = closes[window_end:forward_end + 1]
            if len(future_prices) == 0:
                continue
            
            max_future_price = np.max(future_prices)
            max_return = (max_future_price - entry_price) / entry_price
            
            if max_return >= self.moonshot_threshold:
                label = 1
            elif max_return >= self.min_gain_threshold:
                label = 0
            else:
                label = 0
            
            labels.append(label)
            forward_returns.append(max_return)
            window_indices.append(i)
        
        return labels, forward_returns, window_indices
    
    def generate_regression_labels(
        self,
        df: pd.DataFrame,
        window_size: int = 100,
        step_size: int = 10,
    ) -> Tuple[List[float], List[int]]:
        """
        Generate regression labels (actual forward returns).
        
        Returns:
            forward_returns: Max forward return for each window
            window_indices: Starting index of each window
        """
        forward_returns = []
        window_indices = []
        
        closes = df['close'].values
        n_bars = len(closes)
        
        max_start = n_bars - window_size - self.horizon_bars
        
        for i in range(0, max_start, step_size):
            window_end = i + window_size
            forward_end = min(window_end + self.horizon_bars, n_bars - 1)
            
            entry_price = closes[window_end - 1]
            future_prices = closes[window_end:forward_end + 1]
            
            if len(future_prices) == 0:
                continue
            
            max_return = (np.max(future_prices) - entry_price) / entry_price
            
            forward_returns.append(max_return)
            window_indices.append(i)
        
        return forward_returns, window_indices


class IntradayMoonshotTrainer:
    """
    Complete training pipeline for intraday moonshot detection.
    
    Uses the massive SPY 1-min dataset to learn patterns that precede
    significant price moves.
    """
    
    def __init__(self, config: Optional[IntradayTrainingConfig] = None):
        self.config = config or IntradayTrainingConfig()
        self.pipeline = IntradayTrainingPipeline()
        self.feature_extractor = IntradayFeatureExtractor()
        self.label_generator = IntradayLabelGenerator(
            horizon_bars=self.config.label_horizon_bars,
            moonshot_threshold=self.config.moonshot_threshold,
            min_gain_threshold=self.config.min_gain_threshold,
        )
        
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
    
    def prepare_training_data(
        self,
        symbol: str = "SPY",
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Prepare training data from 1-minute bars.
        
        Returns:
            X: Feature matrix (n_samples, 120)
            y: Labels (n_samples,)
            forward_returns: Actual forward returns for analysis
        """
        logger.info(f"Loading {symbol} data...")
        df = self.pipeline.merger.kaggle_processor.load_symbol(symbol)
        
        if df is None:
            raise ValueError(f"No data found for {symbol}")
        
        logger.info(f"Loaded {len(df):,} bars")
        
        logger.info("Generating labels...")
        labels, forward_returns, window_indices = self.label_generator.generate_labels(
            df,
            window_size=self.config.window_size,
            step_size=self.config.step_size,
        )
        
        logger.info(f"Generated {len(labels):,} labels")
        logger.info(f"Positive samples: {sum(labels):,} ({100*sum(labels)/len(labels):.1f}%)")
        
        if max_samples and len(labels) > max_samples:
            indices = np.random.choice(len(labels), max_samples, replace=False)
            labels = [labels[i] for i in indices]
            forward_returns = [forward_returns[i] for i in indices]
            window_indices = [window_indices[i] for i in indices]
        
        logger.info("Extracting features...")
        X_list = []
        valid_labels = []
        valid_returns = []
        
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        timestamps = pd.to_datetime(df['timestamp']).tolist()
        
        batch_size = 1000
        for batch_start in range(0, len(window_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(window_indices))
            
            for idx in range(batch_start, batch_end):
                i = window_indices[idx]
                window_end = i + self.config.window_size
                
                bars = []
                for j in range(i, window_end):
                    ts = timestamps[j]
                    if hasattr(ts, 'to_pydatetime'):
                        ts = ts.to_pydatetime()
                    bar = OhlcvBar(
                        timestamp=ts,
                        open=float(opens[j]),
                        high=float(highs[j]),
                        low=float(lows[j]),
                        close=float(closes[j]),
                        volume=float(volumes[j]),
                    )
                    bars.append(bar)
                
                window = OhlcvWindow(
                    symbol=symbol,
                    timeframe="1min",
                    bars=bars,
                )
                
                try:
                    features = self.feature_extractor.extract(window)
                    
                    if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                        X_list.append(features)
                        valid_labels.append(labels[idx])
                        valid_returns.append(forward_returns[idx])
                except Exception as e:
                    logger.debug(f"Feature extraction failed for window {idx}: {e}")
                    continue
            
            if batch_end % 10000 == 0:
                logger.info(f"Processed {batch_end:,}/{len(window_indices):,} windows")
        
        X = np.array(X_list)
        y = np.array(valid_labels)
        
        logger.info(f"Final dataset: {len(X):,} samples, {sum(y):,} positive ({100*sum(y)/len(y):.1f}%)")
        
        return X, y, valid_returns
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Any, StandardScaler, Dict[str, float]]:
        """
        Train the moonshot detection model.
        
        Returns:
            model: Trained classifier
            scaler: Fitted feature scaler
            metrics: Performance metrics
        """
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Training GradientBoosting classifier...")
        model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            subsample=0.8,
            max_features='sqrt',
        )
        
        model.fit(X_train_scaled, y_train)
        
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold_metrics = {}
        for thresh in thresholds:
            y_pred_thresh = (y_proba >= thresh).astype(int)
            prec = precision_score(y_test, y_pred_thresh, zero_division=0)
            rec = recall_score(y_test, y_pred_thresh, zero_division=0)
            threshold_metrics[f"precision_at_{thresh}"] = prec
            threshold_metrics[f"recall_at_{thresh}"] = rec
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "test_samples": len(y_test),
            "test_positives": int(sum(y_test)),
            "train_samples": len(y_train),
            "train_positives": int(sum(y_train)),
            **threshold_metrics,
        }
        
        logger.info(f"Model Performance:")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")
        
        for thresh in [0.5, 0.6, 0.7]:
            logger.info(f"  Precision at {thresh}: {threshold_metrics[f'precision_at_{thresh}']:.3f}")
        
        return model, scaler, metrics
    
    def get_feature_importance(
        self,
        model: Any,
    ) -> Dict[str, float]:
        """Get feature importance from trained model."""
        feature_names = self.feature_extractor.FEATURE_NAMES
        importances = model.feature_importances_
        
        importance_dict = dict(zip(feature_names, importances))
        sorted_importance = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sorted_importance
    
    def save_model(
        self,
        result: TrainingResult,
    ) -> str:
        """Save trained model and metadata."""
        model_path = Path(self.config.model_dir) / f"{self.config.model_name}.pkl.gz"
        
        save_data = {
            "model": result.model,
            "scaler": result.scaler,
            "metrics": result.metrics,
            "feature_importance": result.feature_importance,
            "config": self.config.__dict__,
            "training_samples": result.training_samples,
            "positive_samples": result.positive_samples,
            "timestamp": result.timestamp,
            "feature_names": self.feature_extractor.FEATURE_NAMES,
        }
        
        with gzip.open(model_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def run_training(
        self,
        symbol: str = "SPY",
        max_samples: Optional[int] = None,
    ) -> TrainingResult:
        """
        Run complete training pipeline.
        
        Args:
            symbol: Symbol to train on
            max_samples: Optional limit on training samples
        
        Returns:
            TrainingResult with model and metrics
        """
        logger.info("=" * 60)
        logger.info("INTRADAY MOONSHOT MODEL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Window size: {self.config.window_size}")
        logger.info(f"Horizon: {self.config.label_horizon_bars} bars")
        logger.info(f"Moonshot threshold: {self.config.moonshot_threshold:.1%}")
        
        X, y, forward_returns = self.prepare_training_data(symbol, max_samples)
        
        model, scaler, metrics = self.train_model(X, y)
        
        feature_importance = self.get_feature_importance(model)
        
        result = TrainingResult(
            model=model,
            scaler=scaler,
            metrics=metrics,
            feature_importance=feature_importance,
            config=self.config,
            training_samples=len(X),
            positive_samples=int(sum(y)),
        )
        
        model_path = self.save_model(result)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Training samples: {result.training_samples:,}")
        logger.info(f"Positive samples: {result.positive_samples:,}")
        logger.info(f"Precision: {metrics['precision']:.3f}")
        logger.info(f"Recall: {metrics['recall']:.3f}")
        
        logger.info("\nTop 10 Features:")
        for i, (name, importance) in enumerate(list(feature_importance.items())[:10]):
            logger.info(f"  {i+1}. {name}: {importance:.4f}")
        
        return result


def load_intraday_model(model_path: str = "models/intraday_moonshot_v1.pkl.gz") -> Dict:
    """Load a trained intraday model."""
    with gzip.open(model_path, 'rb') as f:
        return pickle.load(f)


def run_intraday_training(
    symbol: str = "SPY",
    max_samples: Optional[int] = None,
    moonshot_threshold: float = 0.02,
    horizon_bars: int = 390,
) -> TrainingResult:
    """
    Convenience function to run intraday training.
    
    Args:
        symbol: Symbol to train on
        max_samples: Optional sample limit (for testing)
        moonshot_threshold: Min return to classify as moonshot (0.02 = 2%)
        horizon_bars: Forward-looking window (390 = 1 trading day)
    
    Returns:
        TrainingResult with trained model
    """
    config = IntradayTrainingConfig(
        moonshot_threshold=moonshot_threshold,
        label_horizon_bars=horizon_bars,
    )
    
    trainer = IntradayMoonshotTrainer(config)
    return trainer.run_training(symbol, max_samples)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    result = run_intraday_training(
        symbol="SPY",
        max_samples=50000,
        moonshot_threshold=0.02,
        horizon_bars=390,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Samples: {result.training_samples:,}")
    print(f"Positives: {result.positive_samples:,}")
    print(f"Precision: {result.metrics['precision']:.3f}")
    print(f"Recall: {result.metrics['recall']:.3f}")
    print(f"F1: {result.metrics['f1_score']:.3f}")
