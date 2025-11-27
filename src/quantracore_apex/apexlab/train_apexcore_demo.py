"""
ApexCore Demo Training Script for ApexLab.

Trains a small neural network on Apex-generated labels.
Uses scikit-learn for simplicity (no PyTorch dependency).
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json
import logging
import pickle

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from .windows import WindowBuilder
from .dataset_builder import DatasetBuilder


logger = logging.getLogger(__name__)


class ApexCoreDemoTrainer:
    """
    Trains a demo ApexCore model using scikit-learn.
    """
    
    def __init__(
        self,
        model_dir: str = "data/training/models",
        hidden_layers: Tuple[int, ...] = (64, 32),
        max_iter: int = 500
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        
        self.scaler = StandardScaler()
        self.regime_model: Optional[MLPClassifier] = None
        self.risk_model: Optional[MLPClassifier] = None
        self.score_model: Optional[MLPRegressor] = None
        
        self.training_metadata: Dict[str, Any] = {}
    
    def generate_demo_data(
        self,
        symbols: list = None,
        n_bars: int = 200
    ) -> Dict[str, Any]:
        """
        Generate demo training data using synthetic adapter.
        """
        if symbols is None:
            symbols = ["DEMO1", "DEMO2", "DEMO3", "DEMO4", "DEMO5"]
        
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100, step=10)
        dataset_builder = DatasetBuilder(enable_logging=False)
        
        all_windows = []
        
        from datetime import datetime, timedelta
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=n_bars)
        
        for symbol in symbols:
            bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            windows = window_builder.build(normalized_bars, symbol)
            all_windows.extend(windows)
            logger.info(f"Generated {len(windows)} windows for {symbol}")
        
        dataset = dataset_builder.build(all_windows, "demo_dataset")
        logger.info(f"Built dataset with {dataset['metadata']['n_samples']} samples")
        
        return dataset
    
    def train(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train demo models on the dataset.
        """
        features = dataset["features"]
        labels = dataset["labels"]
        
        features_scaled = self.scaler.fit_transform(features)
        
        logger.info("Training regime classifier...")
        self.regime_model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.regime_model.fit(features_scaled, labels["regime_class"])
        regime_acc = accuracy_score(labels["regime_class"], self.regime_model.predict(features_scaled))
        logger.info(f"Regime classifier training accuracy: {regime_acc:.4f}")
        
        logger.info("Training risk classifier...")
        self.risk_model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.risk_model.fit(features_scaled, labels["risk_tier"])
        risk_acc = accuracy_score(labels["risk_tier"], self.risk_model.predict(features_scaled))
        logger.info(f"Risk classifier training accuracy: {risk_acc:.4f}")
        
        logger.info("Training QuantraScore regressor...")
        self.score_model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.score_model.fit(features_scaled, labels["quantrascore_numeric"])
        score_mae = mean_absolute_error(labels["quantrascore_numeric"], self.score_model.predict(features_scaled))
        logger.info(f"QuantraScore regressor training MAE: {score_mae:.4f}")
        
        self.training_metadata = {
            "trained_at": datetime.utcnow().isoformat(),
            "n_samples": len(features),
            "feature_dim": features.shape[1],
            "hidden_layers": self.hidden_layers,
            "metrics": {
                "regime_accuracy": float(regime_acc),
                "risk_accuracy": float(risk_acc),
                "score_mae": float(score_mae),
            }
        }
        
        return self.training_metadata
    
    def save(self, model_name: str = "apexcore_demo") -> str:
        """
        Save trained models to disk.
        """
        model_path = self.model_dir / f"{model_name}.pkl"
        
        model_data = {
            "scaler": self.scaler,
            "regime_model": self.regime_model,
            "risk_model": self.risk_model,
            "score_model": self.score_model,
            "metadata": self.training_metadata,
        }
        
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.training_metadata, f, indent=2)
        
        logger.info(f"Saved model to {model_path}")
        return str(model_path)
    
    def load(self, model_path: str) -> None:
        """
        Load trained models from disk.
        """
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data["scaler"]
        self.regime_model = model_data["regime_model"]
        self.risk_model = model_data["risk_model"]
        self.score_model = model_data["score_model"]
        self.training_metadata = model_data["metadata"]
        
        logger.info(f"Loaded model from {model_path}")


def run_demo_training():
    """
    Run the demo training pipeline.
    """
    logging.basicConfig(level=logging.INFO)
    
    trainer = ApexCoreDemoTrainer()
    
    logger.info("Generating demo data...")
    dataset = trainer.generate_demo_data(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        n_bars=300
    )
    
    logger.info("Training models...")
    metrics = trainer.train(dataset)
    
    logger.info("Saving models...")
    model_path = trainer.save()
    
    logger.info(f"Training complete! Metrics: {metrics}")
    return model_path


if __name__ == "__main__":
    run_demo_training()
