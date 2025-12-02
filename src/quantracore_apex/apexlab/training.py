"""
ApexCore V2 Training Pipeline.

Provides walk-forward splits, multi-task training, class balancing,
and ensemble training with metrics logging.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import json

from src.quantracore_apex.apexcore.apexcore_v2 import (
    ApexCoreV2Model,
    ApexCoreV2Big,
    ApexCoreV2Mini,
    ApexCoreV2Ensemble,
    ApexCoreV2Config,
    ModelVariant,
)
from src.quantracore_apex.apexcore.manifest import (
    ApexCoreV2Manifest,
    ManifestMetrics,
    ManifestThresholds,
    compute_file_hash,
)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    variant: str = "big"
    ensemble_size: int = 3
    random_state: int = 42
    val_ratio: float = 0.2
    bootstrap: bool = True
    
    loss_weight_quantra_score: float = 1.0
    loss_weight_runner: float = 1.0
    loss_weight_quality_tier: float = 0.5
    loss_weight_avoid_trade: float = 1.0
    loss_weight_regime: float = 0.5
    
    balance_regimes: bool = True
    balance_runners: bool = True
    hard_negative_weight: float = 2.0
    
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1


def create_walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    time_column: str = "event_time",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time-aware walk-forward splits.
    
    Args:
        df: DataFrame with event data
        n_folds: Number of folds
        val_ratio: Ratio of validation data per fold
        time_column: Column containing timestamps
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    if time_column in df.columns:
        df_sorted = df.sort_values(time_column).reset_index(drop=True)
        sorted_indices = df_sorted.index.values
    else:
        sorted_indices = np.arange(len(df))
    
    n_samples = len(sorted_indices)
    fold_size = n_samples // n_folds
    
    splits = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = min((fold + 1) * fold_size, n_samples)
        
        if fold == 0:
            train_end = int(n_samples * (1 - val_ratio))
            val_start = train_end
            val_end = n_samples
            train_indices = sorted_indices[:train_end]
            val_indices = sorted_indices[val_start:val_end]
        else:
            train_indices = sorted_indices[:val_start]
            val_indices = sorted_indices[val_start:val_end]
        
        if len(train_indices) > 0 and len(val_indices) > 0:
            splits.append((train_indices, val_indices))
    
    if not splits:
        train_end = int(n_samples * (1 - val_ratio))
        splits.append((sorted_indices[:train_end], sorted_indices[train_end:]))
    
    return splits


def compute_sample_weights(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> np.ndarray:
    """
    Compute sample weights for balanced training.
    
    Args:
        df: DataFrame with training data
        config: Training configuration
        
    Returns:
        Array of sample weights
    """
    weights = np.ones(len(df))
    
    if config.balance_regimes and "regime_label" in df.columns:
        regime_counts = df["regime_label"].value_counts()
        max_count = regime_counts.max()
        for regime, count in regime_counts.items():
            mask = df["regime_label"] == regime
            weights[mask] *= max_count / count
    
    if config.balance_runners and "hit_runner_threshold" in df.columns:
        runner_counts = df["hit_runner_threshold"].value_counts()
        max_count = runner_counts.max()
        for runner, count in runner_counts.items():
            mask = df["hit_runner_threshold"] == runner
            weights[mask] *= max_count / count
    
    if config.hard_negative_weight > 1.0:
        if "future_quality_tier" in df.columns and "avoid_trade" in df.columns:
            hard_neg_mask = (df["future_quality_tier"] == "D") | (df["avoid_trade"] == 1)
            weights[hard_neg_mask] *= config.hard_negative_weight
    
    weights = weights / weights.mean()
    
    return weights


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract feature matrix from DataFrame.
    
    Args:
        df: DataFrame with training data
        
    Returns:
        Feature matrix [n_samples, n_features]
    """
    feature_cols = []
    
    numeric_cols = ["quantra_score"]
    for col in numeric_cols:
        if col in df.columns:
            feature_cols.append(df[col].values.reshape(-1, 1))
    
    categorical_cols = [
        "risk_tier", "entropy_band", "volatility_band",
        "liquidity_band", "market_cap_band",
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            encoded = pd.get_dummies(df[col], prefix=col)
            feature_cols.append(encoded.values)
    
    if "protocol_vector" in df.columns:
        first_vec = df["protocol_vector"].iloc[0]
        if isinstance(first_vec, (list, np.ndarray)):
            protocol_matrix = np.stack(df["protocol_vector"].values)
            feature_cols.append(protocol_matrix)
    
    if feature_cols:
        return np.hstack(feature_cols)
    
    return df["quantra_score"].values.reshape(-1, 1) if "quantra_score" in df.columns else np.zeros((len(df), 1))


def extract_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extract target dictionary from DataFrame.
    
    Args:
        df: DataFrame with training data
        
    Returns:
        Dictionary of target arrays
    """
    targets = {}
    
    if "quantra_score" in df.columns:
        targets["quantra_score"] = df["quantra_score"].values
    
    if "hit_runner_threshold" in df.columns:
        targets["hit_runner_threshold"] = df["hit_runner_threshold"].values.astype(int)
    
    if "future_quality_tier" in df.columns:
        targets["future_quality_tier"] = df["future_quality_tier"].values
    
    if "avoid_trade" in df.columns:
        targets["avoid_trade"] = df["avoid_trade"].values.astype(int)
    
    if "regime_label" in df.columns:
        targets["regime_label"] = df["regime_label"].values
    
    return targets


def compute_validation_metrics(
    model: ApexCoreV2Model,
    X_val: np.ndarray,
    targets_val: Dict[str, np.ndarray],
) -> ManifestMetrics:
    """
    Compute validation metrics for a trained model.
    
    Args:
        model: Trained model
        X_val: Validation features
        targets_val: Validation targets
        
    Returns:
        ManifestMetrics object
    """
    outputs = model.forward(X_val)
    
    mse_quantra = 100.0
    if "quantra_score" in targets_val and "quantra_score" in outputs:
        mse_quantra = float(np.mean((outputs["quantra_score"] - targets_val["quantra_score"]) ** 2))
    
    brier_runner = 1.0
    auc_runner = 0.5
    calibration_error = 1.0
    
    if "hit_runner_threshold" in targets_val and "runner_prob" in outputs:
        y_true = targets_val["hit_runner_threshold"]
        y_prob = outputs["runner_prob"]
        
        brier_runner = float(np.mean((y_prob - y_true) ** 2))
        
        if len(np.unique(y_true)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                auc_runner = float(roc_auc_score(y_true, y_prob))
            except (ImportError, ValueError):
                auc_runner = 0.5
        
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / len(y_true)
        calibration_error = float(ece)
    
    acc_quality = 0.2
    if "future_quality_tier" in targets_val and "quality_logits" in outputs:
        pred_tiers = np.argmax(outputs["quality_logits"], axis=1)
        tier_mapping = {"A_PLUS": 0, "A": 1, "B": 2, "C": 3, "D": 4}
        true_tiers = np.array([tier_mapping.get(t, 3) for t in targets_val["future_quality_tier"]])
        acc_quality = float(np.mean(pred_tiers == true_tiers))
    
    acc_regime = 0.2
    if "regime_label" in targets_val and "regime_logits" in outputs:
        pred_regimes = np.argmax(outputs["regime_logits"], axis=1)
        regime_mapping = {"trend_up": 0, "trend_down": 1, "chop": 2, "squeeze": 3, "crash": 4}
        true_regimes = np.array([regime_mapping.get(r, 2) for r in targets_val["regime_label"]])
        acc_regime = float(np.mean(pred_regimes == true_regimes))
    
    return ManifestMetrics(
        val_brier_runner=brier_runner,
        val_auc_runner=auc_runner,
        val_calibration_error_runner=calibration_error,
        val_accuracy_quality_tier=acc_quality,
        val_accuracy_regime=acc_regime,
        val_mse_quantra_score=mse_quantra,
    )


class ApexCoreV2Trainer:
    """
    Training pipeline for ApexCore V2 models.
    
    Supports:
    - Walk-forward time-aware splits
    - Class balancing and hard negative mining
    - Ensemble training with bootstrap
    - Metrics logging and manifest generation
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.training_log: List[Dict[str, Any]] = []
    
    def train(
        self,
        df: pd.DataFrame,
        output_dir: str,
    ) -> Tuple[ApexCoreV2Ensemble, ApexCoreV2Manifest]:
        """
        Train an ensemble model on the dataset.
        
        Args:
            df: DataFrame with training data
            output_dir: Directory to save model and manifest
            
        Returns:
            (trained_ensemble, manifest)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = create_walk_forward_splits(df, val_ratio=self.config.val_ratio)
        train_idx, val_idx = splits[-1]
        
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        
        X_train = extract_features(df_train)
        targets_train = extract_targets(df_train)
        X_val = extract_features(df_val)
        targets_val = extract_targets(df_val)
        
        variant = ModelVariant.BIG if self.config.variant == "big" else ModelVariant.MINI
        ensemble = ApexCoreV2Ensemble(
            ensemble_size=self.config.ensemble_size,
            variant=variant,
            base_random_state=self.config.random_state,
        )
        
        ensemble.fit(X_train, targets_train, bootstrap=self.config.bootstrap)
        
        metrics = compute_validation_metrics(ensemble.members[0], X_val, targets_val)
        
        model_path = output_path / "ensemble"
        ensemble.save(str(model_path))
        
        member_hashes = {}
        for i in range(self.config.ensemble_size):
            member_path = model_path / f"member_{i}.joblib"
            if member_path.exists():
                member_hashes[f"member_{i}"] = compute_file_hash(str(member_path))
        
        manifest = ApexCoreV2Manifest(
            variant=self.config.variant,
            ensemble_size=self.config.ensemble_size,
            hashes=member_hashes,
            metrics=metrics,
            symbols_trained=df["symbol"].nunique() if "symbol" in df.columns else 0,
            samples_trained=len(df_train),
        )
        
        manifest_path = output_path / "manifests"
        manifest_path.mkdir(exist_ok=True)
        manifest.save(str(manifest_path / "latest.json"))
        
        self._log_training_run(df_train, df_val, metrics)
        
        return ensemble, manifest
    
    def train_single_model(
        self,
        df: pd.DataFrame,
        output_dir: str,
    ) -> Tuple[ApexCoreV2Model, ApexCoreV2Manifest]:
        """
        Train a single model (no ensemble).
        
        Args:
            df: DataFrame with training data
            output_dir: Directory to save model
            
        Returns:
            (trained_model, manifest)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = create_walk_forward_splits(df, val_ratio=self.config.val_ratio)
        train_idx, val_idx = splits[-1]
        
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        
        X_train = extract_features(df_train)
        targets_train = extract_targets(df_train)
        X_val = extract_features(df_val)
        targets_val = extract_targets(df_val)
        
        if self.config.variant == "big":
            model = ApexCoreV2Big()
        else:
            model = ApexCoreV2Mini()
        
        model.fit(X_train, targets_train)
        
        metrics = compute_validation_metrics(model, X_val, targets_val)
        
        model_path = output_path / "model.joblib"
        model.save(str(model_path))
        
        manifest = ApexCoreV2Manifest(
            variant=self.config.variant,
            ensemble_size=1,
            hashes={"model": compute_file_hash(str(model_path))},
            metrics=metrics,
            symbols_trained=df["symbol"].nunique() if "symbol" in df.columns else 0,
            samples_trained=len(df_train),
        )
        
        manifest_path = output_path / "manifests"
        manifest_path.mkdir(exist_ok=True)
        manifest.save(str(manifest_path / "latest.json"))
        
        return model, manifest
    
    def _log_training_run(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        metrics: ManifestMetrics,
    ) -> None:
        """Log training run details."""
        self.training_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "metrics": metrics.to_dict(),
        })
    
    def save_training_log(self, path: str) -> None:
        """Save training log to JSON file."""
        with open(path, "w") as f:
            json.dump(self.training_log, f, indent=2)


def run_training(
    dataset_path: str,
    variant: str = "big",
    output_dir: str = "models/apexcore_v2",
    ensemble_size: int = 3,
) -> Tuple[ApexCoreV2Ensemble, ApexCoreV2Manifest]:
    """
    Run training pipeline from command line.
    
    Args:
        dataset_path: Path to Parquet dataset
        variant: Model variant ("big" or "mini")
        output_dir: Output directory
        ensemble_size: Number of ensemble members
        
    Returns:
        (trained_ensemble, manifest)
    """
    df = pd.read_parquet(dataset_path)
    
    config = TrainingConfig(
        variant=variant,
        ensemble_size=ensemble_size,
    )
    
    trainer = ApexCoreV2Trainer(config)
    ensemble, manifest = trainer.train(df, f"{output_dir}/{variant}")
    
    return ensemble, manifest


def run_swing_training_cycle(
    symbols: List[str],
    days_back: int = 60,
    min_samples: int = 50,
    skip_enrichment: bool = False,
) -> Dict[str, Any]:
    """
    Run a swing trading training cycle using Polygon/Alpaca data.
    
    Args:
        symbols: List of stock symbols to train on
        days_back: Number of days of historical data
        min_samples: Minimum samples required for training
        skip_enrichment: If True, skip external API enrichment (FRED, Finnhub, SEC)
        
    Returns:
        Training results dictionary
    """
    import logging
    import time
    from datetime import datetime, timedelta
    from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow
    
    logger = logging.getLogger(__name__)
    results = {
        "symbols_processed": 0,
        "samples_collected": 0,
        "training_completed": False,
        "models_updated": [],
        "errors": [],
    }
    
    try:
        from src.quantracore_apex.apexlab.features import SwingFeatureExtractor
        from src.quantracore_apex.data_layer.adapters.alpaca_data_adapter import AlpacaDataAdapter
        
        # Disable enrichment for faster training with Alpaca-only data
        extractor = SwingFeatureExtractor(enable_enrichment=not skip_enrichment)
        alpaca = AlpacaDataAdapter()
        
        if not alpaca.is_available():
            results["errors"].append("Alpaca adapter not available - check API credentials")
            return results
        
        all_features = []
        all_labels = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back * 2)
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Fetching data for {symbol} ({i+1}/{len(symbols)}) via Alpaca...")
                
                bars = alpaca.fetch_ohlcv(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    timeframe="1d"
                )
                
                if not bars or len(bars) < 30:
                    logger.warning(f"Insufficient data for {symbol}: {len(bars) if bars else 0} bars")
                    continue
                
                window_size = min(30, len(bars) - 5)
                for j in range(len(bars) - window_size - 5):
                    window_bars = bars[j:j + window_size]
                    future_bars = bars[j + window_size:j + window_size + 5]
                    
                    window = OhlcvWindow(
                        symbol=symbol,
                        timeframe="day",
                        bars=window_bars
                    )
                    
                    try:
                        features = extractor.extract(window)
                        
                        current_close = window_bars[-1].close
                        future_close = future_bars[-1].close if future_bars else current_close
                        forward_return = (future_close - current_close) / current_close
                        
                        quantra_score = 0.5 + forward_return * 2
                        quantra_score = max(0.0, min(1.0, quantra_score))
                        
                        runner_label = 1 if forward_return > 0.05 else 0
                        quality_tier = 2 if forward_return > 0.03 else (1 if forward_return > 0 else 0)
                        avoid_trade = 1 if forward_return < -0.03 else 0
                        
                        volatility = np.std([b.close for b in window_bars[-20:]]) / np.mean([b.close for b in window_bars[-20:]])
                        regime_label = 0 if forward_return > 0.02 else (1 if forward_return < -0.02 else (2 if volatility > 0.02 else 3))
                        
                        all_features.append(features)
                        all_labels.append({
                            'quantra_score': quantra_score,
                            'runner_label': runner_label,
                            'quality_tier': quality_tier,
                            'avoid_trade': avoid_trade,
                            'regime_label': regime_label,
                        })
                        
                    except Exception as e:
                        logger.debug(f"Feature extraction error for {symbol} window {j}: {e}")
                        continue
                
                results["symbols_processed"] += 1
                logger.info(f"Processed {symbol}: {len(all_features)} total samples so far")
                    
            except Exception as e:
                results["errors"].append(f"{symbol}: {str(e)}")
                logger.warning(f"Error processing {symbol}: {e}")
        
        if all_features:
            feature_array = np.array(all_features)
            feature_names = extractor.get_feature_names()
            
            combined_df = pd.DataFrame(feature_array, columns=feature_names[:len(feature_array[0])])
            
            for key in ['quantra_score', 'runner_label', 'quality_tier', 'avoid_trade', 'regime_label']:
                combined_df[key] = [l[key] for l in all_labels]
            
            results["samples_collected"] = len(combined_df)
            
            if len(combined_df) >= min_samples:
                logger.info(f"Training with {len(combined_df)} samples...")
                
                config = TrainingConfig(
                    variant="mini",
                    ensemble_size=2,
                    n_estimators=50,
                    max_depth=4,
                )
                
                trainer = ApexCoreV2Trainer(config)
                
                output_dir = Path("models/swing_basic")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                ensemble, manifest = trainer.train(combined_df, str(output_dir / "model"))
                
                results["training_completed"] = True
                results["models_updated"] = ["swing_basic"]
                results["manifest"] = {
                    "version": manifest.version if manifest else "unknown",
                    "created_at": datetime.now().isoformat(),
                }
                
                logger.info("Training cycle completed successfully!")
            else:
                results["message"] = f"Insufficient samples: {len(combined_df)} < {min_samples}"
        else:
            results["message"] = "No samples collected from any symbol"
            
    except Exception as e:
        import traceback
        results["errors"].append(f"Training error: {str(e)}")
        results["traceback"] = traceback.format_exc()
        logger.error(f"Training cycle error: {e}")
    
    return results


__all__ = [
    "TrainingConfig",
    "ApexCoreV2Trainer",
    "create_walk_forward_splits",
    "compute_sample_weights",
    "extract_features",
    "extract_targets",
    "compute_validation_metrics",
    "run_training",
    "run_swing_training_cycle",
]
