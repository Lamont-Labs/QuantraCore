"""
Dataset Builder for ApexLab.

Combines features and labels into training datasets.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.quantracore_apex.core.schemas import OhlcvWindow
from .features import FeatureExtractor
from .labels import LabelGenerator


class DatasetBuilder:
    """
    Builds training datasets from OHLCV windows.
    """
    
    def __init__(
        self,
        output_dir: str = "data/training",
        enable_logging: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor()
        self.label_generator = LabelGenerator(enable_logging=enable_logging)
    
    def build(
        self,
        windows: List[OhlcvWindow],
        dataset_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a training dataset from windows.
        
        Returns:
            Dictionary with features, labels, and metadata
        """
        if not windows:
            return {"error": "No windows provided"}
        
        features = self.feature_extractor.extract_batch(windows)
        
        labels = self.label_generator.generate_batch(windows)
        
        symbols = [w.symbol for w in windows]
        timestamps = [w.bars[-1].timestamp.isoformat() for w in windows]
        window_hashes = [w.get_hash() for w in windows]
        
        dataset = {
            "features": features,
            "labels": labels,
            "metadata": {
                "name": dataset_name or f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "created": datetime.utcnow().isoformat(),
                "n_samples": len(windows),
                "feature_dim": features.shape[1],
                "feature_names": self.feature_extractor.get_feature_names(),
                "label_names": list(labels.keys()),
                "symbols": symbols,
                "timestamps": timestamps,
                "window_hashes": window_hashes,
            }
        }
        
        return dataset
    
    def save(self, dataset: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save dataset to disk.
        
        Returns:
            Path to saved dataset
        """
        if filename is None:
            filename = f"{dataset['metadata']['name']}.npz"
        
        filepath = self.output_dir / filename
        
        np.savez_compressed(
            filepath,
            features=dataset["features"],
            **{f"label_{k}": v for k, v in dataset["labels"].items()}
        )
        
        metadata_path = filepath.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(dataset["metadata"], f, indent=2)
        
        return str(filepath)
    
    def load(self, filepath_str: str) -> Dict[str, Any]:
        """
        Load dataset from disk.
        """
        filepath = Path(filepath_str)
        
        data = np.load(filepath, allow_pickle=True)
        
        features = data["features"]
        labels = {
            k.replace("label_", ""): data[k]
            for k in data.files if k.startswith("label_")
        }
        
        metadata_path = filepath.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return {
            "features": features,
            "labels": labels,
            "metadata": metadata,
        }
    
    def split(
        self,
        dataset: Dict[str, Any],
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Split dataset into train and validation sets.
        """
        n_samples = dataset["features"].shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_dataset = {
            "features": dataset["features"][train_indices],
            "labels": {k: v[train_indices] for k, v in dataset["labels"].items()},
            "metadata": {**dataset["metadata"], "split": "train", "n_samples": len(train_indices)},
        }
        
        val_dataset = {
            "features": dataset["features"][val_indices],
            "labels": {k: v[val_indices] for k, v in dataset["labels"].items()},
            "metadata": {**dataset["metadata"], "split": "validation", "n_samples": len(val_indices)},
        }
        
        return train_dataset, val_dataset


_default_builder = None

def build_dataset(
    bars: List, 
    window_size: int = 20, 
    step: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Module-level convenience function to build dataset from bars.
    
    Args:
        bars: List of OhlcvBar objects
        window_size: Size of each window
        step: Steps between windows
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    global _default_builder
    if _default_builder is None:
        _default_builder = DatasetBuilder(enable_logging=False)
    
    from src.quantracore_apex.data_layer.normalization import build_windows
    
    windows = build_windows(bars, symbol="DATASET", window_size=window_size, step=step)
    
    if not windows:
        return np.array([]), np.array([])
    
    features = _default_builder.feature_extractor.extract_batch(windows)
    
    labels = []
    for window in windows:
        label_dict = _default_builder.label_generator.generate(window)
        labels.append(label_dict.get("quantrascore_numeric", 50.0))
    
    return features, np.array(labels)
