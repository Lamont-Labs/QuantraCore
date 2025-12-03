"""
Model Trainer - Retrains models with feedback from simulations and trades

Features:
- Incremental learning with new data
- Hard negative mining for precision improvement
- Ensemble diversity maintenance
- Model versioning and rollback
"""

import pickle
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    HAS_ML = True
except ImportError:
    HAS_ML = False


class ModelTrainer:
    """Handles model retraining with feedback loops."""
    
    def __init__(self, models_dir: str = 'data/models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.current_model = None
        self.feature_cols = None
        self.training_history = []
    
    def load_current_model(self, model_path: str = None) -> Dict:
        """Load the current production model."""
        if model_path is None:
            model_path = self.models_dir / 'hyper_7model.pkl.gz'
        
        with gzip.open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        
        self.current_model = bundle
        self.feature_cols = bundle['features']
        
        return {
            'models': len(bundle['models']),
            'features': len(bundle['features']),
            'auc': bundle.get('auc', 0),
            'samples': bundle.get('samples', 0)
        }
    
    def prepare_training_data(
        self,
        simulation_results: List[Dict],
        hard_negatives: List[Dict] = None,
        trade_outcomes: List[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from multiple sources."""
        
        X_list, y_list = [], []
        
        # From simulation results
        for r in simulation_results:
            if 'features' in r and r['features']:
                feat = [r['features'].get(c, 0) for c in self.feature_cols]
                X_list.append(feat)
                y_list.append(r['hit'])
        
        print(f"[Trainer] Base data: {len(X_list)} samples")
        
        # Add hard negatives (weight them more)
        if hard_negatives:
            for hn in hard_negatives:
                if 'features' in hn and hn['features']:
                    feat = [hn['features'].get(c, 0) for c in self.feature_cols]
                    # Add multiple times to weight them
                    for _ in range(3):
                        X_list.append(feat)
                        y_list.append(0)
            print(f"[Trainer] Added {len(hard_negatives)} hard negatives (3x weighted)")
        
        # Add trade outcomes
        if trade_outcomes:
            for t in trade_outcomes:
                if 'features' in t and t['features']:
                    feat = [t['features'].get(c, 0) for c in self.feature_cols]
                    label = 1 if t.get('was_correct') else 0
                    # Trade outcomes are real data - weight highly
                    for _ in range(5):
                        X_list.append(feat)
                        y_list.append(label)
            print(f"[Trainer] Added {len(trade_outcomes)} trade outcomes (5x weighted)")
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        
        print(f"[Trainer] Total: {len(X)} samples, {y.sum()} positives")
        
        return X, y
    
    def train_new_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_models: int = 7,
        test_size: float = 0.15
    ) -> Tuple[List, Dict]:
        """Train a new ensemble model."""
        
        if not HAS_ML:
            raise ImportError("LightGBM not available")
        
        # Split data
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        split = int(len(X) * (1 - test_size))
        
        train_idx, test_idx = indices[:split], indices[split:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"[Trainer] Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Diverse model configs
        configs = [
            dict(n=800, d=3, lr=0.008, lv=8, ch=150, sub=0.5, col=0.6, lam=20),
            dict(n=600, d=4, lr=0.012, lv=10, ch=100, sub=0.6, col=0.65, lam=12),
            dict(n=500, d=5, lr=0.015, lv=14, ch=80, sub=0.65, col=0.7, lam=8),
            dict(n=700, d=3, lr=0.006, lv=8, ch=140, sub=0.5, col=0.6, lam=18),
            dict(n=650, d=4, lr=0.01, lv=10, ch=110, sub=0.55, col=0.65, lam=14),
            dict(n=550, d=4, lr=0.013, lv=12, ch=90, sub=0.6, col=0.6, lam=10),
            dict(n=750, d=3, lr=0.007, lv=8, ch=130, sub=0.5, col=0.7, lam=16),
        ][:num_models]
        
        models = []
        probs_list = []
        
        for i, cfg in enumerate(configs):
            model = lgb.LGBMClassifier(
                n_estimators=cfg['n'],
                max_depth=cfg['d'],
                learning_rate=cfg['lr'],
                num_leaves=cfg['lv'],
                min_child_samples=cfg['ch'],
                subsample=cfg['sub'],
                colsample_bytree=cfg['col'],
                reg_lambda=cfg['lam'],
                scale_pos_weight=2,
                random_state=100 + i,
                verbosity=-1,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            models.append(model)
            probs_list.append(model.predict_proba(X_test)[:, 1])
            print(f"  Model {i+1}/{len(configs)} trained")
        
        # Calculate metrics
        probs = np.array(probs_list)
        ensemble_prob = np.mean(probs, axis=0)
        
        auc = roc_auc_score(y_test, ensemble_prob)
        
        # Precision at various thresholds
        metrics = {'auc': auc, 'thresholds': {}}
        
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            pred = (ensemble_prob >= threshold).astype(int)
            if pred.sum() > 0:
                prec = precision_score(y_test, pred)
                rec = recall_score(y_test, pred)
                metrics['thresholds'][f'{int(threshold*100)}%'] = {
                    'precision': prec,
                    'recall': rec,
                    'signals': int(pred.sum())
                }
        
        # Consensus metrics
        for min_votes in [5, 6, 7]:
            votes = np.sum(probs >= 0.5, axis=0)
            pred = (votes >= min_votes).astype(int)
            if pred.sum() > 0:
                prec = precision_score(y_test, pred)
                rec = recall_score(y_test, pred)
                metrics['thresholds'][f'{min_votes}/7'] = {
                    'precision': prec,
                    'recall': rec,
                    'signals': int(pred.sum())
                }
        
        print(f"[Trainer] New model AUC: {auc:.3f}")
        
        return models, metrics
    
    def compare_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        new_models: List,
        old_models: List = None
    ) -> Dict:
        """Compare new model vs current model."""
        
        if old_models is None:
            old_models = self.current_model['models']
        
        # Old model predictions
        old_probs = np.mean([m.predict_proba(X_test)[:, 1] for m in old_models], axis=0)
        old_auc = roc_auc_score(y_test, old_probs)
        
        # New model predictions
        new_probs = np.mean([m.predict_proba(X_test)[:, 1] for m in new_models], axis=0)
        new_auc = roc_auc_score(y_test, new_probs)
        
        improvement = new_auc - old_auc
        
        comparison = {
            'old_auc': old_auc,
            'new_auc': new_auc,
            'improvement': improvement,
            'improved': improvement > 0,
            'improvement_pct': improvement / old_auc * 100 if old_auc > 0 else 0
        }
        
        print(f"[Trainer] Comparison: {old_auc:.3f} -> {new_auc:.3f} ({improvement:+.3f})")
        
        return comparison
    
    def save_model(
        self,
        models: List,
        metrics: Dict,
        version: str = None
    ) -> Path:
        """Save the new model."""
        
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        bundle = {
            'models': models,
            'features': self.feature_cols,
            'auc': metrics['auc'],
            'metrics': metrics,
            'version': version,
            'trained_at': datetime.now().isoformat()
        }
        
        # Save versioned copy
        versioned_path = self.models_dir / f'model_v{version}.pkl.gz'
        with gzip.open(versioned_path, 'wb') as f:
            pickle.dump(bundle, f)
        
        print(f"[Trainer] Saved to {versioned_path}")
        
        return versioned_path
    
    def promote_model(self, version_path: Path) -> Path:
        """Promote a versioned model to production."""
        
        prod_path = self.models_dir / 'hyper_7model.pkl.gz'
        
        # Backup current production model
        if prod_path.exists():
            backup_path = self.models_dir / f'hyper_7model_backup_{datetime.now().strftime("%Y%m%d")}.pkl.gz'
            import shutil
            shutil.copy(prod_path, backup_path)
            print(f"[Trainer] Backed up to {backup_path}")
        
        # Copy new model to production
        import shutil
        shutil.copy(version_path, prod_path)
        
        print(f"[Trainer] Promoted {version_path} to production")
        
        return prod_path
    
    def rollback(self, backup_path: Path) -> bool:
        """Rollback to a previous model version."""
        
        prod_path = self.models_dir / 'hyper_7model.pkl.gz'
        
        if backup_path.exists():
            import shutil
            shutil.copy(backup_path, prod_path)
            print(f"[Trainer] Rolled back to {backup_path}")
            return True
        
        return False
