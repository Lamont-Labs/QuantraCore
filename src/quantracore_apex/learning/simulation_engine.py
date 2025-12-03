"""
Simulation Engine - Runs historical backtests at 1000x speed

Features:
- Replay years of data in minutes
- Track predictions vs outcomes
- Generate training data from simulations
- Identify model failure patterns
"""

import pickle
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')


class SimulationEngine:
    """Runs high-speed backtests to generate learning data."""
    
    def __init__(self, cache_dir: str = 'data/cache/polygon/day'):
        self.cache_dir = Path(cache_dir)
        self.results_dir = Path('data/learning/simulations')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models = None
        self.feature_cols = None
    
    def load_model(self, model_path: str = 'data/models/hyper_7model.pkl.gz'):
        """Load the ensemble model for simulation."""
        with gzip.open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        self.models = bundle['models']
        self.feature_cols = bundle['features']
        return bundle.get('auc', 0)
    
    def _extract_features(self, c, h, l, v, o, idx) -> Optional[Dict]:
        """Extract features for a single data point."""
        if idx < 60 or c[idx] <= 0:
            return None
            
        wc = c[idx-60:idx+1]
        ret = np.diff(wc) / (wc[:-1] + 1e-8)
        
        f = {}
        for p in [3, 5, 10, 20]:
            f[f'vol_{p}'] = np.std(ret[-p:]) if len(ret) >= p else np.std(ret)
        f['vol_comp'] = f['vol_5'] / (f['vol_20'] + 1e-8)
        
        for p in [3, 5, 10, 20]:
            f[f'mom_{p}'] = (wc[-1] - wc[-p-1]) / (wc[-p-1] + 1e-8)
        f['mom_acc'] = f['mom_3'] - f['mom_5']
        
        f['vol_spk'] = np.max(v[idx-3:idx+1]) / (np.mean(v[idx-20:idx+1]) + 1)
        f['vol_r3'] = np.mean(v[idx-3:idx+1]) / (np.mean(v[idx-20:idx+1]) + 1)
        f['h52'] = np.max(h[max(0, idx-252):idx])
        f['brk'] = wc[-1] / (f['h52'] + 1e-8)
        f['price'] = wc[-1]
        f['penny'] = 1 if wc[-1] < 5 else 0
        f['gap'] = (o[idx] - c[idx-1]) / (c[idx-1] + 1e-8)
        f['rng'] = (h[idx] - l[idx]) / (c[idx] + 1e-8)
        f['up'] = sum(1 for i in range(-5, 0) if wc[i] > wc[i-1])
        
        return f
    
    def _process_file(self, file_path: Path) -> List[Dict]:
        """Process a single file for simulation."""
        results = []
        
        try:
            df = pd.read_parquet(file_path)
            if len(df) < 70:
                return results
            
            symbol = file_path.stem
            c = df['close'].values
            h = df['high'].values
            l = df['low'].values
            v = df['volume'].values
            o = df['open'].values
            
            date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
            dates = df[date_col].astype(str).str[:10].values
            
            for idx in range(60, len(df) - 5, 2):  # Every 2 days for speed
                feat = self._extract_features(c, h, l, v, o, idx)
                if feat is None:
                    continue
                
                # Calculate actual outcome
                max_gain = (np.max(h[idx+1:idx+6]) - c[idx]) / c[idx]
                actual_hit = 1 if max_gain >= 0.50 else 0
                
                results.append({
                    'symbol': symbol,
                    'date': dates[idx],
                    'price': c[idx],
                    'features': feat,
                    'max_gain': max_gain,
                    'hit': actual_hit
                })
                
        except Exception:
            pass
        
        return results
    
    def run_simulation(
        self,
        num_files: int = 500,
        parallel_workers: int = 4
    ) -> Tuple[List[Dict], Dict]:
        """Run historical simulation on cached data."""
        
        if self.models is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        files = list(self.cache_dir.glob('*.parquet'))[:num_files]
        print(f"[SimEngine] Running simulation on {len(files)} files...")
        
        # Process files in parallel
        all_results = []
        batch_size = 100
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                batch_results = list(executor.map(self._process_file, batch))
                for r in batch_results:
                    all_results.extend(r)
            print(f"  Processed {min(i+batch_size, len(files))}/{len(files)}...")
        
        print(f"[SimEngine] Generated {len(all_results)} simulation points")
        
        # Run model predictions on all results
        print("[SimEngine] Running model predictions...")
        
        for result in all_results:
            X = np.array([[result['features'].get(c, 0) for c in self.feature_cols]])
            probs = [m.predict_proba(X)[0][1] for m in self.models]
            result['probability'] = np.mean(probs)
            result['votes'] = sum(1 for p in probs if p >= 0.5)
            result['predicted'] = 1 if result['probability'] >= 0.5 else 0
            result['correct'] = result['predicted'] == result['hit']
        
        # Calculate statistics
        stats = self._calculate_stats(all_results)
        
        return all_results, stats
    
    def _calculate_stats(self, results: List[Dict]) -> Dict:
        """Calculate simulation statistics."""
        
        stats = {
            'total_signals': len(results),
            'actual_runners': sum(r['hit'] for r in results),
            'predicted_runners': sum(r['predicted'] for r in results),
            'correct': sum(r['correct'] for r in results),
        }
        
        # By probability tier
        stats['by_probability'] = {}
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            tier = [r for r in results if r['probability'] >= threshold]
            if tier:
                hits = sum(r['hit'] for r in tier)
                stats['by_probability'][f'>={int(threshold*100)}%'] = {
                    'signals': len(tier),
                    'hits': hits,
                    'precision': hits / len(tier),
                    'avg_gain': np.mean([r['max_gain'] for r in tier])
                }
        
        # By vote consensus
        stats['by_votes'] = {}
        for min_votes in [5, 6, 7]:
            tier = [r for r in results if r['votes'] >= min_votes]
            if tier:
                hits = sum(r['hit'] for r in tier)
                stats['by_votes'][f'{min_votes}/7'] = {
                    'signals': len(tier),
                    'hits': hits,
                    'precision': hits / len(tier) if tier else 0,
                    'avg_gain': np.mean([r['max_gain'] for r in tier])
                }
        
        return stats
    
    def get_hard_examples(self, results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Extract hard positives and hard negatives for retraining."""
        
        # Hard negatives: High confidence but wrong
        hard_negatives = [
            r for r in results 
            if r['probability'] >= 0.5 and r['hit'] == 0
        ]
        
        # Hard positives: Low confidence but should have been positive
        hard_positives = [
            r for r in results 
            if r['probability'] < 0.5 and r['hit'] == 1
        ]
        
        print(f"[SimEngine] Found {len(hard_negatives)} hard negatives, {len(hard_positives)} hard positives")
        
        return hard_positives, hard_negatives
    
    def save_results(self, results: List[Dict], stats: Dict, name: str = None):
        """Save simulation results for later analysis."""
        
        if name is None:
            name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        save_path = self.results_dir / f'sim_{name}.pkl.gz'
        
        with gzip.open(save_path, 'wb') as f:
            pickle.dump({
                'results': results,
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        print(f"[SimEngine] Results saved to {save_path}")
        return save_path
