"""
AutoLearner - Orchestrates the entire automated learning pipeline

Features:
- Coordinates trade tracking, simulation, and model training
- Runs overnight training sessions
- Monitors performance and triggers retraining
- Maintains learning history and model versions
"""

import os
import schedule
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from .trade_tracker import TradeTracker
from .simulation_engine import SimulationEngine
from .model_trainer import ModelTrainer


class AutoLearner:
    """
    Automated learning system that continuously improves the model.
    
    Usage:
        learner = AutoLearner()
        learner.start()  # Runs in background
        
        # Or run manually:
        learner.run_learning_cycle()
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        cache_dir: str = 'data/cache/polygon/day',
        models_dir: str = 'data/models'
    ):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        
        # Initialize components
        self.tracker = TradeTracker(self.db_url)
        self.simulator = SimulationEngine(cache_dir)
        self.trainer = ModelTrainer(models_dir)
        
        # Learning config
        self.min_trades_for_learning = 10
        self.simulation_files = 300
        self.improvement_threshold = 0.005  # 0.5% AUC improvement required
        
        # State
        self.is_running = False
        self.last_learning_run = None
        self.learning_thread = None
        
        print("[AutoLearner] Initialized")
    
    def record_trade_entry(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        side: str,
        model_probability: float,
        model_votes: int,
        features: Dict
    ) -> int:
        """Record a new trade entry for learning."""
        return self.tracker.record_entry(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            side=side,
            model_probability=model_probability,
            model_votes=model_votes,
            features=features
        )
    
    def record_trade_exit(
        self,
        symbol: str,
        exit_price: float,
        max_price: float,
        min_price: float
    ) -> Dict:
        """Record trade exit for learning."""
        return self.tracker.record_exit(
            symbol=symbol,
            exit_price=exit_price,
            max_price=max_price,
            min_price=min_price
        )
    
    def run_simulation(self, num_files: int = None) -> Dict:
        """Run a historical simulation."""
        
        if num_files is None:
            num_files = self.simulation_files
        
        print(f"\n{'='*60}")
        print("RUNNING SIMULATION")
        print(f"{'='*60}\n")
        
        # Load current model
        model_info = self.trainer.load_current_model()
        current_auc = self.simulator.load_model()
        print(f"Current model AUC: {current_auc:.3f}")
        
        # Run simulation
        results, stats = self.simulator.run_simulation(num_files=num_files)
        
        # Print results
        print(f"\nSimulation Results:")
        print(f"  Total signals: {stats['total_signals']:,}")
        print(f"  Actual runners: {stats['actual_runners']:,}")
        print(f"  Predicted runners: {stats['predicted_runners']:,}")
        
        if 'by_probability' in stats:
            print("\nBy Probability Threshold:")
            for thresh, data in stats['by_probability'].items():
                print(f"  {thresh}: {data['signals']} signals, {data['precision']:.0%} precision")
        
        if 'by_votes' in stats:
            print("\nBy Consensus:")
            for thresh, data in stats['by_votes'].items():
                print(f"  {thresh}: {data['signals']} signals, {data['precision']:.0%} precision")
        
        return {
            'results': results,
            'stats': stats,
            'current_auc': current_auc
        }
    
    def run_learning_cycle(self, force: bool = False) -> Dict:
        """Run a complete learning cycle."""
        
        print(f"\n{'='*60}")
        print("AUTOMATED LEARNING CYCLE")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # 1. Get trade outcomes
        print("Step 1: Fetching trade outcomes...")
        trades, trade_stats = self.tracker.get_learning_data(days=30)
        print(f"  Found {trade_stats['total']} trades, {trade_stats['precision']:.0%} precision")
        
        # 2. Get hard negatives from trades
        hard_negatives_trades = self.tracker.get_hard_negatives()
        print(f"  Hard negatives from trades: {len(hard_negatives_trades)}")
        
        # 3. Run simulation
        print("\nStep 2: Running simulation...")
        sim_data = self.run_simulation()
        
        # 4. Get hard examples from simulation
        print("\nStep 3: Extracting hard examples...")
        hard_positives, hard_negatives_sim = self.simulator.get_hard_examples(sim_data['results'])
        
        # Combine hard negatives
        all_hard_negatives = hard_negatives_trades + [
            {'features': r['features'], 'probability': r['probability'], 'label': 0}
            for r in hard_negatives_sim
        ]
        
        # 5. Prepare training data
        print("\nStep 4: Preparing training data...")
        X, y = self.trainer.prepare_training_data(
            simulation_results=sim_data['results'],
            hard_negatives=all_hard_negatives,
            trade_outcomes=trades
        )
        
        # Check if we have enough data
        if len(X) < 1000 and not force:
            print(f"\nInsufficient data ({len(X)} samples). Skipping retraining.")
            return {
                'status': 'skipped',
                'reason': 'insufficient_data',
                'samples': len(X)
            }
        
        # 6. Train new model
        print("\nStep 5: Training new model...")
        new_models, metrics = self.trainer.train_new_model(X, y)
        
        # 7. Compare models
        print("\nStep 6: Comparing models...")
        
        # Use simulation data for comparison
        test_size = int(len(X) * 0.15)
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        comparison = self.trainer.compare_models(X_test, y_test, new_models)
        
        # 8. Decide whether to promote
        should_promote = comparison['improvement'] >= self.improvement_threshold
        
        print(f"\nStep 7: Decision...")
        print(f"  Old AUC: {comparison['old_auc']:.4f}")
        print(f"  New AUC: {comparison['new_auc']:.4f}")
        print(f"  Improvement: {comparison['improvement']:+.4f}")
        print(f"  Threshold: {self.improvement_threshold}")
        print(f"  Decision: {'PROMOTE' if should_promote else 'KEEP CURRENT'}")
        
        result = {
            'status': 'completed',
            'old_auc': comparison['old_auc'],
            'new_auc': comparison['new_auc'],
            'improvement': comparison['improvement'],
            'samples_used': len(X),
            'hard_negatives': len(all_hard_negatives),
            'trade_outcomes': len(trades),
            'metrics': metrics
        }
        
        if should_promote or force:
            # Save and promote
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.trainer.save_model(new_models, metrics, version)
            
            if should_promote:
                self.trainer.promote_model(save_path)
                result['promoted'] = True
                result['model_path'] = str(save_path)
                print(f"\n*** NEW MODEL PROMOTED ***")
            else:
                result['promoted'] = False
                print(f"\nModel saved but not promoted (forced run)")
        else:
            result['promoted'] = False
            print(f"\nModel not improved enough. Keeping current.")
        
        # Record learning session
        self.tracker.record_learning_session(
            trades_analyzed=len(trades),
            correct=trade_stats['correct'],
            incorrect=trade_stats['incorrect'],
            precision_before=comparison['old_auc'],
            precision_after=comparison['new_auc'],
            auc_before=comparison['old_auc'],
            auc_after=comparison['new_auc'],
            model_version=result.get('model_path', 'not_promoted'),
            training_samples=len(X),
            notes=f"Hard negatives: {len(all_hard_negatives)}"
        )
        
        self.last_learning_run = datetime.now()
        
        print(f"\n{'='*60}")
        print("LEARNING CYCLE COMPLETE")
        print(f"{'='*60}\n")
        
        return result
    
    def _scheduled_run(self):
        """Run learning cycle on schedule."""
        try:
            self.run_learning_cycle()
        except Exception as e:
            print(f"[AutoLearner] Scheduled run error: {e}")
    
    def start(self, run_time: str = "02:00"):
        """Start the automated learning scheduler."""
        
        if self.is_running:
            print("[AutoLearner] Already running")
            return
        
        print(f"[AutoLearner] Starting scheduler (daily at {run_time})")
        
        # Schedule daily learning
        schedule.every().day.at(run_time).do(self._scheduled_run)
        
        self.is_running = True
        
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)
        
        self.learning_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.learning_thread.start()
        
        print("[AutoLearner] Scheduler started")
    
    def stop(self):
        """Stop the automated learning scheduler."""
        self.is_running = False
        schedule.clear()
        print("[AutoLearner] Stopped")
    
    def get_status(self) -> Dict:
        """Get current learning system status."""
        
        trades, trade_stats = self.tracker.get_learning_data(days=7)
        
        return {
            'is_running': self.is_running,
            'last_run': self.last_learning_run.isoformat() if self.last_learning_run else None,
            'recent_trades': len(trades),
            'trade_precision': trade_stats['precision'],
            'next_run': str(schedule.next_run()) if schedule.jobs else None
        }
    
    def quick_check(self) -> Dict:
        """Quick check of model performance on recent data."""
        
        print("Running quick performance check...")
        
        # Load model
        self.simulator.load_model()
        
        # Run small simulation
        results, stats = self.simulator.run_simulation(num_files=50)
        
        return {
            'signals_tested': len(results),
            'stats': stats
        }


def run_overnight_training():
    """Convenience function to run overnight training."""
    
    learner = AutoLearner()
    result = learner.run_learning_cycle()
    
    print("\n" + "="*60)
    print("OVERNIGHT TRAINING COMPLETE")
    print("="*60)
    print(f"Samples used: {result.get('samples_used', 0):,}")
    print(f"AUC improvement: {result.get('improvement', 0):+.4f}")
    print(f"Model promoted: {result.get('promoted', False)}")
    print("="*60)
    
    return result


if __name__ == '__main__':
    run_overnight_training()
