#!/usr/bin/env python3
"""
Run the Unified Self-Learning Loop.

This script runs the complete self-improving feedback loop that:
1. Generates chaos simulation training samples
2. Generates historical backtest training samples
3. Evaluates training data quality
4. Triggers retraining when conditions are met

Usage:
    python scripts/run_learning_loop.py [--quick] [--full] [--status]
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantracore_apex.self_learning import UnifiedLearningLoop

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run the Self-Learning Loop")
    parser.add_argument("--quick", action="store_true", help="Run quick cycle (chaos only)")
    parser.add_argument("--full", action="store_true", help="Run full cycle with all generators")
    parser.add_argument("--status", action="store_true", help="Show current status only")
    parser.add_argument("--summary", action="store_true", help="Show training data summary")
    args = parser.parse_args()
    
    print("=" * 70)
    print("QuantraCore Apex - Unified Self-Learning Loop")
    print("=" * 70)
    
    loop = UnifiedLearningLoop(
        chaos_runs_per_combo=2,
        backtest_step_size=10,
    )
    
    if args.status:
        status = loop.get_status()
        print("\nCurrent Status:")
        print(f"  Cycle count: {status['cycle_count']}")
        print(f"  Quality score: {status['quality_report']['overall_quality_score']:.2f}")
        print(f"  Total samples: {status['quality_report']['total_samples']}")
        print(f"\nRetrain Decision:")
        print(f"  Should retrain: {status['retrain_decision']['should_retrain']}")
        print(f"  Reason: {status['retrain_decision']['reason']}")
        print(f"  Priority: {status['retrain_decision']['priority']}")
        return
    
    if args.summary:
        summary = loop.get_training_data_summary()
        print("\nTraining Data Summary:")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  - Chaos simulation: {summary['chaos_samples']}")
        print(f"  - Historical backtest: {summary['backtest_samples']}")
        print(f"  - Feedback loop: {summary['feedback_samples']}")
        print(f"\nChaos by scenario:")
        for scenario, count in summary.get('chaos_by_scenario', {}).items():
            print(f"    {scenario}: {count}")
        print(f"\nChaos by quality tier:")
        for tier, count in summary.get('chaos_by_quality', {}).items():
            print(f"    {tier}: {count}")
        return
    
    if args.quick:
        print("\nRunning quick learning cycle (chaos simulation only)...")
        result = loop.run_quick_cycle()
    else:
        print("\nRunning full learning cycle...")
        result = loop.run_cycle(
            generate_chaos=True,
            generate_backtest=True,
            check_retrain=True,
        )
    
    print(f"\n{'=' * 70}")
    print("Cycle Results")
    print(f"{'=' * 70}")
    print(f"  Cycle ID: {result.cycle_id}")
    print(f"  Duration: {result.cycle_duration_seconds:.1f} seconds")
    print(f"\nSamples Generated:")
    print(f"  Chaos simulation: {result.chaos_samples_generated}")
    print(f"  Historical backtest: {result.backtest_samples_generated}")
    print(f"  Feedback collected: {result.feedback_samples_collected}")
    print(f"  Total new: {result.total_new_samples}")
    print(f"\nQuality Metrics:")
    print(f"  Before: {result.quality_before:.3f}")
    print(f"  After: {result.quality_after:.3f}")
    print(f"  Improvement: {result.quality_improvement:+.3f}")
    print(f"\nRetrain Status:")
    print(f"  Triggered: {result.retrain_triggered}")
    if result.retrain_result:
        print(f"  Result: {result.retrain_result}")
    
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")
    
    summary = loop.get_training_data_summary()
    print(f"\nTotal Training Data:")
    print(f"  {summary['total_samples']} samples across all sources")
    
    print(f"\n{'=' * 70}")
    print("Learning cycle complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
