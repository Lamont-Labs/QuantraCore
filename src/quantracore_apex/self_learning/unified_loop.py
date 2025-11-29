"""
Unified Self-Learning Loop.

The master orchestrator that connects all feedback loops together:
1. Chaos Generator → Training Data
2. Backtest Generator → Training Data
3. Alpha Factory Feedback → Training Data
4. Quality Scorer → Recommendations
5. Auto Retrain → Model Updates
6. Improved Model → Better Predictions → Better Signals → Better Outcomes

This creates a truly self-improving system.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict

from .chaos_generator import ChaosTrainingGenerator
from .backtest_generator import BacktestTrainingGenerator
from .quality_scorer import TrainingQualityScorer, QualityReport
from .auto_retrain import AutoRetrainTrigger, RetrainDecision

logger = logging.getLogger(__name__)


@dataclass
class LoopCycleResult:
    """Result of a single learning loop cycle."""
    cycle_id: str
    timestamp: str
    
    chaos_samples_generated: int
    backtest_samples_generated: int
    feedback_samples_collected: int
    total_new_samples: int
    
    quality_before: float
    quality_after: float
    quality_improvement: float
    
    retrain_triggered: bool
    retrain_result: Optional[Dict[str, Any]]
    
    recommendations: List[str]
    cycle_duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UnifiedLearningLoop:
    """
    The unified self-learning loop that orchestrates all components.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Unified Learning Loop                         │
    │                                                                  │
    │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
    │   │    Chaos     │     │   Backtest   │     │   Feedback   │   │
    │   │  Generator   │     │  Generator   │     │    Loop      │   │
    │   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘   │
    │          │                    │                     │           │
    │          └────────────────────┼─────────────────────┘           │
    │                               ▼                                  │
    │                    ┌──────────────────┐                         │
    │                    │  Training Data   │                         │
    │                    │     Pool         │                         │
    │                    └────────┬─────────┘                         │
    │                             │                                    │
    │                             ▼                                    │
    │                    ┌──────────────────┐                         │
    │                    │  Quality Scorer  │                         │
    │                    └────────┬─────────┘                         │
    │                             │                                    │
    │           ┌─────────────────┼─────────────────┐                 │
    │           ▼                 ▼                 ▼                 │
    │   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
    │   │ Recommendations│ │Auto Retrain │ │   Metrics    │           │
    │   │  (Generate   │ │  Trigger    │ │   Dashboard  │           │
    │   │   More of X) │ └──────┬───────┘ └──────────────┘           │
    │   └──────────────┘        │                                     │
    │                           ▼                                     │
    │                    ┌──────────────────┐                         │
    │                    │  ApexLab Train   │                         │
    │                    └────────┬─────────┘                         │
    │                             │                                    │
    │                             ▼                                    │
    │                    ┌──────────────────┐                         │
    │                    │  ApexCore Model  │──────┐                  │
    │                    └──────────────────┘      │                  │
    │                                              │                  │
    │                    ┌──────────────────┐      │                  │
    │                    │ Better Predictions│◄─────┘                  │
    │                    └────────┬─────────┘                         │
    │                             │                                    │
    │                             ▼                                    │
    │                    ┌──────────────────┐                         │
    │                    │ Better Outcomes  │───► Back to Feedback    │
    │                    └──────────────────┘                         │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        data_dir: str = "data/apexlab",
        chaos_runs_per_combo: int = 2,
        backtest_step_size: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.chaos_runs = chaos_runs_per_combo
        self.backtest_step = backtest_step_size
        
        self.chaos_generator = ChaosTrainingGenerator(
            output_path=str(self.data_dir / "chaos_simulation_samples.json")
        )
        self.backtest_generator = BacktestTrainingGenerator(
            output_path=str(self.data_dir / "backtest_samples.json")
        )
        self.quality_scorer = TrainingQualityScorer()
        self.retrain_trigger = AutoRetrainTrigger(
            state_path=str(self.data_dir / "retrain_state.json")
        )
        
        self._cycle_count = 0
        self._history_path = self.data_dir / "loop_history.json"
    
    def run_cycle(
        self,
        generate_chaos: bool = True,
        generate_backtest: bool = True,
        check_retrain: bool = True,
        train_function: Optional[Callable] = None,
    ) -> LoopCycleResult:
        """
        Run a complete learning loop cycle.
        
        Args:
            generate_chaos: Generate new chaos simulation samples
            generate_backtest: Generate new backtest samples
            check_retrain: Check if retraining should be triggered
            train_function: Function to call for training (if None, skip)
            
        Returns:
            LoopCycleResult with cycle metrics
        """
        import time
        start_time = time.time()
        
        self._cycle_count += 1
        cycle_id = f"cycle_{self._cycle_count}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting learning cycle: {cycle_id}")
        
        quality_before = self.quality_scorer.score()
        
        chaos_count = 0
        backtest_count = 0
        feedback_count = 0
        
        if generate_chaos:
            logger.info("Generating chaos simulation samples...")
            chaos_samples = self.chaos_generator.generate_batch(
                runs_per_combo=self.chaos_runs
            )
            chaos_count = len(chaos_samples)
            self.chaos_generator.save_samples(chaos_samples)
        
        if generate_backtest:
            logger.info("Generating backtest samples...")
            backtest_samples = self.backtest_generator.generate_batch(
                step_size=self.backtest_step
            )
            backtest_count = len(backtest_samples)
            self.backtest_generator.save_samples(backtest_samples)
        
        feedback_path = self.data_dir / "feedback_samples.json"
        if feedback_path.exists():
            with open(feedback_path) as f:
                feedback_samples = json.load(f)
            feedback_count = len(feedback_samples)
        
        quality_after = self.quality_scorer.score()
        
        retrain_triggered = False
        retrain_result = None
        
        if check_retrain and train_function:
            decision = self.retrain_trigger.check()
            if decision.should_retrain:
                retrain_triggered = True
                retrain_result = self.retrain_trigger.trigger_retrain(train_function)
        
        duration = time.time() - start_time
        
        result = LoopCycleResult(
            cycle_id=cycle_id,
            timestamp=datetime.utcnow().isoformat(),
            chaos_samples_generated=chaos_count,
            backtest_samples_generated=backtest_count,
            feedback_samples_collected=feedback_count,
            total_new_samples=chaos_count + backtest_count,
            quality_before=quality_before.overall_quality_score,
            quality_after=quality_after.overall_quality_score,
            quality_improvement=quality_after.overall_quality_score - quality_before.overall_quality_score,
            retrain_triggered=retrain_triggered,
            retrain_result=retrain_result,
            recommendations=quality_after.recommendations,
            cycle_duration_seconds=round(duration, 2),
        )
        
        self._save_cycle_result(result)
        
        logger.info(f"Cycle complete: {chaos_count + backtest_count} new samples, "
                   f"quality {quality_before.overall_quality_score:.2f} → {quality_after.overall_quality_score:.2f}")
        
        return result
    
    def run_quick_cycle(self) -> LoopCycleResult:
        """Run a quick cycle with minimal generation for testing."""
        return self.run_cycle(
            generate_chaos=True,
            generate_backtest=False,
            check_retrain=False,
        )
    
    def run_full_cycle(self, train_function: Optional[Callable] = None) -> LoopCycleResult:
        """Run a full cycle with all generators and retraining."""
        return self.run_cycle(
            generate_chaos=True,
            generate_backtest=True,
            check_retrain=True,
            train_function=train_function,
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current loop status."""
        quality = self.quality_scorer.score()
        retrain_decision = self.retrain_trigger.check()
        
        return {
            "cycle_count": self._cycle_count,
            "quality_report": quality.to_dict(),
            "retrain_decision": {
                "should_retrain": retrain_decision.should_retrain,
                "reason": retrain_decision.reason,
                "priority": retrain_decision.priority,
                "metrics": retrain_decision.metrics,
            },
            "data_sources": {
                "chaos": str(self.chaos_generator.output_path),
                "backtest": str(self.backtest_generator.output_path),
                "feedback": str(self.data_dir / "feedback_samples.json"),
            },
        }
    
    def get_training_data_summary(self) -> Dict[str, Any]:
        """Get summary of all training data."""
        chaos_stats = self.chaos_generator.get_sample_stats()
        
        backtest_count = 0
        backtest_path = self.backtest_generator.output_path
        if backtest_path.exists():
            with open(backtest_path) as f:
                backtest_count = len(json.load(f))
        
        feedback_count = 0
        feedback_path = self.data_dir / "feedback_samples.json"
        if feedback_path.exists():
            with open(feedback_path) as f:
                feedback_count = len(json.load(f))
        
        return {
            "chaos_samples": chaos_stats.get("total", 0),
            "backtest_samples": backtest_count,
            "feedback_samples": feedback_count,
            "total_samples": chaos_stats.get("total", 0) + backtest_count + feedback_count,
            "chaos_by_scenario": chaos_stats.get("by_scenario", {}),
            "chaos_by_quality": chaos_stats.get("by_quality", {}),
        }
    
    def _save_cycle_result(self, result: LoopCycleResult) -> None:
        """Save cycle result to history."""
        history = []
        
        if self._history_path.exists():
            with open(self._history_path) as f:
                history = json.load(f)
        
        history.append(result.to_dict())
        
        if len(history) > 100:
            history = history[-100:]
        
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._history_path, "w") as f:
            json.dump(history, f, indent=2)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get cycle history."""
        if self._history_path.exists():
            with open(self._history_path) as f:
                return json.load(f)
        return []


def run_learning_loop_demo():
    """Run a demo of the learning loop."""
    print("=" * 60)
    print("QuantraCore Apex - Unified Self-Learning Loop Demo")
    print("=" * 60)
    
    loop = UnifiedLearningLoop(
        chaos_runs_per_combo=1,
        backtest_step_size=20,
    )
    
    print("\nRunning quick learning cycle...")
    result = loop.run_quick_cycle()
    
    print(f"\nCycle Results:")
    print(f"  Chaos samples: {result.chaos_samples_generated}")
    print(f"  Backtest samples: {result.backtest_samples_generated}")
    print(f"  Quality: {result.quality_before:.2f} → {result.quality_after:.2f}")
    print(f"  Duration: {result.cycle_duration_seconds:.1f}s")
    
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")
    
    print("\nTraining Data Summary:")
    summary = loop.get_training_data_summary()
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  - Chaos: {summary['chaos_samples']}")
    print(f"  - Backtest: {summary['backtest_samples']}")
    print(f"  - Feedback: {summary['feedback_samples']}")
    
    print("\n" + "=" * 60)
    print("Learning loop demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_learning_loop_demo()
