"""
Training Data Quality Scorer.

Evaluates the quality and diversity of training data to ensure
the feedback loop produces high-quality samples that improve the model.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Quality assessment report for training data."""
    timestamp: str
    total_samples: int
    unique_symbols: int
    unique_scenarios: int
    
    class_balance_score: float
    diversity_score: float
    outcome_distribution_score: float
    temporal_coverage_score: float
    
    overall_quality_score: float
    
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_samples": self.total_samples,
            "unique_symbols": self.unique_symbols,
            "unique_scenarios": self.unique_scenarios,
            "class_balance_score": self.class_balance_score,
            "diversity_score": self.diversity_score,
            "outcome_distribution_score": self.outcome_distribution_score,
            "temporal_coverage_score": self.temporal_coverage_score,
            "overall_quality_score": self.overall_quality_score,
            "recommendations": self.recommendations,
            "detailed_metrics": self.detailed_metrics,
        }


class TrainingQualityScorer:
    """
    Scores training data quality to ensure feedback loop produces
    samples that actually improve model performance.
    
    Quality Dimensions:
    1. Class Balance - Are labels evenly distributed?
    2. Diversity - Are all scenarios/symbols represented?
    3. Outcome Distribution - Are win/loss/avoid labels realistic?
    4. Temporal Coverage - Is time series coverage adequate?
    
    The scorer feeds back into the generation process by recommending
    which types of samples to generate more of.
    """
    
    IDEAL_QUALITY_DISTRIBUTION = {
        "A+": 0.05,  # 5% exceptional
        "A": 0.15,   # 15% high quality
        "B": 0.35,   # 35% good
        "C": 0.30,   # 30% average
        "D": 0.15,   # 15% poor
    }
    
    IDEAL_AVOID_RATIO = 0.25
    IDEAL_RUNNER_RATIO = 0.10
    
    def __init__(self, data_paths: Optional[List[str]] = None):
        self.data_paths = data_paths or [
            "data/apexlab/chaos_simulation_samples.json",
            "data/apexlab/backtest_samples.json",
            "data/apexlab/feedback_samples.json",
        ]
    
    def load_all_samples(self) -> List[Dict[str, Any]]:
        """Load all training samples from configured paths."""
        all_samples = []
        
        for path_str in self.data_paths:
            path = Path(path_str)
            if path.exists():
                try:
                    with open(path, "r") as f:
                        samples = json.load(f)
                    all_samples.extend(samples)
                    logger.info(f"Loaded {len(samples)} samples from {path}")
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
        
        return all_samples
    
    def score(self) -> QualityReport:
        """Generate comprehensive quality report."""
        samples = self.load_all_samples()
        
        if not samples:
            return self._empty_report()
        
        class_balance = self._score_class_balance(samples)
        diversity = self._score_diversity(samples)
        outcomes = self._score_outcome_distribution(samples)
        temporal = self._score_temporal_coverage(samples)
        
        overall = (
            class_balance * 0.30 +
            diversity * 0.25 +
            outcomes * 0.30 +
            temporal * 0.15
        )
        
        recommendations = self._generate_recommendations(
            samples, class_balance, diversity, outcomes, temporal
        )
        
        detailed = self._compute_detailed_metrics(samples)
        
        return QualityReport(
            timestamp=datetime.utcnow().isoformat(),
            total_samples=len(samples),
            unique_symbols=len(set(s.get("symbol", "") for s in samples)),
            unique_scenarios=len(set(s.get("scenario_type", "") for s in samples)),
            class_balance_score=class_balance,
            diversity_score=diversity,
            outcome_distribution_score=outcomes,
            temporal_coverage_score=temporal,
            overall_quality_score=overall,
            recommendations=recommendations,
            detailed_metrics=detailed,
        )
    
    def _score_class_balance(self, samples: List[Dict]) -> float:
        """Score how well-balanced the quality tier distribution is."""
        tiers = [s.get("quality_tier", "C") for s in samples]
        tier_counts = Counter(tiers)
        total = len(tiers)
        
        if total == 0:
            return 0.0
        
        actual_dist = {k: v / total for k, v in tier_counts.items()}
        
        kl_divergence = 0.0
        for tier, ideal_ratio in self.IDEAL_QUALITY_DISTRIBUTION.items():
            actual = actual_dist.get(tier, 0.001)
            kl_divergence += ideal_ratio * np.log(ideal_ratio / max(actual, 0.001))
        
        score = max(0, 1 - kl_divergence / 2)
        return round(score, 3)
    
    def _score_diversity(self, samples: List[Dict]) -> float:
        """Score sample diversity across symbols and scenarios."""
        symbols = set(s.get("symbol", "") for s in samples)
        scenarios = set(s.get("scenario_type", "") for s in samples)
        sources = set(s.get("source", "") for s in samples)
        
        symbol_score = min(1.0, len(symbols) / 20)
        scenario_score = min(1.0, len(scenarios) / 10)
        source_score = min(1.0, len(sources) / 3)
        
        score = (symbol_score * 0.4 + scenario_score * 0.4 + source_score * 0.2)
        return round(score, 3)
    
    def _score_outcome_distribution(self, samples: List[Dict]) -> float:
        """Score realism of outcome distributions."""
        total = len(samples)
        if total == 0:
            return 0.0
        
        avoid_count = sum(1 for s in samples if s.get("avoid_trade", 0) == 1)
        runner_count = sum(1 for s in samples if s.get("hit_runner_threshold", 0) == 1)
        
        avoid_ratio = avoid_count / total
        runner_ratio = runner_count / total
        
        avoid_deviation = abs(avoid_ratio - self.IDEAL_AVOID_RATIO)
        runner_deviation = abs(runner_ratio - self.IDEAL_RUNNER_RATIO)
        
        score = max(0, 1 - (avoid_deviation + runner_deviation))
        return round(score, 3)
    
    def _score_temporal_coverage(self, samples: List[Dict]) -> float:
        """Score temporal spread of samples."""
        timestamps = []
        for s in samples:
            ts = s.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    timestamps.append(dt)
                except:
                    pass
        
        if len(timestamps) < 2:
            return 0.5
        
        timestamps.sort()
        span_days = (timestamps[-1] - timestamps[0]).days
        
        ideal_span = 365
        score = min(1.0, span_days / ideal_span)
        
        return round(score, 3)
    
    def _generate_recommendations(
        self,
        samples: List[Dict],
        class_balance: float,
        diversity: float,
        outcomes: float,
        temporal: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if len(samples) < 500:
            recommendations.append(f"Generate more samples. Current: {len(samples)}, Target: 500+")
        
        if class_balance < 0.7:
            tier_counts = Counter(s.get("quality_tier", "C") for s in samples)
            underrep = [t for t, ideal in self.IDEAL_QUALITY_DISTRIBUTION.items()
                       if tier_counts.get(t, 0) / len(samples) < ideal * 0.5]
            if underrep:
                recommendations.append(f"Quality tiers underrepresented: {underrep}")
        
        if diversity < 0.7:
            symbols = set(s.get("symbol", "") for s in samples)
            if len(symbols) < 15:
                recommendations.append(f"Add more symbol diversity. Current: {len(symbols)}")
            scenarios = set(s.get("scenario_type", "") for s in samples)
            if len(scenarios) < 8:
                recommendations.append(f"Add more scenario types. Current: {len(scenarios)}")
        
        if outcomes < 0.7:
            total = len(samples)
            avoid_ratio = sum(1 for s in samples if s.get("avoid_trade", 0) == 1) / total
            if avoid_ratio < 0.15:
                recommendations.append("Add more high-risk/avoid samples for negative training")
            elif avoid_ratio > 0.40:
                recommendations.append("Add more positive outcome samples to balance avoids")
        
        if temporal < 0.6:
            recommendations.append("Extend temporal range of samples for better generalization")
        
        if not recommendations:
            recommendations.append("Training data quality is good. Continue current approach.")
        
        return recommendations
    
    def _compute_detailed_metrics(self, samples: List[Dict]) -> Dict[str, Any]:
        """Compute detailed metrics for analysis."""
        total = len(samples)
        
        tier_counts = Counter(s.get("quality_tier", "C") for s in samples)
        scenario_counts = Counter(s.get("scenario_type", "unknown") for s in samples)
        regime_counts = Counter(s.get("regime", "unknown") for s in samples)
        source_counts = Counter(s.get("source", "unknown") for s in samples)
        
        quantra_scores = [s.get("quantra_score", 50) for s in samples]
        
        return {
            "tier_distribution": dict(tier_counts),
            "scenario_distribution": dict(scenario_counts),
            "regime_distribution": dict(regime_counts),
            "source_distribution": dict(source_counts),
            "quantra_score_stats": {
                "mean": round(np.mean(quantra_scores), 2),
                "std": round(np.std(quantra_scores), 2),
                "min": round(min(quantra_scores), 2),
                "max": round(max(quantra_scores), 2),
            },
            "avoid_ratio": sum(1 for s in samples if s.get("avoid_trade", 0) == 1) / total,
            "runner_ratio": sum(1 for s in samples if s.get("hit_runner_threshold", 0) == 1) / total,
            "monster_ratio": sum(1 for s in samples if s.get("hit_monster_runner_threshold", 0) == 1) / total,
        }
    
    def _empty_report(self) -> QualityReport:
        """Generate empty report when no data available."""
        return QualityReport(
            timestamp=datetime.utcnow().isoformat(),
            total_samples=0,
            unique_symbols=0,
            unique_scenarios=0,
            class_balance_score=0.0,
            diversity_score=0.0,
            outcome_distribution_score=0.0,
            temporal_coverage_score=0.0,
            overall_quality_score=0.0,
            recommendations=["No training data available. Run generators first."],
            detailed_metrics={},
        )
