"""
Adversarial Learning Engine.

Learns from battles against institutions to improve our signals.

METHODOLOGY:
1. Analyze patterns where institutions outperformed
2. Identify systematic weaknesses in our signals
3. Generate improvement vectors
4. Feed insights back to ApexLab for training

COMPLIANCE:
- All learning from public data only
- No forward-looking or predictive claims
- Research and educational purposes only
"""

import logging
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

from ..models import (
    BattleResult,
    BattleOutcome,
    StrategyFingerprint,
    AdversarialInsight,
    ComplianceStatus,
)

logger = logging.getLogger(__name__)


class AdversarialLearner:
    """
    Learns from battle results to improve our signals.
    
    Analyzes patterns in where institutions outperformed us
    and generates improvement insights.
    """
    
    def __init__(self):
        self._battle_history: List[BattleResult] = []
        self._insights: List[AdversarialInsight] = []
        self._learning_metrics: Dict[str, Any] = defaultdict(float)
        
        logger.info("[AdversarialLearner] Initialized for pattern learning")
    
    def ingest_battle_result(self, result: BattleResult) -> None:
        """Ingest a battle result for learning."""
        self._battle_history.append(result)
        
        if result.outcome == BattleOutcome.LOSS:
            self._analyze_loss(result)
        elif result.outcome == BattleOutcome.WIN:
            self._analyze_win(result)
    
    def ingest_batch(self, results: List[BattleResult]) -> None:
        """Ingest a batch of battle results."""
        for result in results:
            self.ingest_battle_result(result)
        
        if len(results) >= 10:
            self._generate_batch_insights(results)
    
    def _analyze_loss(self, result: BattleResult) -> None:
        """Analyze a loss to extract lessons."""
        self._learning_metrics["losses"] += 1
        self._learning_metrics["total_negative_alpha"] += result.alpha_generated
        
        if result.alpha_generated < -5:
            self._learning_metrics["significant_losses"] += 1
    
    def _analyze_win(self, result: BattleResult) -> None:
        """Analyze a win to understand what worked."""
        self._learning_metrics["wins"] += 1
        self._learning_metrics["total_positive_alpha"] += result.alpha_generated
        
        if result.alpha_generated > 5:
            self._learning_metrics["significant_wins"] += 1
    
    def _generate_batch_insights(self, results: List[BattleResult]) -> None:
        """Generate insights from a batch of results."""
        losses = [r for r in results if r.outcome == BattleOutcome.LOSS]
        wins = [r for r in results if r.outcome == BattleOutcome.WIN]
        
        if len(losses) >= 3:
            loss_patterns = self._find_loss_patterns(losses)
            for pattern in loss_patterns:
                self._create_insight(
                    category="loss_pattern",
                    insight_type=pattern["type"],
                    description=pattern["description"],
                    improvement_vector=pattern.get("improvement", {}),
                    sample_size=len(losses),
                )
        
        if len(wins) >= 3:
            win_patterns = self._find_win_patterns(wins)
            for pattern in win_patterns:
                self._create_insight(
                    category="win_pattern",
                    insight_type=pattern["type"],
                    description=pattern["description"],
                    improvement_vector=pattern.get("reinforcement", {}),
                    sample_size=len(wins),
                )
    
    def _find_loss_patterns(self, losses: List[BattleResult]) -> List[Dict[str, Any]]:
        """Find patterns in losing battles."""
        patterns = []
        
        avg_score = sum(
            r.scenario.our_quantrascore for r in losses
        ) / len(losses) if losses else 0
        
        if avg_score > 70:
            patterns.append({
                "type": "high_conviction_failure",
                "description": (
                    f"High conviction signals (avg score {avg_score:.0f}) "
                    "are underperforming vs institutions"
                ),
                "improvement": {
                    "action": "review_high_conviction_criteria",
                    "priority": "high",
                    "suggested_adjustment": -5,
                },
            })
        
        timing_losses = [r for r in losses if r.timing_advantage < -0.2]
        if len(timing_losses) > len(losses) * 0.5:
            patterns.append({
                "type": "timing_weakness",
                "description": (
                    "Institutions consistently have better timing on entries"
                ),
                "improvement": {
                    "action": "improve_entry_timing",
                    "priority": "medium",
                    "suggested_adjustment": "delay_entries",
                },
            })
        
        return patterns
    
    def _find_win_patterns(self, wins: List[BattleResult]) -> List[Dict[str, Any]]:
        """Find patterns in winning battles."""
        patterns = []
        
        avg_alpha = sum(r.alpha_generated for r in wins) / len(wins) if wins else 0
        
        if avg_alpha > 10:
            patterns.append({
                "type": "strong_alpha_generation",
                "description": (
                    f"Strong alpha generation ({avg_alpha:.1f}%) vs institutions"
                ),
                "reinforcement": {
                    "action": "maintain_strategy",
                    "confidence": min(0.9, avg_alpha / 20),
                },
            })
        
        return patterns
    
    def _create_insight(
        self,
        category: str,
        insight_type: str,
        description: str,
        improvement_vector: Dict[str, Any],
        sample_size: int,
    ) -> AdversarialInsight:
        """Create and store an insight."""
        insight = AdversarialInsight(
            insight_id=f"INS-{uuid.uuid4().hex[:8]}",
            generated_at=datetime.utcnow(),
            category=category,
            insight_type=insight_type,
            description=description,
            confidence=min(0.9, sample_size / 20),
            sample_size=sample_size,
            improvement_vector=improvement_vector,
            compliance_status=ComplianceStatus.RESEARCH_ONLY,
        )
        
        self._insights.append(insight)
        logger.info(f"[AdversarialLearner] Generated insight: {insight_type}")
        
        return insight
    
    def learn_from_fingerprints(
        self,
        fingerprints: List[StrategyFingerprint],
    ) -> List[AdversarialInsight]:
        """
        Learn from institutional strategy fingerprints.
        
        Analyzes what makes top institutions successful.
        """
        if not fingerprints:
            return []
        
        insights = []
        
        avg_concentration = sum(
            fp.concentration_score for fp in fingerprints
        ) / len(fingerprints)
        
        avg_turnover = sum(
            fp.turnover_rate for fp in fingerprints
        ) / len(fingerprints)
        
        avg_conviction = sum(
            fp.conviction_score for fp in fingerprints
        ) / len(fingerprints)
        
        if avg_concentration > 0.6:
            insight = self._create_insight(
                category="institutional_pattern",
                insight_type="concentration_preference",
                description=(
                    f"Top institutions run concentrated portfolios "
                    f"(avg {avg_concentration:.0%} concentration)"
                ),
                improvement_vector={
                    "suggested_concentration": avg_concentration,
                    "action": "consider_more_concentrated_positions",
                },
                sample_size=len(fingerprints),
            )
            insights.append(insight)
        
        if avg_turnover < 0.3:
            insight = self._create_insight(
                category="institutional_pattern",
                insight_type="low_turnover",
                description=(
                    f"Top institutions have low turnover "
                    f"({avg_turnover:.0%} quarterly) - patience is rewarded"
                ),
                improvement_vector={
                    "suggested_holding_period": "longer",
                    "action": "reduce_trading_frequency",
                },
                sample_size=len(fingerprints),
            )
            insights.append(insight)
        
        if avg_conviction > 0.7:
            insight = self._create_insight(
                category="institutional_pattern",
                insight_type="high_conviction",
                description=(
                    f"Top institutions show high conviction "
                    f"(top 5 positions = {avg_conviction:.0%} of portfolio)"
                ),
                improvement_vector={
                    "suggested_conviction": avg_conviction,
                    "action": "increase_conviction_in_best_ideas",
                },
                sample_size=len(fingerprints),
            )
            insights.append(insight)
        
        sector_counts: Dict[str, int] = defaultdict(int)
        for fp in fingerprints:
            for sector in fp.top_sectors:
                sector_counts[sector] += 1
        
        if sector_counts:
            top_sectors = sorted(
                sector_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            insight = self._create_insight(
                category="institutional_pattern",
                insight_type="sector_preferences",
                description=(
                    f"Top institutions favor: "
                    f"{', '.join(s[0] for s in top_sectors)}"
                ),
                improvement_vector={
                    "preferred_sectors": [s[0] for s in top_sectors],
                    "action": "align_sector_exposure",
                },
                sample_size=len(fingerprints),
            )
            insights.append(insight)
        
        return insights
    
    def generate_improvement_recommendations(self) -> Dict[str, Any]:
        """
        Generate recommendations for improving our signals.
        
        Based on all accumulated learning from battles and fingerprints.
        """
        if not self._insights and not self._battle_history:
            return {
                "status": "insufficient_data",
                "message": "Need more battle data to generate recommendations",
                "recommendations": [],
            }
        
        recommendations = []
        
        loss_insights = [i for i in self._insights if i.category == "loss_pattern"]
        if loss_insights:
            for insight in loss_insights:
                recommendations.append({
                    "type": "fix_weakness",
                    "priority": insight.improvement_vector.get("priority", "medium"),
                    "description": insight.description,
                    "action": insight.improvement_vector.get("action", "review"),
                    "confidence": insight.confidence,
                })
        
        win_insights = [i for i in self._insights if i.category == "win_pattern"]
        if win_insights:
            for insight in win_insights:
                recommendations.append({
                    "type": "reinforce_strength",
                    "priority": "medium",
                    "description": insight.description,
                    "action": insight.improvement_vector.get("action", "maintain"),
                    "confidence": insight.confidence,
                })
        
        pattern_insights = [
            i for i in self._insights if i.category == "institutional_pattern"
        ]
        if pattern_insights:
            for insight in pattern_insights:
                recommendations.append({
                    "type": "adopt_institutional_practice",
                    "priority": "low",
                    "description": insight.description,
                    "action": insight.improvement_vector.get("action", "consider"),
                    "confidence": insight.confidence,
                })
        
        recommendations = sorted(
            recommendations,
            key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]],
            reverse=True,
        )
        
        return {
            "status": "recommendations_generated",
            "total_battles_analyzed": len(self._battle_history),
            "total_insights_generated": len(self._insights),
            "recommendations": recommendations,
            "metrics": dict(self._learning_metrics),
            "compliance_status": ComplianceStatus.RESEARCH_ONLY.value,
        }
    
    def export_for_apexlab(self) -> Dict[str, Any]:
        """
        Export learning data for ApexLab training integration.
        
        Formats insights for use in the self-learning pipeline.
        """
        improvement_vectors = []
        
        for insight in self._insights:
            if insight.improvement_vector:
                improvement_vectors.append({
                    "insight_id": insight.insight_id,
                    "category": insight.category,
                    "type": insight.insight_type,
                    "vector": insight.improvement_vector,
                    "confidence": insight.confidence,
                    "sample_size": insight.sample_size,
                })
        
        return {
            "export_timestamp": datetime.utcnow().isoformat(),
            "source": "battle_simulator_adversarial_learning",
            "total_battles": len(self._battle_history),
            "win_rate": (
                self._learning_metrics["wins"] /
                (self._learning_metrics["wins"] + self._learning_metrics["losses"])
                if self._learning_metrics["wins"] + self._learning_metrics["losses"] > 0
                else 0
            ),
            "total_alpha": (
                self._learning_metrics["total_positive_alpha"] +
                self._learning_metrics["total_negative_alpha"]
            ),
            "improvement_vectors": improvement_vectors,
            "recommendations": self.generate_improvement_recommendations()["recommendations"],
            "compliance_status": ComplianceStatus.RESEARCH_ONLY.value,
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress."""
        total = self._learning_metrics["wins"] + self._learning_metrics["losses"]
        
        return {
            "total_battles_analyzed": len(self._battle_history),
            "total_insights_generated": len(self._insights),
            "wins": self._learning_metrics["wins"],
            "losses": self._learning_metrics["losses"],
            "win_rate": self._learning_metrics["wins"] / total if total > 0 else 0,
            "total_alpha": (
                self._learning_metrics["total_positive_alpha"] +
                self._learning_metrics["total_negative_alpha"]
            ),
            "insight_categories": {
                "loss_patterns": len([i for i in self._insights if i.category == "loss_pattern"]),
                "win_patterns": len([i for i in self._insights if i.category == "win_pattern"]),
                "institutional_patterns": len([i for i in self._insights if i.category == "institutional_pattern"]),
            },
            "compliance_status": ComplianceStatus.RESEARCH_ONLY.value,
        }
