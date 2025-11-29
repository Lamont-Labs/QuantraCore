"""
HyperLearner Meta-Learning System.

Learns how to learn better - optimizing the learning process itself.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import logging
import math

from ..models import (
    MetaLearningInsight,
    LearningMetrics,
    Pattern,
    LearningPriority,
    EventCategory,
)


logger = logging.getLogger(__name__)


class MetaLearner:
    """
    Meta-learning system that optimizes the learning process.
    
    Analyzes:
    - Which types of events provide the most learning value
    - Optimal batch sizes and intervals
    - Feature importance evolution
    - Learning velocity optimization
    - Diminishing returns detection
    """
    
    def __init__(self):
        self._insights: List[MetaLearningInsight] = []
        self._metrics_history: List[Tuple[datetime, LearningMetrics]] = []
        self._category_value: Dict[EventCategory, float] = defaultdict(float)
        self._priority_effectiveness: Dict[LearningPriority, float] = defaultdict(float)
        self._feature_importance: Dict[str, float] = defaultdict(float)
        
        self._learning_velocity: List[float] = []
        self._batch_size_experiments: List[Tuple[int, float]] = []
        self._interval_experiments: List[Tuple[int, float]] = []
        
        self._optimal_batch_size = 100
        self._optimal_interval_seconds = 60
        self._optimal_priority_weights = {
            LearningPriority.CRITICAL: 5.0,
            LearningPriority.HIGH: 3.0,
            LearningPriority.MEDIUM: 2.0,
            LearningPriority.LOW: 1.0,
            LearningPriority.BACKGROUND: 0.5,
        }
        
    def record_metrics(self, metrics: LearningMetrics):
        """Record metrics for meta-analysis."""
        self._metrics_history.append((datetime.utcnow(), metrics))
        
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-1000:]
            
        if len(self._metrics_history) >= 10:
            self._analyze_learning_velocity()
            
    def record_category_outcome(
        self,
        category: EventCategory,
        was_valuable: bool,
        learning_contribution: float,
    ):
        """Record how valuable a category was for learning."""
        alpha = 0.1
        current = self._category_value.get(category, 0.5)
        
        if was_valuable:
            self._category_value[category] = current + alpha * (learning_contribution - current)
        else:
            self._category_value[category] = current * (1 - alpha)
            
    def record_priority_outcome(
        self,
        priority: LearningPriority,
        actual_value: float,
    ):
        """Record actual learning value for a priority level."""
        alpha = 0.1
        current = self._priority_effectiveness.get(priority, 1.0)
        self._priority_effectiveness[priority] = current + alpha * (actual_value - current)
        
    def record_feature_importance(self, feature_name: str, importance: float):
        """Record feature importance from model training."""
        alpha = 0.2
        current = self._feature_importance.get(feature_name, 0.5)
        self._feature_importance[feature_name] = current + alpha * (importance - current)
        
    def _analyze_learning_velocity(self):
        """Analyze how fast we're learning."""
        if len(self._metrics_history) < 10:
            return
            
        recent = self._metrics_history[-20:]
        
        if len(recent) < 2:
            return
            
        time_span = (recent[-1][0] - recent[0][0]).total_seconds() / 3600
        
        if time_span <= 0:
            return
            
        samples_learned = recent[-1][1].total_events_captured - recent[0][1].total_events_captured
        velocity = samples_learned / time_span
        
        self._learning_velocity.append(velocity)
        
        if len(self._learning_velocity) > 100:
            self._learning_velocity = self._learning_velocity[-100:]
            
        if len(self._learning_velocity) >= 5:
            recent_avg = sum(self._learning_velocity[-5:]) / 5
            earlier_avg = sum(self._learning_velocity[:5]) / 5 if len(self._learning_velocity) >= 10 else recent_avg
            
            if recent_avg < earlier_avg * 0.8:
                self._generate_insight(
                    "velocity_decline",
                    "Learning velocity is declining",
                    "Consider increasing data diversity or adjusting priority weights",
                    expected_improvement=0.2,
                    confidence=0.7,
                )
                
    def generate_optimization_insights(self) -> List[MetaLearningInsight]:
        """Generate insights for optimizing the learning process."""
        insights = []
        
        if self._category_value:
            max_value = max(self._category_value.values())
            min_value = min(self._category_value.values())
            
            if max_value > min_value * 2:
                top_categories = sorted(
                    self._category_value.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                insight = self._generate_insight(
                    "category_focus",
                    f"High-value categories identified: {[c[0].value for c in top_categories]}",
                    f"Increase sampling from {top_categories[0][0].value} category for faster learning",
                    expected_improvement=0.15,
                    confidence=0.75,
                )
                insights.append(insight)
                
        if self._feature_importance:
            sorted_features = sorted(
                self._feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_features = [f[0] for f in sorted_features[:5]]
            low_features = [f[0] for f in sorted_features[-3:] if f[1] < 0.1]
            
            if low_features:
                insight = self._generate_insight(
                    "feature_pruning",
                    f"Low-value features detected: {low_features}",
                    "Consider removing these features to reduce noise",
                    expected_improvement=0.05,
                    confidence=0.6,
                )
                insights.append(insight)
                
        if self._learning_velocity and len(self._learning_velocity) >= 10:
            velocities = self._learning_velocity[-10:]
            variance = sum((v - sum(velocities)/len(velocities))**2 for v in velocities) / len(velocities)
            
            if variance > 100:
                insight = self._generate_insight(
                    "velocity_stabilization",
                    "Learning velocity is highly variable",
                    "Implement more consistent batch scheduling",
                    expected_improvement=0.1,
                    confidence=0.65,
                )
                insights.append(insight)
                
        return insights
        
    def _generate_insight(
        self,
        insight_type: str,
        description: str,
        recommendation: str,
        expected_improvement: float,
        confidence: float,
    ) -> MetaLearningInsight:
        """Generate and store a meta-learning insight."""
        import hashlib
        
        insight_id = hashlib.sha256(
            f"{insight_type}-{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        insight = MetaLearningInsight(
            insight_id=insight_id,
            insight_type=insight_type,
            description=description,
            recommendation=recommendation,
            expected_improvement=expected_improvement,
            confidence=confidence,
            discovered_at=datetime.utcnow(),
            applied=False,
        )
        
        self._insights.append(insight)
        logger.info(f"[MetaLearner] New insight: {description}")
        
        return insight
        
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Get optimized learning parameters."""
        category_weights = {
            cat.value: round(value, 3)
            for cat, value in self._category_value.items()
        }
        
        priority_weights = {
            priority.name: round(eff, 3)
            for priority, eff in self._priority_effectiveness.items()
        }
        
        feature_ranking = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        avg_velocity = sum(self._learning_velocity[-10:]) / max(len(self._learning_velocity[-10:]), 1)
        
        return {
            "optimal_batch_size": self._optimal_batch_size,
            "optimal_interval_seconds": self._optimal_interval_seconds,
            "category_weights": category_weights,
            "priority_weights": priority_weights,
            "top_features": feature_ranking[:10],
            "current_velocity": round(avg_velocity, 2),
            "velocity_trend": "increasing" if len(self._learning_velocity) >= 2 and self._learning_velocity[-1] > self._learning_velocity[-2] else "stable",
        }
        
    def get_insights(self, limit: int = 20) -> List[MetaLearningInsight]:
        """Get recent meta-learning insights."""
        return sorted(
            self._insights,
            key=lambda i: i.discovered_at,
            reverse=True
        )[:limit]
        
    def mark_insight_applied(self, insight_id: str) -> bool:
        """Mark an insight as applied."""
        for insight in self._insights:
            if insight.insight_id == insight_id:
                insight.applied = True
                logger.info(f"[MetaLearner] Insight applied: {insight_id}")
                return True
        return False
        
    def get_learning_health(self) -> Dict[str, Any]:
        """Get overall learning health metrics."""
        if not self._learning_velocity:
            return {
                "status": "initializing",
                "velocity": 0,
                "trend": "unknown",
                "health_score": 0.5,
            }
            
        avg_velocity = sum(self._learning_velocity[-10:]) / len(self._learning_velocity[-10:])
        
        if len(self._learning_velocity) >= 5:
            recent = sum(self._learning_velocity[-3:]) / 3
            earlier = sum(self._learning_velocity[-6:-3]) / 3 if len(self._learning_velocity) >= 6 else recent
            
            if recent > earlier * 1.1:
                trend = "accelerating"
                trend_score = 0.2
            elif recent < earlier * 0.9:
                trend = "decelerating"
                trend_score = -0.2
            else:
                trend = "stable"
                trend_score = 0.1
        else:
            trend = "initializing"
            trend_score = 0
            
        active_categories = len([v for v in self._category_value.values() if v > 0.3])
        category_score = min(active_categories / 5, 1.0) * 0.2
        
        pattern_count = sum(1 for i in self._insights if not i.applied)
        insight_score = min(pattern_count / 10, 1.0) * 0.1
        
        health_score = 0.5 + trend_score + category_score + insight_score
        health_score = max(0.0, min(1.0, health_score))
        
        if health_score >= 0.8:
            status = "excellent"
        elif health_score >= 0.6:
            status = "good"
        elif health_score >= 0.4:
            status = "fair"
        else:
            status = "needs_attention"
            
        return {
            "status": status,
            "health_score": round(health_score, 3),
            "velocity": round(avg_velocity, 2),
            "trend": trend,
            "active_categories": active_categories,
            "pending_insights": pattern_count,
        }
