"""
HyperLearner Pattern Miner.

Discovers winning and losing patterns from historical event-outcome pairs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib
import logging
import math

from ..models import (
    Pattern,
    EventOutcomePair,
    OutcomeType,
    LearningPriority,
    EventCategory,
    EventType,
)


logger = logging.getLogger(__name__)


class PatternMiner:
    """
    Discovers patterns in event-outcome data.
    
    Pattern Types:
    - Win patterns (what leads to success)
    - Loss patterns (what to avoid)
    - Regime patterns (market condition signatures)
    - Timing patterns (when things work)
    - Protocol patterns (which protocols predict outcomes)
    """
    
    def __init__(self, min_occurrences: int = 5, min_confidence: float = 0.6):
        self._min_occurrences = min_occurrences
        self._min_confidence = min_confidence
        
        self._patterns: Dict[str, Pattern] = {}
        self._pattern_candidates: Dict[str, List[EventOutcomePair]] = defaultdict(list)
        
        self._feature_buckets = {
            "quantrascore": [(0, 50), (50, 65), (65, 75), (75, 85), (85, 95), (95, 100)],
            "entropy": [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)],
            "volatility": [(0, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.1), (0.1, 1.0)],
            "confidence": [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)],
        }
        
    def ingest_pair(self, pair: EventOutcomePair):
        """Ingest a learning pair for pattern mining."""
        signatures = self._generate_signatures(pair)
        
        for sig in signatures:
            self._pattern_candidates[sig].append(pair)
            
            if len(self._pattern_candidates[sig]) >= self._min_occurrences:
                self._evaluate_pattern(sig)
                
    def ingest_batch(self, pairs: List[EventOutcomePair]):
        """Ingest a batch of learning pairs."""
        for pair in pairs:
            self.ingest_pair(pair)
            
    def _generate_signatures(self, pair: EventOutcomePair) -> List[str]:
        """Generate pattern signatures for a pair."""
        signatures = []
        event = pair.event
        features = pair.features_extracted
        
        base_sig = f"{event.category.value}:{event.event_type.value}"
        signatures.append(base_sig)
        
        for feature, buckets in self._feature_buckets.items():
            if feature in features:
                value = features[feature]
                for low, high in buckets:
                    if low <= value < high:
                        signatures.append(f"{base_sig}|{feature}:{low}-{high}")
                        break
                        
        if event.category == EventCategory.PROTOCOL:
            protocol = event.context.get("protocol_id", "unknown")
            signatures.append(f"protocol:{protocol}")
            
        if event.category == EventCategory.OMEGA:
            omega_type = event.context.get("omega_type", "unknown")
            signatures.append(f"omega:{omega_type}")
            
        if "regime" in event.context:
            regime = event.context["regime"]
            signatures.append(f"{base_sig}|regime:{regime}")
            
        if "quantrascore" in features and "volatility" in features:
            qs_bucket = self._get_bucket("quantrascore", features["quantrascore"])
            vol_bucket = self._get_bucket("volatility", features["volatility"])
            signatures.append(f"combo:qs{qs_bucket}+vol{vol_bucket}")
            
        return signatures
        
    def _get_bucket(self, feature: str, value: float) -> str:
        """Get bucket label for a feature value."""
        buckets = self._feature_buckets.get(feature, [])
        for low, high in buckets:
            if low <= value < high:
                return f"{low}-{high}"
        return "unk"
        
    def _evaluate_pattern(self, signature: str):
        """Evaluate if candidates form a valid pattern."""
        candidates = self._pattern_candidates[signature]
        
        if len(candidates) < self._min_occurrences:
            return None
            
        wins = sum(1 for p in candidates if p.outcome.outcome_type == OutcomeType.WIN)
        losses = sum(1 for p in candidates if p.outcome.outcome_type == OutcomeType.LOSS)
        total = len(candidates)
        
        win_rate = wins / total if total > 0 else 0.5
        
        returns = [p.outcome.return_pct for p in candidates if p.outcome.return_pct is not None]
        avg_return = sum(returns) / len(returns) if returns else 0.0
        
        confidence = self._calculate_confidence(candidates, win_rate)
        
        if confidence < self._min_confidence:
            return None
            
        if win_rate >= 0.55 or win_rate <= 0.40:
            pattern_id = hashlib.sha256(signature.encode()).hexdigest()[:12]
            
            if pattern_id in self._patterns:
                pattern = self._patterns[pattern_id]
                pattern.occurrence_count = total
                pattern.win_rate = win_rate
                pattern.avg_return = avg_return
                pattern.confidence = confidence
                pattern.last_seen = datetime.utcnow()
                pattern.examples = [p.event.event_id for p in candidates[-5:]]
            else:
                pattern = Pattern(
                    pattern_id=pattern_id,
                    pattern_type="WIN" if win_rate >= 0.55 else "LOSS",
                    description=self._generate_description(signature, win_rate, avg_return),
                    feature_signature=self._extract_signature_features(candidates),
                    occurrence_count=total,
                    win_rate=win_rate,
                    avg_return=avg_return,
                    confidence=confidence,
                    discovered_at=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    examples=[p.event.event_id for p in candidates[-5:]],
                )
                self._patterns[pattern_id] = pattern
                logger.info(f"[PatternMiner] Discovered pattern: {pattern.description} (win_rate={win_rate:.1%})")
                
            return pattern
            
        return None
        
    def _calculate_confidence(self, candidates: List[EventOutcomePair], win_rate: float) -> float:
        """Calculate statistical confidence in the pattern."""
        n = len(candidates)
        
        if n < 5:
            return 0.0
            
        p = win_rate
        se = math.sqrt(p * (1 - p) / n)
        
        z_score = abs(p - 0.5) / max(se, 0.001)
        
        confidence = min(0.5 + (z_score / 10), 0.99)
        
        if n >= 50:
            confidence = min(confidence + 0.1, 0.99)
        elif n >= 20:
            confidence = min(confidence + 0.05, 0.99)
            
        return confidence
        
    def _generate_description(self, signature: str, win_rate: float, avg_return: float) -> str:
        """Generate human-readable pattern description."""
        parts = signature.split("|")
        base = parts[0]
        
        outcome = "WINNING" if win_rate >= 0.55 else "LOSING"
        desc = f"{outcome} pattern in {base}"
        
        for part in parts[1:]:
            if ":" in part:
                key, val = part.split(":", 1)
                desc += f" when {key}={val}"
                
        desc += f" ({win_rate:.0%} win rate, {avg_return:+.1f}% avg return)"
        
        return desc
        
    def _extract_signature_features(self, candidates: List[EventOutcomePair]) -> Dict[str, Any]:
        """Extract common features from candidate pairs."""
        feature_sums = defaultdict(float)
        feature_counts = defaultdict(int)
        
        for pair in candidates:
            for key, value in pair.features_extracted.items():
                if isinstance(value, (int, float)):
                    feature_sums[key] += value
                    feature_counts[key] += 1
                    
        return {
            key: round(feature_sums[key] / feature_counts[key], 4)
            for key in feature_sums
        }
        
    def get_win_patterns(self, min_confidence: float = 0.7, limit: int = 20) -> List[Pattern]:
        """Get top winning patterns."""
        patterns = [
            p for p in self._patterns.values()
            if p.pattern_type == "WIN" and p.confidence >= min_confidence
        ]
        patterns.sort(key=lambda p: (p.win_rate * p.confidence, p.avg_return), reverse=True)
        return patterns[:limit]
        
    def get_loss_patterns(self, min_confidence: float = 0.7, limit: int = 20) -> List[Pattern]:
        """Get patterns to avoid."""
        patterns = [
            p for p in self._patterns.values()
            if p.pattern_type == "LOSS" and p.confidence >= min_confidence
        ]
        patterns.sort(key=lambda p: (1 - p.win_rate) * p.confidence, reverse=True)
        return patterns[:limit]
        
    def get_all_patterns(self) -> List[Pattern]:
        """Get all discovered patterns."""
        return list(self._patterns.values())
        
    def match_pattern(self, pair: EventOutcomePair) -> List[Pattern]:
        """Find patterns that match a new pair."""
        signatures = self._generate_signatures(pair)
        matched = []
        
        for sig in signatures:
            pattern_id = hashlib.sha256(sig.encode()).hexdigest()[:12]
            if pattern_id in self._patterns:
                matched.append(self._patterns[pattern_id])
                
        return matched
        
    def get_stats(self) -> Dict[str, Any]:
        """Get pattern mining statistics."""
        win_patterns = len([p for p in self._patterns.values() if p.pattern_type == "WIN"])
        loss_patterns = len([p for p in self._patterns.values() if p.pattern_type == "LOSS"])
        
        total_candidates = sum(len(c) for c in self._pattern_candidates.values())
        
        return {
            "total_patterns": len(self._patterns),
            "win_patterns": win_patterns,
            "loss_patterns": loss_patterns,
            "candidate_signatures": len(self._pattern_candidates),
            "total_candidate_pairs": total_candidates,
        }
