"""
Protocol Telemetry System - Track Protocol Contribution to Trade Outcomes.

Measures which of the 145+ protocols actually contribute to profitable trades vs noise.
Provides:
- Per-protocol lift metrics (how much does firing improve outcomes)
- Redundancy scoring (which protocols fire together)
- Win rate by protocol combination
- Protocol attribution for each trade outcome

This enables data-driven protocol pruning and weighting.
"""

import os
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProtocolMetrics:
    """Metrics for a single protocol."""
    protocol_id: str
    total_fires: int = 0
    win_count: int = 0
    loss_count: int = 0
    avg_pnl_when_fired: float = 0.0
    avg_pnl_when_not_fired: float = 0.0
    lift: float = 0.0
    confidence: float = 0.0
    last_updated: str = ""
    
    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["win_rate"] = self.win_rate
        return d


@dataclass
class ProtocolCombination:
    """Metrics for a combination of protocols."""
    protocols: Tuple[str, ...]
    total_occurrences: int = 0
    win_count: int = 0
    avg_pnl: float = 0.0
    
    @property
    def win_rate(self) -> float:
        return self.win_count / self.total_occurrences if self.total_occurrences > 0 else 0.5


@dataclass
class TelemetrySnapshot:
    """Point-in-time snapshot of telemetry data."""
    timestamp: str
    total_trades: int
    protocols: Dict[str, ProtocolMetrics]
    top_combinations: List[Dict[str, Any]]
    version: str = "1.0.0"


class ProtocolTelemetry:
    """
    Tracks protocol contribution to trade outcomes.
    
    Usage:
        telemetry = ProtocolTelemetry()
        
        # Record trade with protocols that fired
        telemetry.record_trade(
            protocols_fired=["T03", "T17", "MR05"],
            pnl_percent=2.5,
            is_win=True
        )
        
        # Get protocol effectiveness
        metrics = telemetry.get_protocol_metrics("T03")
        print(f"T03 lift: {metrics.lift:.2f}")
    """
    
    def __init__(
        self,
        data_dir: str = "data/telemetry",
        min_samples_for_confidence: int = 30,
    ):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        self._min_samples = min_samples_for_confidence
        
        self._protocol_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "fires": 0,
            "wins": 0,
            "losses": 0,
            "pnl_sum": 0.0,
            "pnl_squared_sum": 0.0,
        })
        
        self._combination_stats: Dict[Tuple[str, ...], Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "wins": 0,
            "pnl_sum": 0.0,
        })
        
        self._baseline_stats = {
            "total_trades": 0,
            "wins": 0,
            "pnl_sum": 0.0,
        }
        
        self._redundancy_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        self._lock = threading.Lock()
        
        self._load_state()
    
    def record_trade(
        self,
        protocols_fired: List[str],
        pnl_percent: float,
        is_win: bool,
        all_protocols: Optional[List[str]] = None,
    ) -> None:
        """
        Record a completed trade with its protocol context.
        
        Args:
            protocols_fired: List of protocol IDs that fired for this trade
            pnl_percent: P&L percentage of the trade
            is_win: Whether the trade was profitable
            all_protocols: Optional list of all available protocols (for non-fire tracking)
        """
        with self._lock:
            self._baseline_stats["total_trades"] += 1
            self._baseline_stats["pnl_sum"] += pnl_percent
            if is_win:
                self._baseline_stats["wins"] += 1
            
            for protocol_id in protocols_fired:
                stats = self._protocol_stats[protocol_id]
                stats["fires"] += 1
                stats["pnl_sum"] += pnl_percent
                stats["pnl_squared_sum"] += pnl_percent ** 2
                if is_win:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
            
            for i, p1 in enumerate(protocols_fired):
                for p2 in protocols_fired[i+1:]:
                    key = tuple(sorted([p1, p2]))
                    self._redundancy_matrix[key[0]][key[1]] += 1
            
            if len(protocols_fired) >= 2:
                combo_key = tuple(sorted(protocols_fired[:5]))
                combo_stats = self._combination_stats[combo_key]
                combo_stats["count"] += 1
                combo_stats["pnl_sum"] += pnl_percent
                if is_win:
                    combo_stats["wins"] += 1
            
            if self._baseline_stats["total_trades"] % 100 == 0:
                self._save_state()
    
    def get_protocol_metrics(self, protocol_id: str) -> ProtocolMetrics:
        """Get metrics for a specific protocol."""
        with self._lock:
            stats = self._protocol_stats.get(protocol_id, {})
            
            fires = stats.get("fires", 0)
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            pnl_sum = stats.get("pnl_sum", 0.0)
            
            avg_pnl_fired = pnl_sum / fires if fires > 0 else 0.0
            
            total_trades = self._baseline_stats["total_trades"]
            non_fire_trades = total_trades - fires
            non_fire_pnl = self._baseline_stats["pnl_sum"] - pnl_sum
            avg_pnl_not_fired = non_fire_pnl / non_fire_trades if non_fire_trades > 0 else 0.0
            
            lift = avg_pnl_fired - avg_pnl_not_fired
            
            confidence = min(1.0, fires / self._min_samples) if fires > 0 else 0.0
            
            return ProtocolMetrics(
                protocol_id=protocol_id,
                total_fires=fires,
                win_count=wins,
                loss_count=losses,
                avg_pnl_when_fired=avg_pnl_fired,
                avg_pnl_when_not_fired=avg_pnl_not_fired,
                lift=lift,
                confidence=confidence,
                last_updated=datetime.utcnow().isoformat(),
            )
    
    def get_all_metrics(self) -> Dict[str, ProtocolMetrics]:
        """Get metrics for all tracked protocols."""
        with self._lock:
            return {
                pid: self.get_protocol_metrics(pid)
                for pid in self._protocol_stats.keys()
            }
    
    def get_top_protocols(self, n: int = 20, min_fires: int = 10) -> List[ProtocolMetrics]:
        """Get top N protocols by lift."""
        all_metrics = self.get_all_metrics()
        
        filtered = [m for m in all_metrics.values() if m.total_fires >= min_fires]
        sorted_metrics = sorted(filtered, key=lambda x: x.lift, reverse=True)
        
        return sorted_metrics[:n]
    
    def get_redundant_protocols(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Find protocols that always fire together (redundant).
        
        Args:
            threshold: Co-occurrence threshold (0-1)
            
        Returns:
            List of (protocol1, protocol2, co_occurrence_rate) tuples
        """
        redundant = []
        
        with self._lock:
            for p1, inner in self._redundancy_matrix.items():
                p1_fires = self._protocol_stats[p1].get("fires", 0)
                if p1_fires < self._min_samples:
                    continue
                    
                for p2, co_fire_count in inner.items():
                    p2_fires = self._protocol_stats[p2].get("fires", 0)
                    if p2_fires < self._min_samples:
                        continue
                    
                    co_occurrence = co_fire_count / min(p1_fires, p2_fires)
                    if co_occurrence >= threshold:
                        redundant.append((p1, p2, co_occurrence))
        
        return sorted(redundant, key=lambda x: x[2], reverse=True)
    
    def get_best_combinations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N protocol combinations by win rate."""
        combos = []
        
        with self._lock:
            for combo_key, stats in self._combination_stats.items():
                if stats["count"] < 5:
                    continue
                
                win_rate = stats["wins"] / stats["count"]
                avg_pnl = stats["pnl_sum"] / stats["count"]
                
                combos.append({
                    "protocols": list(combo_key),
                    "count": stats["count"],
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                })
        
        sorted_combos = sorted(combos, key=lambda x: (x["win_rate"], x["avg_pnl"]), reverse=True)
        return sorted_combos[:n]
    
    def compute_protocol_weights(self) -> Dict[str, float]:
        """
        Compute optimal weights for each protocol based on historical lift.
        
        Returns:
            Dictionary of protocol_id -> weight (0-2, where 1 is neutral)
        """
        all_metrics = self.get_all_metrics()
        
        lifts = [m.lift for m in all_metrics.values() if m.total_fires >= self._min_samples]
        
        if not lifts:
            return {pid: 1.0 for pid in all_metrics.keys()}
        
        lift_std = np.std(lifts) if len(lifts) > 1 else 1.0
        lift_mean = np.mean(lifts)
        
        weights = {}
        for pid, metrics in all_metrics.items():
            if metrics.total_fires < self._min_samples // 2:
                weights[pid] = 1.0
            else:
                z_score = (metrics.lift - lift_mean) / lift_std if lift_std > 0 else 0
                weight = 1.0 + (z_score * 0.3)
                weight = max(0.2, min(2.0, weight))
                weights[pid] = weight * metrics.confidence + 1.0 * (1 - metrics.confidence)
        
        return weights
    
    def get_snapshot(self) -> TelemetrySnapshot:
        """Get complete telemetry snapshot."""
        return TelemetrySnapshot(
            timestamp=datetime.utcnow().isoformat(),
            total_trades=self._baseline_stats["total_trades"],
            protocols=self.get_all_metrics(),
            top_combinations=self.get_best_combinations(20),
        )
    
    def _save_state(self) -> None:
        """Persist telemetry state to disk."""
        state = {
            "baseline": self._baseline_stats,
            "protocols": dict(self._protocol_stats),
            "combinations": {
                str(k): v for k, v in self._combination_stats.items()
            },
            "saved_at": datetime.utcnow().isoformat(),
        }
        
        state_file = self._data_dir / "telemetry_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self) -> None:
        """Load telemetry state from disk."""
        state_file = self._data_dir / "telemetry_state.json"
        if not state_file.exists():
            return
        
        try:
            with open(state_file) as f:
                state = json.load(f)
            
            self._baseline_stats = state.get("baseline", self._baseline_stats)
            
            for pid, stats in state.get("protocols", {}).items():
                self._protocol_stats[pid] = stats
            
            logger.info(f"[Telemetry] Loaded state: {self._baseline_stats['total_trades']} trades")
        except Exception as e:
            logger.warning(f"[Telemetry] Could not load state: {e}")


_telemetry_instance: Optional[ProtocolTelemetry] = None


def get_protocol_telemetry() -> ProtocolTelemetry:
    """Get global telemetry instance."""
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = ProtocolTelemetry()
    return _telemetry_instance


__all__ = [
    "ProtocolTelemetry",
    "ProtocolMetrics",
    "TelemetrySnapshot",
    "get_protocol_telemetry",
]
