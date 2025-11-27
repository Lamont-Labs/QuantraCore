"""
Main Apex Engine module for QuantraCore Apex.

This is the primary entry point for deterministic analysis.
"""

import hashlib
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from .schemas import (
    OhlcvWindow, ApexContext, ApexResult, ProtocolResult,
    RegimeType, RiskTier, ScoreBucket
)
from .microtraits import compute_microtraits
from .entropy import compute_entropy
from .suppression import compute_suppression
from .drift import compute_drift
from .continuation import compute_continuation
from .volume_spike import compute_volume_metrics
from .regime import classify_regime
from .quantrascore import compute_quantrascore
from .verdict import build_verdict
from .sector_context import SectorContext
from .proof_logger import proof_logger


class ApexEngine:
    """
    The main Apex analysis engine.
    
    Provides deterministic, reproducible analysis of OHLCV data.
    """
    
    def __init__(self, enable_logging: bool = True, auto_load_protocols: bool = True):
        self.enable_logging = enable_logging
        self._protocol_runner = None
        
        if auto_load_protocols:
            self._init_protocol_runner()
    
    def _init_protocol_runner(self) -> None:
        """Initialize the tier protocol runner."""
        try:
            from src.quantracore_apex.protocols.tier.tier_loader import TierProtocolRunner
            self._protocol_runner = TierProtocolRunner()
        except ImportError:
            pass
    
    def set_protocol_runner(self, runner) -> None:
        """Set the protocol runner for tier protocol execution."""
        self._protocol_runner = runner
    
    def run(
        self,
        window: OhlcvWindow,
        context: Optional[ApexContext] = None
    ) -> ApexResult:
        """
        Run the Apex engine on an OHLCV window.
        
        Args:
            window: 100-bar OHLCV window
            context: Optional execution context
            
        Returns:
            ApexResult with complete analysis
        """
        if context is None:
            context = ApexContext()
        
        sector_ctx = SectorContext(context.sector)
        
        microtraits = compute_microtraits(window)
        
        entropy_metrics = compute_entropy(window)
        suppression_metrics = compute_suppression(window)
        drift_metrics = compute_drift(window)
        continuation_metrics = compute_continuation(window)
        volume_metrics = compute_volume_metrics(window)
        
        regime = classify_regime(window, microtraits)
        
        quantrascore, score_bucket = compute_quantrascore(
            microtraits=microtraits,
            entropy_metrics=entropy_metrics,
            suppression_metrics=suppression_metrics,
            drift_metrics=drift_metrics,
            continuation_metrics=continuation_metrics,
            volume_metrics=volume_metrics,
            regime=regime,
        )
        
        quantrascore = sector_ctx.adjust_score_for_sector(quantrascore)
        
        verdict, risk_tier = build_verdict(
            quantrascore=quantrascore,
            score_bucket=score_bucket,
            regime=regime,
            entropy_state=entropy_metrics.entropy_state,
            suppression_state=suppression_metrics.suppression_state,
            drift_state=drift_metrics.drift_state,
        )
        
        protocol_results = []
        if self._protocol_runner:
            protocol_results = self._protocol_runner.run_all(window, microtraits)
        
        omega_overrides = self._apply_omega_directives(
            quantrascore=quantrascore,
            entropy_metrics=entropy_metrics,
            drift_metrics=drift_metrics,
            context=context,
        )
        
        result = ApexResult(
            symbol=window.symbol,
            timestamp=datetime.utcnow(),
            window_hash=window.get_hash(),
            quantrascore=quantrascore,
            score_bucket=score_bucket,
            regime=regime,
            risk_tier=risk_tier,
            entropy_state=entropy_metrics.entropy_state,
            suppression_state=suppression_metrics.suppression_state,
            drift_state=drift_metrics.drift_state,
            microtraits=microtraits,
            entropy_metrics=entropy_metrics,
            suppression_metrics=suppression_metrics,
            drift_metrics=drift_metrics,
            continuation_metrics=continuation_metrics,
            volume_metrics=volume_metrics,
            protocol_results=protocol_results,
            verdict=verdict,
            omega_overrides=omega_overrides,
        )
        
        if self.enable_logging:
            proof_logger.log_execution(result, context.model_dump() if context else None)
        
        return result
    
    def _apply_omega_directives(
        self,
        quantrascore: float,
        entropy_metrics,
        drift_metrics,
        context: ApexContext,
    ) -> Dict[str, bool]:
        """
        Apply Omega directives as final safety checks.
        """
        overrides = {
            "omega_1_safety": False,
            "omega_2_entropy": False,
            "omega_3_drift": False,
            "omega_4_compliance": False,
        }
        
        from .schemas import EntropyState, DriftState
        
        if entropy_metrics.entropy_state == EntropyState.CHAOTIC:
            overrides["omega_2_entropy"] = True
        
        if drift_metrics.drift_state == DriftState.CRITICAL:
            overrides["omega_3_drift"] = True
        
        if context.compliance_mode:
            overrides["omega_4_compliance"] = True
        
        return overrides


def run_apex(
    window: OhlcvWindow,
    context: Optional[ApexContext] = None
) -> ApexResult:
    """
    Convenience function to run Apex engine.
    
    Args:
        window: 100-bar OHLCV window
        context: Optional execution context
        
    Returns:
        ApexResult with complete analysis
    """
    engine = ApexEngine()
    return engine.run(window, context)
