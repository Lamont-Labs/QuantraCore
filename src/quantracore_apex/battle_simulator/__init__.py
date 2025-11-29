"""
QuantraCore Apex - Competitive Intelligence Battle Simulator.

A fully compliant institutional intelligence system that:
1. Analyzes publicly disclosed institutional holdings (SEC 13F)
2. Fingerprints institutional trading strategies
3. Simulates battles against top institutions
4. Learns from institutional patterns to improve
5. Provides acquirer-compatible abstraction layers

DATA SOURCES (100% Legal & Compliant):
- SEC EDGAR 13F Filings (quarterly institutional holdings)
- SEC Form 4 (insider transactions)
- Public company filings (10-K, 10-Q, 8-K)
- Publicly available hedge fund letters

COMPLIANCE NOTES:
- All data sourced from public SEC filings
- No non-public information used
- Research and educational purposes only
- Not intended as investment advice
"""

from .models import (
    Institution,
    InstitutionalHolding,
    Filing13F,
    StrategyFingerprint,
    BattleResult,
    AdversarialInsight,
    AdaptationProfile,
    ComplianceStatus,
)
from .data_sources.sec_edgar import SECEdgarClient
from .fingerprinting.strategy_analyzer import StrategyAnalyzer
from .simulation.battle_engine import BattleEngine
from .learning.adversarial_learner import AdversarialLearner
from .adaptation.acquirer_adapter import AcquirerAdapter

__all__ = [
    "Institution",
    "InstitutionalHolding",
    "Filing13F",
    "StrategyFingerprint",
    "BattleResult",
    "AdversarialInsight",
    "AdaptationProfile",
    "ComplianceStatus",
    "SECEdgarClient",
    "StrategyAnalyzer",
    "BattleEngine",
    "AdversarialLearner",
    "AcquirerAdapter",
]
