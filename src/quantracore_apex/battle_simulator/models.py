"""
Battle Simulator Data Models.

All models designed for compliance with SEC regulations and
institutional analysis best practices.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Set
from enum import Enum


class ComplianceStatus(str, Enum):
    """Compliance verification status."""
    VERIFIED = "verified"
    PENDING_REVIEW = "pending_review"
    PUBLIC_DATA = "public_data"
    RESEARCH_ONLY = "research_only"


class InstitutionType(str, Enum):
    """Types of institutional investors."""
    HEDGE_FUND = "hedge_fund"
    MUTUAL_FUND = "mutual_fund"
    PENSION_FUND = "pension_fund"
    INSURANCE_COMPANY = "insurance_company"
    INVESTMENT_ADVISOR = "investment_advisor"
    BANK = "bank"
    ENDOWMENT = "endowment"
    SOVEREIGN_WEALTH = "sovereign_wealth"
    FAMILY_OFFICE = "family_office"
    OTHER = "other"


class StrategyType(str, Enum):
    """Institutional strategy classifications."""
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"
    MARKET_NEUTRAL = "market_neutral"
    EVENT_DRIVEN = "event_driven"
    MACRO = "macro"
    QUANTITATIVE = "quantitative"
    VALUE = "value"
    GROWTH = "growth"
    MOMENTUM = "momentum"
    ACTIVIST = "activist"
    MULTI_STRATEGY = "multi_strategy"
    UNKNOWN = "unknown"


class BattleOutcome(str, Enum):
    """Outcome of battle simulation."""
    WIN = "win"
    LOSS = "loss"
    TIE = "tie"
    INCONCLUSIVE = "inconclusive"


@dataclass
class Institution:
    """
    Represents an institutional investor.
    
    Data sourced from SEC EDGAR 13F filings (public).
    """
    cik: str
    name: str
    institution_type: InstitutionType = InstitutionType.OTHER
    aum_estimate: Optional[float] = None
    filing_count: int = 0
    first_filing_date: Optional[date] = None
    latest_filing_date: Optional[date] = None
    headquarters: Optional[str] = None
    notable_holdings: List[str] = field(default_factory=list)
    
    compliance_status: ComplianceStatus = ComplianceStatus.PUBLIC_DATA
    data_source: str = "SEC EDGAR 13F"
    
    def __post_init__(self):
        self.cik = str(self.cik).zfill(10)


@dataclass
class InstitutionalHolding:
    """
    A single holding from a 13F filing.
    
    All data from public SEC filings.
    """
    cusip: str
    issuer_name: str
    class_title: str
    value: float
    shares: int
    share_type: str
    investment_discretion: str
    voting_authority_sole: int = 0
    voting_authority_shared: int = 0
    voting_authority_none: int = 0
    
    put_call: Optional[str] = None
    
    @property
    def avg_price(self) -> float:
        """Estimate average price from value and shares."""
        if self.shares > 0:
            return (self.value * 1000) / self.shares
        return 0.0


@dataclass
class Filing13F:
    """
    A complete 13F-HR filing from SEC EDGAR.
    
    13F filings are required quarterly disclosures of institutional
    holdings for managers with $100M+ in qualifying assets.
    """
    accession_number: str
    cik: str
    institution_name: str
    filing_date: date
    period_of_report: date
    
    holdings: List[InstitutionalHolding] = field(default_factory=list)
    
    total_value: float = 0.0
    total_holdings_count: int = 0
    
    amendment_type: Optional[str] = None
    is_amendment: bool = False
    
    compliance_status: ComplianceStatus = ComplianceStatus.PUBLIC_DATA
    data_source: str = "SEC EDGAR"
    
    def __post_init__(self):
        self.cik = str(self.cik).zfill(10)
        if not self.total_value and self.holdings:
            self.total_value = sum(h.value for h in self.holdings)
        if not self.total_holdings_count:
            self.total_holdings_count = len(self.holdings)


@dataclass
class PositionChange:
    """Represents a change in institutional position between filings."""
    symbol: str
    cusip: str
    issuer_name: str
    
    prior_shares: int
    current_shares: int
    shares_change: int
    shares_change_pct: float
    
    prior_value: float
    current_value: float
    value_change: float
    value_change_pct: float
    
    action: str
    
    @property
    def is_new_position(self) -> bool:
        return self.prior_shares == 0 and self.current_shares > 0
    
    @property
    def is_exit(self) -> bool:
        return self.prior_shares > 0 and self.current_shares == 0
    
    @property
    def is_increase(self) -> bool:
        return self.shares_change > 0
    
    @property
    def is_decrease(self) -> bool:
        return self.shares_change < 0


@dataclass
class StrategyFingerprint:
    """
    Fingerprint of an institution's trading strategy.
    
    Derived from analyzing patterns in public 13F filings.
    """
    institution_cik: str
    institution_name: str
    analysis_date: datetime
    
    primary_strategy: StrategyType = StrategyType.UNKNOWN
    secondary_strategies: List[StrategyType] = field(default_factory=list)
    
    concentration_score: float = 0.0
    turnover_rate: float = 0.0
    conviction_score: float = 0.0
    
    sector_allocations: Dict[str, float] = field(default_factory=dict)
    market_cap_preferences: Dict[str, float] = field(default_factory=dict)
    
    avg_holding_period_quarters: float = 0.0
    position_sizing_style: str = "unknown"
    
    top_sectors: List[str] = field(default_factory=list)
    top_holdings: List[str] = field(default_factory=list)
    recent_buys: List[str] = field(default_factory=list)
    recent_sells: List[str] = field(default_factory=list)
    
    confidence_score: float = 0.0
    
    quarters_analyzed: int = 0
    filings_analyzed: int = 0
    
    compliance_status: ComplianceStatus = ComplianceStatus.PUBLIC_DATA
    methodology: str = "Public 13F filing analysis"


@dataclass
class BattleScenario:
    """A specific battle scenario configuration."""
    scenario_id: str
    name: str
    description: str
    
    symbol: str
    start_date: date
    end_date: date
    
    our_signal_date: date
    our_signal_direction: str
    our_quantrascore: float
    our_entry_price: float
    our_exit_price: Optional[float] = None
    
    institution_cik: str = ""
    institution_name: str = ""
    institution_action: str = ""
    institution_timing: str = ""


@dataclass
class BattleResult:
    """
    Result of simulating our signals against an institution.
    
    Compares our hypothetical trades against institutional moves
    discovered in subsequent 13F filings.
    """
    battle_id: str
    scenario: BattleScenario
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    outcome: BattleOutcome = BattleOutcome.INCONCLUSIVE
    
    our_return_pct: float = 0.0
    institution_return_pct: float = 0.0
    alpha_generated: float = 0.0
    
    our_timing_score: float = 0.0
    institution_timing_score: float = 0.0
    timing_advantage: float = 0.0
    
    our_sizing_efficiency: float = 0.0
    institution_sizing_efficiency: float = 0.0
    
    lessons_learned: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    compliance_status: ComplianceStatus = ComplianceStatus.RESEARCH_ONLY
    methodology: str = "Backtested comparison using public data"


@dataclass
class AdversarialInsight:
    """
    Insight derived from adversarial learning against institutions.
    
    These are patterns we've learned from analyzing where institutions
    outperformed or underperformed our signals.
    """
    insight_id: str
    generated_at: datetime
    
    category: str
    insight_type: str
    description: str
    
    source_institutions: List[str] = field(default_factory=list)
    
    applicable_scenarios: List[str] = field(default_factory=list)
    applicable_regimes: List[str] = field(default_factory=list)
    applicable_sectors: List[str] = field(default_factory=list)
    
    confidence: float = 0.0
    sample_size: int = 0
    
    improvement_vector: Dict[str, float] = field(default_factory=dict)
    
    recommended_adjustments: Dict[str, Any] = field(default_factory=dict)
    
    compliance_status: ComplianceStatus = ComplianceStatus.RESEARCH_ONLY


@dataclass
class AdaptationProfile:
    """
    Profile for adapting QuantraCore to acquirer infrastructure.
    
    Provides abstraction layer for M&A compatibility.
    """
    profile_id: str
    profile_name: str
    created_at: datetime
    
    target_infrastructure: str
    
    data_feed_mappings: Dict[str, str] = field(default_factory=dict)
    risk_framework_overrides: Dict[str, Any] = field(default_factory=dict)
    compliance_overlays: Dict[str, Any] = field(default_factory=dict)
    protocol_configurations: Dict[str, Any] = field(default_factory=dict)
    
    supported_asset_classes: List[str] = field(default_factory=list)
    supported_markets: List[str] = field(default_factory=list)
    
    omega_directive_mappings: Dict[str, str] = field(default_factory=dict)
    
    integration_endpoints: Dict[str, str] = field(default_factory=dict)
    
    validation_status: str = "pending"
    
    @classmethod
    def create_default(cls) -> "AdaptationProfile":
        """Create default adaptation profile."""
        return cls(
            profile_id="default",
            profile_name="QuantraCore Standard",
            created_at=datetime.utcnow(),
            target_infrastructure="quantracore_native",
            supported_asset_classes=["equities", "etfs"],
            supported_markets=["US"],
            validation_status="validated",
        )
    
    @classmethod
    def create_bloomberg_compatible(cls) -> "AdaptationProfile":
        """Create Bloomberg-compatible adaptation profile."""
        return cls(
            profile_id="bloomberg",
            profile_name="Bloomberg Terminal Compatible",
            created_at=datetime.utcnow(),
            target_infrastructure="bloomberg_terminal",
            data_feed_mappings={
                "price": "PX_LAST",
                "volume": "VOLUME",
                "open": "PX_OPEN",
                "high": "PX_HIGH",
                "low": "PX_LOW",
                "vwap": "VWAP",
            },
            supported_asset_classes=["equities", "etfs", "futures", "options", "fx"],
            supported_markets=["US", "EU", "APAC", "LATAM"],
            validation_status="template",
        )
    
    @classmethod
    def create_refinitiv_compatible(cls) -> "AdaptationProfile":
        """Create Refinitiv/LSEG compatible adaptation profile."""
        return cls(
            profile_id="refinitiv",
            profile_name="Refinitiv Eikon Compatible",
            created_at=datetime.utcnow(),
            target_infrastructure="refinitiv_eikon",
            data_feed_mappings={
                "price": "TRDPRC_1",
                "volume": "ACVOL_UNS",
                "open": "OPEN_PRC",
                "high": "HIGH_1",
                "low": "LOW_1",
            },
            supported_asset_classes=["equities", "etfs", "fixed_income"],
            supported_markets=["US", "EU", "APAC"],
            validation_status="template",
        )


@dataclass
class InstitutionalLeaderboard:
    """Leaderboard of institutions ranked by battle performance."""
    generated_at: datetime
    period_start: date
    period_end: date
    
    rankings: List[Dict[str, Any]] = field(default_factory=list)
    
    top_performers: List[str] = field(default_factory=list)
    most_improved: List[str] = field(default_factory=list)
    highest_alpha: List[str] = field(default_factory=list)
    
    our_rank: int = 0
    our_percentile: float = 0.0
    
    total_institutions_tracked: int = 0
    total_battles_simulated: int = 0
    
    compliance_status: ComplianceStatus = ComplianceStatus.RESEARCH_ONLY
