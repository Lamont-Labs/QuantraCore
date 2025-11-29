"""
Institutional Strategy Analyzer.

Fingerprints institutional trading strategies by analyzing
patterns in their public 13F filings.

METHODOLOGY:
1. Concentration Analysis - How concentrated vs diversified
2. Turnover Analysis - Trading frequency patterns
3. Sector Preferences - Industry allocations
4. Position Sizing - Conviction patterns
5. Timing Patterns - Entry/exit behaviors
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from ..models import (
    Institution,
    Filing13F,
    InstitutionalHolding,
    PositionChange,
    StrategyFingerprint,
    StrategyType,
    ComplianceStatus,
)
from ..data_sources.sec_edgar import SECEdgarClient

logger = logging.getLogger(__name__)


SECTOR_CUSIP_PREFIXES = {
    "Technology": ["594918", "67066G", "02079K", "88160R", "79466L"],
    "Healthcare": ["58933Y", "09062X", "478160"],
    "Financials": ["46625H", "060505", "172967"],
    "Consumer": ["023135", "931142", "88579Y"],
    "Energy": ["20825C", "30231G", "88642R"],
    "Industrials": ["097023", "149123"],
    "Communications": ["00206R", "92343V"],
}


class StrategyAnalyzer:
    """
    Analyzes institutional trading strategies from public filings.
    
    All analysis is based on publicly available SEC 13F data.
    """
    
    def __init__(self, sec_client: Optional[SECEdgarClient] = None):
        self.sec_client = sec_client or SECEdgarClient()
        self._fingerprint_cache: Dict[str, StrategyFingerprint] = {}
        
        logger.info("[StrategyAnalyzer] Initialized with public data analysis")
    
    def fingerprint_institution(
        self,
        cik: str,
        quarters: int = 4,
    ) -> Optional[StrategyFingerprint]:
        """
        Generate a strategy fingerprint for an institution.
        
        Analyzes multiple quarters of 13F filings to identify patterns.
        """
        cik = str(cik).zfill(10)
        
        filings = self.sec_client.get_quarterly_holdings_comparison(cik, quarters)
        
        if not filings:
            logger.warning(f"[StrategyAnalyzer] No filings found for {cik}")
            return None
        
        institution_name = filings[0].institution_name if filings else "Unknown"
        
        concentration = self._analyze_concentration(filings[-1])
        turnover = self._analyze_turnover(filings) if len(filings) > 1 else 0.0
        sectors = self._analyze_sector_allocation(filings[-1])
        market_caps = self._estimate_market_cap_preferences(filings[-1])
        position_changes = self._get_position_changes(filings) if len(filings) > 1 else []
        
        strategy = self._classify_strategy(
            concentration=concentration,
            turnover=turnover,
            sectors=sectors,
            position_changes=position_changes,
        )
        
        fingerprint = StrategyFingerprint(
            institution_cik=cik,
            institution_name=institution_name,
            analysis_date=datetime.utcnow(),
            primary_strategy=strategy,
            concentration_score=concentration,
            turnover_rate=turnover,
            conviction_score=self._calculate_conviction(filings[-1]),
            sector_allocations=sectors,
            market_cap_preferences=market_caps,
            top_sectors=sorted(sectors.keys(), key=lambda x: sectors[x], reverse=True)[:5],
            top_holdings=[h.issuer_name for h in sorted(
                filings[-1].holdings, key=lambda x: x.value, reverse=True
            )[:10]],
            recent_buys=[pc.issuer_name for pc in position_changes if pc.is_new_position][:5],
            recent_sells=[pc.issuer_name for pc in position_changes if pc.is_exit][:5],
            confidence_score=min(0.9, 0.3 + (len(filings) * 0.15)),
            quarters_analyzed=len(filings),
            filings_analyzed=len(filings),
            compliance_status=ComplianceStatus.PUBLIC_DATA,
            methodology="Public 13F filing analysis",
        )
        
        self._fingerprint_cache[cik] = fingerprint
        
        logger.info(f"[StrategyAnalyzer] Fingerprinted {institution_name}: {strategy.value}")
        
        return fingerprint
    
    def _analyze_concentration(self, filing: Filing13F) -> float:
        """
        Analyze portfolio concentration (0 = diversified, 1 = concentrated).
        
        Uses Herfindahl-Hirschman Index (HHI) methodology.
        """
        if not filing.holdings or filing.total_value == 0:
            return 0.0
        
        total_value = sum(h.value for h in filing.holdings)
        if total_value == 0:
            return 0.0
        
        hhi = sum(
            ((h.value / total_value) ** 2)
            for h in filing.holdings
        )
        
        min_hhi = 1.0 / len(filing.holdings) if filing.holdings else 1.0
        max_hhi = 1.0
        
        if max_hhi == min_hhi:
            return 0.0
        
        normalized = (hhi - min_hhi) / (max_hhi - min_hhi)
        return min(1.0, max(0.0, normalized))
    
    def _analyze_turnover(self, filings: List[Filing13F]) -> float:
        """
        Analyze portfolio turnover rate between filings.
        
        Higher turnover = more active trading.
        """
        if len(filings) < 2:
            return 0.0
        
        total_turnover = 0.0
        comparisons = 0
        
        for i in range(1, len(filings)):
            current = filings[i - 1]
            previous = filings[i]
            
            current_cusips = {h.cusip: h for h in current.holdings}
            previous_cusips = {h.cusip: h for h in previous.holdings}
            
            new_positions = set(current_cusips.keys()) - set(previous_cusips.keys())
            exited = set(previous_cusips.keys()) - set(current_cusips.keys())
            
            new_value = sum(current_cusips[c].value for c in new_positions)
            exited_value = sum(previous_cusips[c].value for c in exited)
            
            avg_portfolio = (current.total_value + previous.total_value) / 2
            if avg_portfolio > 0:
                turnover = (new_value + exited_value) / avg_portfolio
                total_turnover += turnover
                comparisons += 1
        
        if comparisons == 0:
            return 0.0
        
        return total_turnover / comparisons
    
    def _analyze_sector_allocation(self, filing: Filing13F) -> Dict[str, float]:
        """Analyze sector allocations from holdings."""
        if not filing.holdings or filing.total_value == 0:
            return {}
        
        sector_values: Dict[str, float] = defaultdict(float)
        total = sum(h.value for h in filing.holdings)
        
        for holding in filing.holdings:
            sector = self._infer_sector(holding)
            sector_values[sector] += holding.value
        
        return {
            sector: value / total
            for sector, value in sector_values.items()
            if value > 0
        }
    
    def _infer_sector(self, holding: InstitutionalHolding) -> str:
        """Infer sector from holding information."""
        name_lower = holding.issuer_name.lower()
        
        tech_keywords = ["apple", "microsoft", "google", "nvidia", "meta", "amazon", "tesla"]
        healthcare_keywords = ["pfizer", "johnson", "unitedhealth", "merck", "abbvie"]
        finance_keywords = ["jpmorgan", "bank", "goldman", "morgan stanley", "visa"]
        energy_keywords = ["exxon", "chevron", "conocophillips", "schlumberger"]
        
        if any(kw in name_lower for kw in tech_keywords):
            return "Technology"
        elif any(kw in name_lower for kw in healthcare_keywords):
            return "Healthcare"
        elif any(kw in name_lower for kw in finance_keywords):
            return "Financials"
        elif any(kw in name_lower for kw in energy_keywords):
            return "Energy"
        else:
            return "Other"
    
    def _estimate_market_cap_preferences(self, filing: Filing13F) -> Dict[str, float]:
        """
        Estimate market cap preferences from holding sizes.
        
        Larger holdings (by value) tend to be larger companies.
        """
        if not filing.holdings:
            return {}
        
        sorted_holdings = sorted(filing.holdings, key=lambda x: x.value, reverse=True)
        total = sum(h.value for h in sorted_holdings)
        
        if total == 0:
            return {}
        
        mega_cap = sum(h.value for h in sorted_holdings[:5]) / total
        large_cap = sum(h.value for h in sorted_holdings[5:20]) / total
        mid_cap = sum(h.value for h in sorted_holdings[20:50]) / total
        small_cap = sum(h.value for h in sorted_holdings[50:]) / total
        
        return {
            "mega_cap": mega_cap,
            "large_cap": large_cap,
            "mid_cap": mid_cap,
            "small_cap": small_cap,
        }
    
    def _get_position_changes(self, filings: List[Filing13F]) -> List[PositionChange]:
        """Calculate position changes between most recent filings."""
        if len(filings) < 2:
            return []
        
        current = filings[0]
        previous = filings[1]
        
        current_holdings = {h.cusip: h for h in current.holdings}
        previous_holdings = {h.cusip: h for h in previous.holdings}
        
        all_cusips = set(current_holdings.keys()) | set(previous_holdings.keys())
        
        changes = []
        for cusip in all_cusips:
            curr = current_holdings.get(cusip)
            prev = previous_holdings.get(cusip)
            
            curr_shares = curr.shares if curr else 0
            prev_shares = prev.shares if prev else 0
            curr_value = curr.value if curr else 0
            prev_value = prev.value if prev else 0
            
            shares_change = curr_shares - prev_shares
            value_change = curr_value - prev_value
            
            if shares_change == 0:
                continue
            
            if prev_shares == 0:
                action = "NEW"
            elif curr_shares == 0:
                action = "EXIT"
            elif shares_change > 0:
                action = "INCREASE"
            else:
                action = "DECREASE"
            
            changes.append(PositionChange(
                symbol="",
                cusip=cusip,
                issuer_name=(curr or prev).issuer_name,
                prior_shares=prev_shares,
                current_shares=curr_shares,
                shares_change=shares_change,
                shares_change_pct=shares_change / prev_shares if prev_shares > 0 else float('inf'),
                prior_value=prev_value,
                current_value=curr_value,
                value_change=value_change,
                value_change_pct=value_change / prev_value if prev_value > 0 else float('inf'),
                action=action,
            ))
        
        return sorted(changes, key=lambda x: abs(x.value_change), reverse=True)
    
    def _calculate_conviction(self, filing: Filing13F) -> float:
        """
        Calculate conviction score based on position sizing.
        
        High conviction = large positions relative to portfolio.
        """
        if not filing.holdings or filing.total_value == 0:
            return 0.0
        
        total = sum(h.value for h in filing.holdings)
        if total == 0:
            return 0.0
        
        weights = sorted(
            [h.value / total for h in filing.holdings],
            reverse=True
        )
        
        top_5_weight = sum(weights[:5]) if len(weights) >= 5 else sum(weights)
        
        return min(1.0, top_5_weight * 1.5)
    
    def _classify_strategy(
        self,
        concentration: float,
        turnover: float,
        sectors: Dict[str, float],
        position_changes: List[PositionChange],
    ) -> StrategyType:
        """Classify strategy based on analyzed patterns."""
        if concentration > 0.7:
            if turnover < 0.2:
                return StrategyType.VALUE
            else:
                return StrategyType.ACTIVIST
        
        if turnover > 0.5:
            return StrategyType.QUANTITATIVE
        
        tech_weight = sectors.get("Technology", 0)
        if tech_weight > 0.4:
            return StrategyType.GROWTH
        
        if concentration < 0.3 and turnover < 0.3:
            return StrategyType.LONG_ONLY
        
        if len(sectors) > 5 and concentration < 0.4:
            return StrategyType.MULTI_STRATEGY
        
        return StrategyType.UNKNOWN
    
    def compare_fingerprints(
        self,
        fp1: StrategyFingerprint,
        fp2: StrategyFingerprint,
    ) -> Dict[str, Any]:
        """Compare two strategy fingerprints."""
        return {
            "concentration_diff": fp1.concentration_score - fp2.concentration_score,
            "turnover_diff": fp1.turnover_rate - fp2.turnover_rate,
            "conviction_diff": fp1.conviction_score - fp2.conviction_score,
            "strategy_match": fp1.primary_strategy == fp2.primary_strategy,
            "sector_overlap": self._calculate_sector_overlap(
                fp1.sector_allocations,
                fp2.sector_allocations,
            ),
        }
    
    def _calculate_sector_overlap(
        self,
        sectors1: Dict[str, float],
        sectors2: Dict[str, float],
    ) -> float:
        """Calculate sector allocation overlap between two portfolios."""
        all_sectors = set(sectors1.keys()) | set(sectors2.keys())
        
        overlap = 0.0
        for sector in all_sectors:
            w1 = sectors1.get(sector, 0)
            w2 = sectors2.get(sector, 0)
            overlap += min(w1, w2)
        
        return overlap
    
    def get_institutions_by_strategy(
        self,
        strategy: StrategyType,
    ) -> List[StrategyFingerprint]:
        """Get all fingerprinted institutions matching a strategy."""
        return [
            fp for fp in self._fingerprint_cache.values()
            if fp.primary_strategy == strategy
        ]
