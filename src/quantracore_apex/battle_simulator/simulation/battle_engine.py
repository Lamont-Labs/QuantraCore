"""
Battle Simulation Engine.

Compares our trading signals against institutional actions
discovered in subsequent 13F filings.

METHODOLOGY:
- Uses historical price data to evaluate trades
- Compares timing, sizing, and returns
- All comparisons are backtested (not real-time)
- Educational/research purposes only

COMPLIANCE:
- Uses only public data (SEC filings, historical prices)
- No forward-looking claims
- Research and educational purposes only
"""

import logging
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from ..models import (
    Institution,
    Filing13F,
    PositionChange,
    StrategyFingerprint,
    BattleScenario,
    BattleResult,
    BattleOutcome,
    InstitutionalLeaderboard,
    ComplianceStatus,
)
from ..data_sources.sec_edgar import SECEdgarClient
from ..fingerprinting.strategy_analyzer import StrategyAnalyzer

logger = logging.getLogger(__name__)


class BattleEngine:
    """
    Engine for simulating battles against institutional investors.
    
    Compares our hypothetical signals against what institutions
    actually did (as revealed in subsequent 13F filings).
    
    IMPORTANT: This is backtesting only, not live trading.
    All comparisons use historical data after the fact.
    """
    
    def __init__(
        self,
        sec_client: Optional[SECEdgarClient] = None,
        strategy_analyzer: Optional[StrategyAnalyzer] = None,
    ):
        self.sec_client = sec_client or SECEdgarClient()
        self.strategy_analyzer = strategy_analyzer or StrategyAnalyzer(self.sec_client)
        
        self._battle_history: List[BattleResult] = []
        self._institution_scores: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total_alpha": 0.0}
        )
        
        logger.info("[BattleEngine] Initialized for backtested analysis")
    
    def create_battle_scenario(
        self,
        symbol: str,
        our_signal_date: date,
        our_signal_direction: str,
        our_quantrascore: float,
        our_entry_price: float,
        our_exit_price: Optional[float] = None,
        end_date: Optional[date] = None,
    ) -> BattleScenario:
        """
        Create a battle scenario from our signal.
        
        The scenario will be used to compare against institutional
        actions in subsequent 13F filings.
        """
        scenario_id = f"BATTLE-{symbol}-{our_signal_date.isoformat()}-{uuid.uuid4().hex[:8]}"
        
        return BattleScenario(
            scenario_id=scenario_id,
            name=f"{symbol} Battle {our_signal_date}",
            description=f"Battle scenario for {symbol} signal on {our_signal_date}",
            symbol=symbol,
            start_date=our_signal_date,
            end_date=end_date or (our_signal_date + timedelta(days=90)),
            our_signal_date=our_signal_date,
            our_signal_direction=our_signal_direction,
            our_quantrascore=our_quantrascore,
            our_entry_price=our_entry_price,
            our_exit_price=our_exit_price,
        )
    
    def battle_against_institution(
        self,
        scenario: BattleScenario,
        institution_cik: str,
        historical_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[BattleResult]:
        """
        Battle our signal against a specific institution.
        
        Compares our hypothetical trade against what the institution
        actually did, as revealed in their 13F filings.
        """
        institution_cik = str(institution_cik).zfill(10)
        
        filings = self.sec_client.get_quarterly_holdings_comparison(
            institution_cik, quarters=4
        )
        
        if len(filings) < 2:
            logger.warning(f"[BattleEngine] Not enough filings for {institution_cik}")
            return None
        
        symbol_cusip = self._find_cusip_for_symbol(scenario.symbol, filings)
        if not symbol_cusip:
            logger.info(f"[BattleEngine] {scenario.symbol} not found in {institution_cik}'s holdings")
            return None
        
        institution_action = self._determine_institution_action(
            filings, symbol_cusip, scenario.start_date
        )
        
        our_return = self._calculate_our_return(scenario, historical_prices)
        inst_return = self._estimate_institution_return(
            institution_action, scenario, historical_prices
        )
        
        alpha = our_return - inst_return
        
        if alpha > 0.02:
            outcome = BattleOutcome.WIN
        elif alpha < -0.02:
            outcome = BattleOutcome.LOSS
        elif abs(our_return) < 0.001 and abs(inst_return) < 0.001:
            outcome = BattleOutcome.INCONCLUSIVE
        else:
            outcome = BattleOutcome.TIE
        
        lessons = self._extract_lessons(scenario, institution_action, alpha)
        
        result = BattleResult(
            battle_id=f"{scenario.scenario_id}-{institution_cik}",
            scenario=scenario,
            outcome=outcome,
            our_return_pct=our_return * 100,
            institution_return_pct=inst_return * 100,
            alpha_generated=alpha * 100,
            our_timing_score=self._score_timing(scenario, institution_action),
            institution_timing_score=self._score_institution_timing(institution_action),
            timing_advantage=0.0,
            lessons_learned=lessons,
            compliance_status=ComplianceStatus.RESEARCH_ONLY,
            methodology="Backtested comparison using public data",
        )
        
        self._record_result(result, institution_cik)
        
        return result
    
    def battle_against_top_institutions(
        self,
        scenario: BattleScenario,
        top_n: int = 10,
        historical_prices: Optional[Dict[str, float]] = None,
    ) -> List[BattleResult]:
        """
        Battle our signal against top institutional investors.
        
        Compares our trade against multiple major institutions.
        """
        results = []
        
        top_institutions = self.sec_client.get_top_institutions()
        
        for institution in top_institutions[:top_n]:
            result = self.battle_against_institution(
                scenario, institution.cik, historical_prices
            )
            if result:
                results.append(result)
        
        return results
    
    def _find_cusip_for_symbol(
        self,
        symbol: str,
        filings: List[Filing13F],
    ) -> Optional[str]:
        """Find CUSIP for a symbol in filings."""
        symbol_lower = symbol.lower()
        
        for filing in filings:
            for holding in filing.holdings:
                if symbol_lower in holding.issuer_name.lower():
                    return holding.cusip
        
        return None
    
    def _determine_institution_action(
        self,
        filings: List[Filing13F],
        cusip: str,
        signal_date: date,
    ) -> Dict[str, Any]:
        """
        Determine what the institution did with a position.
        
        Analyzes changes between filings around the signal date.
        """
        relevant_filings = [
            f for f in filings
            if f.period_of_report >= signal_date - timedelta(days=90)
        ]
        
        if len(relevant_filings) < 2:
            return {"action": "UNKNOWN", "details": "Insufficient data"}
        
        recent = relevant_filings[0]
        previous = relevant_filings[1] if len(relevant_filings) > 1 else None
        
        recent_holding = next(
            (h for h in recent.holdings if h.cusip == cusip), None
        )
        previous_holding = next(
            (h for h in previous.holdings if h.cusip == cusip), None
        ) if previous else None
        
        recent_shares = recent_holding.shares if recent_holding else 0
        previous_shares = previous_holding.shares if previous_holding else 0
        
        if previous_shares == 0 and recent_shares > 0:
            action = "NEW_POSITION"
        elif previous_shares > 0 and recent_shares == 0:
            action = "EXIT"
        elif recent_shares > previous_shares:
            action = "INCREASE"
        elif recent_shares < previous_shares:
            action = "DECREASE"
        else:
            action = "HOLD"
        
        return {
            "action": action,
            "prior_shares": previous_shares,
            "current_shares": recent_shares,
            "shares_change": recent_shares - previous_shares,
            "prior_value": previous_holding.value if previous_holding else 0,
            "current_value": recent_holding.value if recent_holding else 0,
            "filing_date": recent.filing_date,
            "period": recent.period_of_report,
        }
    
    def _calculate_our_return(
        self,
        scenario: BattleScenario,
        historical_prices: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate our hypothetical return from the scenario."""
        if scenario.our_exit_price and scenario.our_entry_price > 0:
            return (scenario.our_exit_price - scenario.our_entry_price) / scenario.our_entry_price
        
        if historical_prices:
            end_price = historical_prices.get(scenario.end_date.isoformat())
            if end_price and scenario.our_entry_price > 0:
                return (end_price - scenario.our_entry_price) / scenario.our_entry_price
        
        return 0.0
    
    def _estimate_institution_return(
        self,
        action: Dict[str, Any],
        scenario: BattleScenario,
        historical_prices: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Estimate institution's return based on their action.
        
        Note: This is an approximation since we don't know exact
        entry/exit prices from 13F filings.
        """
        action_type = action.get("action", "UNKNOWN")
        
        if action_type == "EXIT":
            return 0.0
        
        if action_type in ["NEW_POSITION", "INCREASE"]:
            return self._calculate_our_return(scenario, historical_prices) * 0.8
        
        if action_type == "HOLD":
            return self._calculate_our_return(scenario, historical_prices)
        
        if action_type == "DECREASE":
            return self._calculate_our_return(scenario, historical_prices) * 0.5
        
        return 0.0
    
    def _score_timing(
        self,
        scenario: BattleScenario,
        institution_action: Dict[str, Any],
    ) -> float:
        """Score our signal timing (0-1)."""
        base_score = min(1.0, scenario.our_quantrascore / 100)
        
        return base_score
    
    def _score_institution_timing(
        self,
        action: Dict[str, Any],
    ) -> float:
        """Score institution's timing based on their action."""
        action_type = action.get("action", "UNKNOWN")
        
        timing_scores = {
            "NEW_POSITION": 0.8,
            "INCREASE": 0.7,
            "HOLD": 0.5,
            "DECREASE": 0.4,
            "EXIT": 0.3,
            "UNKNOWN": 0.5,
        }
        
        return timing_scores.get(action_type, 0.5)
    
    def _extract_lessons(
        self,
        scenario: BattleScenario,
        institution_action: Dict[str, Any],
        alpha: float,
    ) -> List[str]:
        """Extract lessons from the battle."""
        lessons = []
        
        action = institution_action.get("action", "UNKNOWN")
        
        if alpha > 0.05:
            lessons.append(f"Strong outperformance ({alpha*100:.1f}% alpha) - signal timing was superior")
        elif alpha < -0.05:
            lessons.append(f"Underperformance ({alpha*100:.1f}% alpha) - institution had better timing")
        
        if action in ["NEW_POSITION", "INCREASE"]:
            if scenario.our_signal_direction.upper() == "LONG":
                lessons.append("Institution agreed with bullish thesis")
            else:
                lessons.append("Institution took opposite view - worth investigating")
        
        if action == "EXIT" and scenario.our_signal_direction.upper() == "LONG":
            lessons.append("Institution exited while we were bullish - risk signal")
        
        if scenario.our_quantrascore > 75 and alpha < 0:
            lessons.append("High conviction signal underperformed - review scoring criteria")
        
        return lessons
    
    def _record_result(self, result: BattleResult, institution_cik: str) -> None:
        """Record battle result for analytics."""
        self._battle_history.append(result)
        
        scores = self._institution_scores[institution_cik]
        if result.outcome == BattleOutcome.WIN:
            scores["wins"] += 1
        elif result.outcome == BattleOutcome.LOSS:
            scores["losses"] += 1
        scores["total_alpha"] += result.alpha_generated
    
    def get_leaderboard(self) -> InstitutionalLeaderboard:
        """Generate leaderboard from battle history."""
        if not self._battle_history:
            return InstitutionalLeaderboard(
                generated_at=datetime.utcnow(),
                period_start=date.today() - timedelta(days=365),
                period_end=date.today(),
            )
        
        rankings = []
        for cik, scores in self._institution_scores.items():
            total_battles = scores["wins"] + scores["losses"]
            if total_battles > 0:
                win_rate = scores["wins"] / total_battles
                rankings.append({
                    "cik": cik,
                    "wins": scores["wins"],
                    "losses": scores["losses"],
                    "win_rate": win_rate,
                    "total_alpha": scores["total_alpha"],
                })
        
        rankings = sorted(rankings, key=lambda x: x["total_alpha"], reverse=True)
        
        our_alpha = sum(r.alpha_generated for r in self._battle_history)
        our_wins = sum(1 for r in self._battle_history if r.outcome == BattleOutcome.WIN)
        
        return InstitutionalLeaderboard(
            generated_at=datetime.utcnow(),
            period_start=min(r.scenario.start_date for r in self._battle_history),
            period_end=max(r.scenario.end_date for r in self._battle_history),
            rankings=rankings,
            top_performers=[r["cik"] for r in rankings[:5]],
            highest_alpha=[r["cik"] for r in sorted(rankings, key=lambda x: x["total_alpha"], reverse=True)[:5]],
            our_rank=1,
            total_institutions_tracked=len(rankings),
            total_battles_simulated=len(self._battle_history),
            compliance_status=ComplianceStatus.RESEARCH_ONLY,
        )
    
    def get_battle_statistics(self) -> Dict[str, Any]:
        """Get overall battle statistics."""
        if not self._battle_history:
            return {
                "total_battles": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "win_rate": 0.0,
                "avg_alpha": 0.0,
            }
        
        wins = sum(1 for r in self._battle_history if r.outcome == BattleOutcome.WIN)
        losses = sum(1 for r in self._battle_history if r.outcome == BattleOutcome.LOSS)
        ties = sum(1 for r in self._battle_history if r.outcome == BattleOutcome.TIE)
        
        total = wins + losses
        
        return {
            "total_battles": len(self._battle_history),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / total if total > 0 else 0.0,
            "avg_alpha": sum(r.alpha_generated for r in self._battle_history) / len(self._battle_history),
            "total_alpha": sum(r.alpha_generated for r in self._battle_history),
            "compliance_status": ComplianceStatus.RESEARCH_ONLY.value,
        }
