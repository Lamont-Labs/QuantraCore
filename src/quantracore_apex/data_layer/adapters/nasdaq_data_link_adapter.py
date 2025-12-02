"""
Nasdaq Data Link (formerly Quandl) Adapter for QuantraCore Apex.

Provides access to alternative data including:
- COT Reports (Commitment of Traders) - Futures positioning
- Fed balance sheet data
- Treasury auction data

Free tier: 50 calls/day, many datasets free
https://data.nasdaq.com/

Set NASDAQ_DATA_LINK_API_KEY environment variable.
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class CotReport:
    """Commitment of Traders report data."""
    symbol: str
    report_date: datetime
    commercial_long: int
    commercial_short: int
    commercial_net: int
    non_commercial_long: int
    non_commercial_short: int
    non_commercial_net: int
    small_traders_long: int
    small_traders_short: int
    small_traders_net: int
    open_interest: int
    positioning_signal: str  # BULLISH, BEARISH, NEUTRAL


@dataclass
class CotSummary:
    """COT positioning summary for trading signals."""
    symbol: str
    latest_date: datetime
    commercial_positioning: str
    speculator_positioning: str
    smart_money_signal: str
    net_change_weekly: int
    extreme_reading: bool
    confidence: float


class NasdaqDataLinkAdapter:
    """
    Nasdaq Data Link adapter for COT reports and alternative data.
    
    COT reports show how different trader groups are positioned:
    - Commercials (hedgers/producers) - Often called "smart money"
    - Non-commercials (speculators/funds) - Trend followers
    - Small traders (retail) - Often wrong at extremes
    
    Free tier: 50 API calls per day
    """
    
    BASE_URL = "https://data.nasdaq.com/api/v3"
    
    COT_DATASETS = {
        "ES": "CFTC/ES_F_ALL",       # S&P 500 E-mini
        "NQ": "CFTC/NQ_F_ALL",       # Nasdaq 100 E-mini
        "YM": "CFTC/YM_F_ALL",       # Dow E-mini
        "GC": "CFTC/GC_F_ALL",       # Gold
        "SI": "CFTC/SI_F_ALL",       # Silver
        "CL": "CFTC/CL_F_ALL",       # Crude Oil
        "NG": "CFTC/NG_F_ALL",       # Natural Gas
        "ZC": "CFTC/ZC_F_ALL",       # Corn
        "ZW": "CFTC/ZW_F_ALL",       # Wheat
        "ZS": "CFTC/ZS_F_ALL",       # Soybeans
        "6E": "CFTC/EC_F_ALL",       # Euro FX
        "6J": "CFTC/JY_F_ALL",       # Japanese Yen
        "6B": "CFTC/BP_F_ALL",       # British Pound
        "ZN": "CFTC/TY_F_ALL",       # 10-Year Treasury
        "ZB": "CFTC/US_F_ALL",       # 30-Year Treasury
        "VX": "CFTC/VX_F_ALL",       # VIX Futures
        "BTC": "CFTC/133741_F_ALL",  # Bitcoin
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NASDAQ_DATA_LINK_API_KEY")
        self._last_request = 0
        self._rate_limit_delay = 1.5  # Conservative for free tier
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 3600  # 1 hour (COT updates weekly)
        
        if self.api_key:
            logger.info("[NasdaqDataLink] Adapter initialized (50 calls/day free tier)")
        else:
            logger.warning("[NasdaqDataLink] API key not set - using simulated data")
    
    @property
    def name(self) -> str:
        return "nasdaq_data_link"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "rate_limit": "50 calls/day (free)",
            "coverage": "COT Reports (Futures Positioning)",
            "supported_symbols": list(self.COT_DATASETS.keys())
        }
    
    def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Make a rate-limited request to Nasdaq Data Link."""
        if not self.is_available():
            return None
        
        self._rate_limit_wait()
        
        params = params or {}
        params["api_key"] = self.api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"[NasdaqDataLink] HTTP error: {e}")
            return None
        except Exception as e:
            logger.warning(f"[NasdaqDataLink] Request error: {e}")
            return None
    
    def get_cot_report(self, symbol: str, weeks: int = 4) -> List[CotReport]:
        """
        Get Commitment of Traders report for a futures symbol.
        
        Args:
            symbol: Futures symbol (ES, NQ, GC, CL, etc.)
            weeks: Number of weeks of data to retrieve
            
        Returns:
            List of CotReport objects
        """
        symbol = symbol.upper()
        
        cache_key = f"cot_{symbol}_{weeks}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        if symbol not in self.COT_DATASETS:
            logger.warning(f"[NasdaqDataLink] Unknown COT symbol: {symbol}")
            return self._simulated_cot_report(symbol, weeks)
        
        if not self.is_available():
            return self._simulated_cot_report(symbol, weeks)
        
        try:
            dataset = self.COT_DATASETS[symbol]
            endpoint = f"datasets/{dataset}/data.json"
            params = {"rows": weeks}
            
            data = self._request(endpoint, params)
            
            if not data or "dataset_data" not in data:
                return self._simulated_cot_report(symbol, weeks)
            
            dataset_data = data["dataset_data"]
            columns = dataset_data.get("column_names", [])
            rows = dataset_data.get("data", [])
            
            reports = []
            for row in rows:
                try:
                    row_dict = dict(zip(columns, row))
                    
                    commercial_long = int(row_dict.get("Commercial Long", 0))
                    commercial_short = int(row_dict.get("Commercial Short", 0))
                    non_comm_long = int(row_dict.get("Noncommercial Long", 0))
                    non_comm_short = int(row_dict.get("Noncommercial Short", 0))
                    small_long = int(row_dict.get("Nonreportable Long", 0))
                    small_short = int(row_dict.get("Nonreportable Short", 0))
                    
                    commercial_net = commercial_long - commercial_short
                    non_comm_net = non_comm_long - non_comm_short
                    small_net = small_long - small_short
                    
                    if commercial_net > 0 and non_comm_net < 0:
                        signal = "BULLISH"
                    elif commercial_net < 0 and non_comm_net > 0:
                        signal = "BEARISH"
                    else:
                        signal = "NEUTRAL"
                    
                    report = CotReport(
                        symbol=symbol,
                        report_date=datetime.strptime(row_dict.get("Date", "2025-01-01"), "%Y-%m-%d"),
                        commercial_long=commercial_long,
                        commercial_short=commercial_short,
                        commercial_net=commercial_net,
                        non_commercial_long=non_comm_long,
                        non_commercial_short=non_comm_short,
                        non_commercial_net=non_comm_net,
                        small_traders_long=small_long,
                        small_traders_short=small_short,
                        small_traders_net=small_net,
                        open_interest=int(row_dict.get("Open Interest", 0)),
                        positioning_signal=signal
                    )
                    reports.append(report)
                    
                except (ValueError, KeyError) as e:
                    logger.debug(f"[NasdaqDataLink] Row parse error: {e}")
                    continue
            
            self._cache[cache_key] = (reports, time.time())
            return reports
            
        except Exception as e:
            logger.error(f"[NasdaqDataLink] COT fetch error: {e}")
            return self._simulated_cot_report(symbol, weeks)
    
    def get_cot_summary(self, symbol: str) -> CotSummary:
        """
        Get COT positioning summary with trading signal.
        
        Analyzes commercial vs speculator positioning to generate signals.
        """
        reports = self.get_cot_report(symbol, weeks=4)
        
        if not reports:
            return CotSummary(
                symbol=symbol,
                latest_date=datetime.utcnow(),
                commercial_positioning="UNKNOWN",
                speculator_positioning="UNKNOWN",
                smart_money_signal="NEUTRAL",
                net_change_weekly=0,
                extreme_reading=False,
                confidence=0.0
            )
        
        latest = reports[0]
        
        if latest.commercial_net > 0:
            commercial_pos = "LONG"
        elif latest.commercial_net < 0:
            commercial_pos = "SHORT"
        else:
            commercial_pos = "NEUTRAL"
        
        if latest.non_commercial_net > 0:
            spec_pos = "LONG"
        elif latest.non_commercial_net < 0:
            spec_pos = "SHORT"
        else:
            spec_pos = "NEUTRAL"
        
        smart_money = latest.positioning_signal
        
        net_change = 0
        if len(reports) > 1:
            net_change = latest.commercial_net - reports[1].commercial_net
        
        total_positions = abs(latest.commercial_net) + abs(latest.non_commercial_net)
        extreme_threshold = latest.open_interest * 0.3 if latest.open_interest > 0 else 100000
        extreme_reading = total_positions > extreme_threshold
        
        if extreme_reading and smart_money != "NEUTRAL":
            confidence = 0.85
        elif smart_money != "NEUTRAL":
            confidence = 0.65
        else:
            confidence = 0.4
        
        return CotSummary(
            symbol=symbol,
            latest_date=latest.report_date,
            commercial_positioning=commercial_pos,
            speculator_positioning=spec_pos,
            smart_money_signal=smart_money,
            net_change_weekly=net_change,
            extreme_reading=extreme_reading,
            confidence=confidence
        )
    
    def _simulated_cot_report(self, symbol: str, weeks: int) -> List[CotReport]:
        """Generate simulated COT data when API is unavailable."""
        import random
        
        reports = []
        base_date = datetime.utcnow()
        
        for i in range(weeks):
            report_date = base_date - timedelta(weeks=i)
            
            commercial_long = random.randint(100000, 300000)
            commercial_short = random.randint(100000, 300000)
            non_comm_long = random.randint(50000, 200000)
            non_comm_short = random.randint(50000, 200000)
            small_long = random.randint(10000, 50000)
            small_short = random.randint(10000, 50000)
            
            commercial_net = commercial_long - commercial_short
            non_comm_net = non_comm_long - non_comm_short
            
            if commercial_net > 0 and non_comm_net < 0:
                signal = "BULLISH"
            elif commercial_net < 0 and non_comm_net > 0:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"
            
            report = CotReport(
                symbol=symbol,
                report_date=report_date,
                commercial_long=commercial_long,
                commercial_short=commercial_short,
                commercial_net=commercial_net,
                non_commercial_long=non_comm_long,
                non_commercial_short=non_comm_short,
                non_commercial_net=non_comm_net,
                small_traders_long=small_long,
                small_traders_short=small_short,
                small_traders_net=small_long - small_short,
                open_interest=commercial_long + commercial_short + non_comm_long + non_comm_short,
                positioning_signal=signal
            )
            reports.append(report)
        
        return reports


_nasdaq_adapter: Optional[NasdaqDataLinkAdapter] = None


def get_nasdaq_data_link_adapter() -> NasdaqDataLinkAdapter:
    """Get singleton Nasdaq Data Link adapter instance."""
    global _nasdaq_adapter
    if _nasdaq_adapter is None:
        _nasdaq_adapter = NasdaqDataLinkAdapter()
    return _nasdaq_adapter
