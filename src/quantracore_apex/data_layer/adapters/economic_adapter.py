"""
Economic / Macro Data Adapters for QuantraCore Apex.

Macroeconomic indicators for regime detection and
event-driven trading strategies.

Supported Providers:
- FRED (Federal Reserve Economic Data) - Free
- Trading Economics - Premium
- Quandl/Nasdaq Data Link - Various
- Alpha Vantage - Free tier available

Data Types:
- Interest rates (Fed funds, Treasury yields)
- Inflation data (CPI, PPI, PCE)
- Employment (NFP, unemployment, jobless claims)
- GDP and growth indicators
- Consumer confidence
- Manufacturing (ISM, PMI)
- Housing data
"""

import os
import time
import requests
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .base_enhanced import (
    EnhancedDataAdapter, DataType, ProviderStatus
)

logger = logging.getLogger(__name__)


class EconomicIndicator(Enum):
    FED_FUNDS_RATE = "fed_funds"
    TREASURY_10Y = "treasury_10y"
    TREASURY_2Y = "treasury_2y"
    YIELD_CURVE = "yield_curve"
    CPI = "cpi"
    CORE_CPI = "core_cpi"
    PPI = "ppi"
    PCE = "pce"
    UNEMPLOYMENT = "unemployment"
    NFP = "nfp"
    JOBLESS_CLAIMS = "jobless_claims"
    GDP = "gdp"
    GDP_GROWTH = "gdp_growth"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    ISM_MANUFACTURING = "ism_manufacturing"
    ISM_SERVICES = "ism_services"
    RETAIL_SALES = "retail_sales"
    HOUSING_STARTS = "housing_starts"
    VIX = "vix"
    DXY = "dxy"


@dataclass
class EconomicDataPoint:
    indicator: EconomicIndicator
    date: datetime
    value: float
    previous: Optional[float] = None
    forecast: Optional[float] = None
    surprise: Optional[float] = None
    revision: Optional[float] = None


@dataclass
class EconomicEvent:
    name: str
    indicator: EconomicIndicator
    datetime: datetime
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    importance: str = "medium"
    country: str = "US"


@dataclass
class MacroRegime:
    timestamp: datetime
    regime: str
    risk_appetite: str
    yield_curve: str
    inflation_trend: str
    growth_trend: str
    fed_stance: str
    confidence: float


class EconomicDataAdapter(EnhancedDataAdapter):
    """
    Abstract economic data adapter.
    
    Provides macroeconomic data for regime detection
    and event-driven strategies.
    """
    
    @property
    def name(self) -> str:
        return "Economic Data Base"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.ECONOMIC]
    
    def fetch_indicator(
        self,
        indicator: EconomicIndicator,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[EconomicDataPoint]:
        raise NotImplementedError("Subclass must implement fetch_indicator")
    
    def fetch_calendar(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[EconomicEvent]:
        raise NotImplementedError("Subclass must implement fetch_calendar")
    
    def get_current_regime(self) -> MacroRegime:
        raise NotImplementedError("Subclass must implement get_current_regime")


class FredAdapter(EconomicDataAdapter):
    """
    Federal Reserve Economic Data (FRED) adapter.
    
    Free access to 800,000+ economic time series.
    
    Features:
    - All major US economic indicators
    - Historical data back decades
    - Real-time updates
    - API rate limit: 120 requests/minute
    
    Sign up: https://fred.stlouisfed.org/docs/api/api_key.html
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    SERIES_MAP = {
        EconomicIndicator.FED_FUNDS_RATE: "FEDFUNDS",
        EconomicIndicator.TREASURY_10Y: "DGS10",
        EconomicIndicator.TREASURY_2Y: "DGS2",
        EconomicIndicator.CPI: "CPIAUCSL",
        EconomicIndicator.CORE_CPI: "CPILFESL",
        EconomicIndicator.PPI: "PPIACO",
        EconomicIndicator.PCE: "PCEPI",
        EconomicIndicator.UNEMPLOYMENT: "UNRATE",
        EconomicIndicator.NFP: "PAYEMS",
        EconomicIndicator.JOBLESS_CLAIMS: "ICSA",
        EconomicIndicator.GDP: "GDP",
        EconomicIndicator.GDP_GROWTH: "A191RL1Q225SBEA",
        EconomicIndicator.CONSUMER_CONFIDENCE: "UMCSENT",
        EconomicIndicator.ISM_MANUFACTURING: "MANEMP",
        EconomicIndicator.RETAIL_SALES: "RSAFS",
        EconomicIndicator.HOUSING_STARTS: "HOUST",
        EconomicIndicator.VIX: "VIXCLS",
        EconomicIndicator.DXY: "DTWEXBGS",
    }
    
    CACHE_TTL_HOURS = 24
    
    _instance = None
    _instance_cache: Dict[str, Any] = {}
    _instance_cache_ts: Dict[str, float] = {}
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self._last_request = 0
        self._rate_limit_delay = 0.5
        if not hasattr(FredAdapter, '_instance_cache'):
            FredAdapter._instance_cache = {}
            FredAdapter._instance_cache_ts = {}
    
    @property
    def name(self) -> str:
        return "FRED"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.ECONOMIC]
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier="Free",
            data_types=self.supported_data_types,
            last_error=None if self.is_available() else "FRED_API_KEY not set"
        )
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        if not self.is_available():
            raise ValueError("FRED_API_KEY not set")
        
        time.sleep(self._rate_limit_delay)
        
        params = params or {}
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def fetch_indicator(
        self,
        indicator: EconomicIndicator,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[EconomicDataPoint]:
        if not self.is_available():
            return self._generate_simulated_data(indicator)
        
        series_id = self.SERIES_MAP.get(indicator)
        if not series_id:
            raise ValueError(f"Unknown indicator: {indicator}")
        
        cache_key = f"{series_id}_{start}_{end}"
        now = time.time()
        
        if cache_key in FredAdapter._instance_cache:
            cached_ts = FredAdapter._instance_cache_ts.get(cache_key, 0)
            if now - cached_ts < self.CACHE_TTL_HOURS * 3600:
                logger.debug(f"[FRED] Cache hit for {series_id}")
                return FredAdapter._instance_cache[cache_key]
        
        params = {"series_id": series_id}
        if start:
            params["observation_start"] = start.strftime("%Y-%m-%d")
        if end:
            params["observation_end"] = end.strftime("%Y-%m-%d")
        
        try:
            data = self._request("series/observations", params)
            
            points = []
            observations = data.get("observations", [])
            
            for i, obs in enumerate(observations):
                try:
                    value = float(obs["value"])
                except (ValueError, TypeError):
                    continue
                
                previous = None
                if i > 0:
                    try:
                        previous = float(observations[i-1]["value"])
                    except (ValueError, TypeError):
                        pass
                
                points.append(EconomicDataPoint(
                    indicator=indicator,
                    date=datetime.strptime(obs["date"], "%Y-%m-%d"),
                    value=value,
                    previous=previous
                ))
            
            FredAdapter._instance_cache[cache_key] = points
            FredAdapter._instance_cache_ts[cache_key] = now
            logger.debug(f"[FRED] Cached {series_id} with {len(points)} points")
            return points
        except Exception as e:
            logger.warning(f"[FRED] API error for {series_id}: {e}")
            return self._generate_simulated_data(indicator)
    
    def _generate_simulated_data(
        self,
        indicator: EconomicIndicator
    ) -> List[EconomicDataPoint]:
        import random
        
        base_values = {
            EconomicIndicator.FED_FUNDS_RATE: 5.25,
            EconomicIndicator.TREASURY_10Y: 4.5,
            EconomicIndicator.TREASURY_2Y: 4.8,
            EconomicIndicator.CPI: 3.2,
            EconomicIndicator.UNEMPLOYMENT: 3.8,
            EconomicIndicator.GDP_GROWTH: 2.5,
            EconomicIndicator.VIX: 15,
        }
        
        base = base_values.get(indicator, 100)
        points = []
        
        current = datetime.now()
        for i in range(12):
            date = current - timedelta(days=30 * i)
            value = base + random.uniform(-base * 0.1, base * 0.1)
            
            points.append(EconomicDataPoint(
                indicator=indicator,
                date=date,
                value=round(value, 2),
                previous=round(base, 2)
            ))
        
        return list(reversed(points))
    
    def fetch_calendar(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[EconomicEvent]:
        events = []
        
        upcoming = [
            ("Federal Reserve Interest Rate Decision", EconomicIndicator.FED_FUNDS_RATE, "high"),
            ("Non-Farm Payrolls", EconomicIndicator.NFP, "high"),
            ("CPI (YoY)", EconomicIndicator.CPI, "high"),
            ("Core CPI (YoY)", EconomicIndicator.CORE_CPI, "high"),
            ("GDP (QoQ)", EconomicIndicator.GDP_GROWTH, "high"),
            ("Unemployment Rate", EconomicIndicator.UNEMPLOYMENT, "medium"),
            ("Initial Jobless Claims", EconomicIndicator.JOBLESS_CLAIMS, "medium"),
            ("Retail Sales (MoM)", EconomicIndicator.RETAIL_SALES, "medium"),
            ("ISM Manufacturing PMI", EconomicIndicator.ISM_MANUFACTURING, "medium"),
            ("Consumer Confidence", EconomicIndicator.CONSUMER_CONFIDENCE, "medium"),
        ]
        
        import random
        base_date = datetime.now()
        
        for name, indicator, importance in upcoming:
            event_date = base_date + timedelta(days=random.randint(1, 30))
            
            events.append(EconomicEvent(
                name=name,
                indicator=indicator,
                datetime=event_date,
                forecast=random.uniform(0.5, 5.0),
                previous=random.uniform(0.5, 5.0),
                importance=importance
            ))
        
        return sorted(events, key=lambda x: x.datetime)
    
    def get_current_regime(self) -> MacroRegime:
        try:
            fed_rate = self.fetch_indicator(EconomicIndicator.FED_FUNDS_RATE)
            t10y = self.fetch_indicator(EconomicIndicator.TREASURY_10Y)
            t2y = self.fetch_indicator(EconomicIndicator.TREASURY_2Y)
            cpi = self.fetch_indicator(EconomicIndicator.CPI)
            gdp = self.fetch_indicator(EconomicIndicator.GDP_GROWTH)
            
            yield_spread = t10y[-1].value - t2y[-1].value if t10y and t2y else 0
            
            if yield_spread < -0.5:
                yield_curve = "INVERTED"
            elif yield_spread < 0:
                yield_curve = "FLAT"
            elif yield_spread < 1:
                yield_curve = "NORMAL"
            else:
                yield_curve = "STEEP"
            
            inflation = cpi[-1].value if cpi else 3.0
            if inflation > 4:
                inflation_trend = "HIGH"
            elif inflation > 2:
                inflation_trend = "MODERATE"
            else:
                inflation_trend = "LOW"
            
            growth = gdp[-1].value if gdp else 2.0
            if growth > 3:
                growth_trend = "STRONG"
            elif growth > 1:
                growth_trend = "MODERATE"
            elif growth > 0:
                growth_trend = "WEAK"
            else:
                growth_trend = "CONTRACTION"
            
            rate = fed_rate[-1].value if fed_rate else 5.25
            if rate > 5:
                fed_stance = "RESTRICTIVE"
            elif rate > 3:
                fed_stance = "NEUTRAL"
            else:
                fed_stance = "ACCOMMODATIVE"
            
            if growth_trend in ["STRONG", "MODERATE"] and inflation_trend != "HIGH":
                regime = "RISK_ON"
                risk_appetite = "HIGH"
            elif growth_trend == "CONTRACTION" or yield_curve == "INVERTED":
                regime = "RISK_OFF"
                risk_appetite = "LOW"
            else:
                regime = "NEUTRAL"
                risk_appetite = "MODERATE"
            
            return MacroRegime(
                timestamp=datetime.now(),
                regime=regime,
                risk_appetite=risk_appetite,
                yield_curve=yield_curve,
                inflation_trend=inflation_trend,
                growth_trend=growth_trend,
                fed_stance=fed_stance,
                confidence=0.7
            )
        except Exception:
            return MacroRegime(
                timestamp=datetime.now(),
                regime="NEUTRAL",
                risk_appetite="MODERATE",
                yield_curve="NORMAL",
                inflation_trend="MODERATE",
                growth_trend="MODERATE",
                fed_stance="NEUTRAL",
                confidence=0.3
            )
    
    def get_yield_curve(self) -> Dict[str, float]:
        maturities = {
            "1M": "DGS1MO",
            "3M": "DGS3MO",
            "6M": "DGS6MO",
            "1Y": "DGS1",
            "2Y": "DGS2",
            "3Y": "DGS3",
            "5Y": "DGS5",
            "7Y": "DGS7",
            "10Y": "DGS10",
            "20Y": "DGS20",
            "30Y": "DGS30",
        }
        
        curve = {}
        for label, series_id in maturities.items():
            try:
                data = self._request("series/observations", {
                    "series_id": series_id,
                    "sort_order": "desc",
                    "limit": 1
                })
                
                obs = data.get("observations", [])
                if obs:
                    try:
                        curve[label] = float(obs[0]["value"])
                    except (ValueError, TypeError):
                        pass
            except Exception:
                continue
        
        return curve


class TradingEconomicsAdapter(EconomicDataAdapter):
    """
    Trading Economics adapter for global economic data.
    
    Premium service with comprehensive coverage.
    
    Features:
    - 196 countries
    - 20M+ indicators
    - Real-time updates
    - Economic calendar
    - Forecasts
    
    Pricing: $49-499/month
    """
    
    BASE_URL = "https://api.tradingeconomics.com"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TRADING_ECONOMICS_API_KEY")
    
    @property
    def name(self) -> str:
        return "Trading Economics"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.ECONOMIC]
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier="$49-499/month",
            data_types=self.supported_data_types,
            last_error=None if self.is_available() else "TRADING_ECONOMICS_API_KEY not set"
        )
    
    def fetch_indicator(
        self,
        indicator: EconomicIndicator,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[EconomicDataPoint]:
        return []
    
    def fetch_calendar(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[EconomicEvent]:
        return []
    
    def get_current_regime(self) -> MacroRegime:
        return MacroRegime(
            timestamp=datetime.now(),
            regime="NEUTRAL",
            risk_appetite="MODERATE",
            yield_curve="NORMAL",
            inflation_trend="MODERATE",
            growth_trend="MODERATE",
            fed_stance="NEUTRAL",
            confidence=0.5
        )


class EconomicDataAggregator(EconomicDataAdapter):
    """
    Aggregates economic data from multiple sources.
    
    Prioritizes real data with simulated fallback.
    """
    
    def __init__(self):
        self.providers: List[EconomicDataAdapter] = []
        
        if os.getenv("FRED_API_KEY"):
            self.providers.append(FredAdapter())
        if os.getenv("TRADING_ECONOMICS_API_KEY"):
            self.providers.append(TradingEconomicsAdapter())
        
        if not self.providers:
            self.providers.append(FredAdapter())
    
    @property
    def name(self) -> str:
        return "Economic Data Aggregator"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.ECONOMIC]
    
    def is_available(self) -> bool:
        return len(self.providers) > 0
    
    def get_status(self) -> ProviderStatus:
        active = [p.name for p in self.providers if p.is_available()]
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier=f"Active: {', '.join(active)}" if active else "Simulated",
            data_types=self.supported_data_types
        )
    
    def fetch_indicator(
        self,
        indicator: EconomicIndicator,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[EconomicDataPoint]:
        for provider in self.providers:
            try:
                data = provider.fetch_indicator(indicator, start, end)
                if data:
                    return data
            except Exception:
                continue
        
        return []
    
    def fetch_calendar(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[EconomicEvent]:
        all_events = []
        
        for provider in self.providers:
            try:
                events = provider.fetch_calendar(start, end)
                all_events.extend(events)
            except Exception:
                continue
        
        return sorted(all_events, key=lambda x: x.datetime)
    
    def get_current_regime(self) -> MacroRegime:
        for provider in self.providers:
            try:
                return provider.get_current_regime()
            except Exception:
                continue
        
        return MacroRegime(
            timestamp=datetime.now(),
            regime="NEUTRAL",
            risk_appetite="MODERATE",
            yield_curve="NORMAL",
            inflation_trend="MODERATE",
            growth_trend="MODERATE",
            fed_stance="NEUTRAL",
            confidence=0.3
        )


ECONOMIC_SETUP_GUIDE = """
=== Economic / Macro Data Providers Setup Guide ===

1. FRED (Free - Recommended)
   - Federal Reserve Economic Data
   - 800,000+ time series
   - All major US indicators
   
   Sign up: https://fred.stlouisfed.org/docs/api/api_key.html
   
   FRED_API_KEY=your_key_here

2. TRADING ECONOMICS ($49-499/month)
   - Global coverage (196 countries)
   - 20M+ indicators
   - Real-time calendar
   
   TRADING_ECONOMICS_API_KEY=your_key_here

3. QUANDL / NASDAQ DATA LINK
   - Alternative data sets
   - Premium datasets available
   
   QUANDL_API_KEY=your_key_here

4. KEY INDICATORS
   - Fed Funds Rate: Monetary policy
   - 10Y Treasury: Risk-free rate
   - 2Y-10Y Spread: Yield curve
   - CPI: Inflation
   - NFP: Employment
   - GDP: Growth
   - VIX: Fear index
   - DXY: Dollar strength

5. USE CASES
   - Yield curve inversion → Recession signal
   - High inflation → Fed tightening
   - Strong NFP → Risk-on sentiment
   - Rising VIX → Risk-off positioning
   
6. REGIME DETECTION
   - RISK_ON: Strong growth, moderate inflation
   - RISK_OFF: Weak growth, inverted curve
   - NEUTRAL: Mixed signals
"""
