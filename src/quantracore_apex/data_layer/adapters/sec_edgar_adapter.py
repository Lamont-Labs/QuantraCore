"""
SEC EDGAR Adapter for QuantraCore Apex

Provides direct access to SEC filings:
- Form 4: Insider trading (buys/sells by executives)
- 13F: Institutional holdings (hedge fund positions)
- 8-K: Material events (earnings, acquisitions)

NO API KEY REQUIRED - Free government data
Rate limit: 10 requests/second (be respectful)

https://www.sec.gov/developer
"""

import logging
import time
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import httpx

logger = logging.getLogger(__name__)


@dataclass
class InsiderTransaction:
    """Represents an insider trading transaction from Form 4."""
    symbol: str
    company_name: str
    insider_name: str
    insider_title: str
    transaction_date: datetime
    transaction_type: str  # "BUY" or "SELL"
    shares: float
    price_per_share: float
    total_value: float
    shares_owned_after: float
    filing_date: datetime
    form_type: str
    accession_number: str


@dataclass
class InstitutionalHolding:
    """Represents an institutional holding from 13F."""
    institution_name: str
    symbol: str
    company_name: str
    shares: int
    value: float  # In thousands
    share_change: int
    percent_change: float
    filing_date: datetime
    quarter: str


@dataclass
class MaterialEvent:
    """Represents a material event from 8-K filing."""
    symbol: str
    company_name: str
    event_type: str
    description: str
    filing_date: datetime
    accession_number: str


class SecEdgarAdapter:
    """
    SEC EDGAR API adapter for accessing government filings.
    
    Features:
    - Form 4 insider transactions (executive buys/sells)
    - 13F institutional holdings (hedge fund positions)
    - 8-K material events (earnings, acquisitions)
    - CIK lookup for ticker symbols
    
    Rate limits: 10 requests/second (SEC guideline)
    No API key required.
    """
    
    BASE_URL = "https://data.sec.gov"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
    
    HEADERS = {
        "User-Agent": "QuantraCore-Apex/9.0 (quantracore@trading.ai)",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate"
    }
    
    FORM_4_TRANSACTION_CODES = {
        "P": "BUY",
        "S": "SELL", 
        "A": "GRANT",
        "D": "DISPOSITION",
        "F": "TAX",
        "M": "EXERCISE",
        "C": "CONVERSION",
        "G": "GIFT",
        "J": "OTHER"
    }
    
    def __init__(self):
        self._last_request = 0
        self._rate_limit_delay = 0.15  # 10 requests/sec = 100ms, add buffer
        self._cik_cache: Dict[str, str] = {}
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 900  # 15 minutes
        
        logger.info("[SEC EDGAR] Adapter initialized (free, no API key required)")
    
    @property
    def name(self) -> str:
        return "sec_edgar"
    
    def is_available(self) -> bool:
        return True  # Always available, no API key needed
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "available": True,
            "rate_limit": "10 req/sec",
            "coverage": "Insider Filings (Form 4), Institutional Holdings (13F), Material Events (8-K)",
            "api_key_required": False
        }
    
    def _rate_limit_wait(self):
        """Respect SEC rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()
    
    def _request(self, url: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Make a rate-limited request to SEC EDGAR."""
        self._rate_limit_wait()
        
        try:
            with httpx.Client(timeout=30.0, headers=self.HEADERS) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"[SEC EDGAR] Resource not found: {url}")
                return None
            logger.warning(f"[SEC EDGAR] HTTP error: {e}")
            return None
        except Exception as e:
            logger.warning(f"[SEC EDGAR] Request error: {e}")
            return None
    
    def _get_cik(self, symbol: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a ticker symbol."""
        symbol = symbol.upper()
        
        if symbol in self._cik_cache:
            return self._cik_cache[symbol]
        
        try:
            self._rate_limit_wait()
            
            with httpx.Client(timeout=30.0, headers=self.HEADERS) as client:
                response = client.get(self.TICKERS_URL)
                response.raise_for_status()
                data = response.json()
            
            for entry in data.values():
                if entry.get("ticker", "").upper() == symbol:
                    cik = str(entry.get("cik_str", "")).zfill(10)
                    self._cik_cache[symbol] = cik
                    return cik
            
            logger.debug(f"[SEC EDGAR] CIK not found for {symbol}")
            return None
            
        except Exception as e:
            logger.warning(f"[SEC EDGAR] CIK lookup error: {e}")
            return None
    
    def get_insider_transactions(
        self, 
        symbol: str, 
        days: int = 90
    ) -> List[InsiderTransaction]:
        """
        Get recent insider transactions (Form 4 filings) for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back (default 90)
            
        Returns:
            List of InsiderTransaction objects
        """
        cache_key = f"insider_{symbol}_{days}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        cik = self._get_cik(symbol)
        if not cik:
            return []
        
        try:
            url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
            data = self._request(url, {})
            
            if not data:
                return []
            
            company_name = data.get("name", symbol)
            filings = data.get("filings", {}).get("recent", {})
            
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            
            transactions = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for i, form in enumerate(forms):
                if form not in ["4", "4/A"]:
                    continue
                    
                try:
                    filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
                    if filing_date < cutoff_date:
                        continue
                    
                    accession = accessions[i].replace("-", "")
                    
                    form4_data = self._parse_form4(cik, accession, symbol, company_name, filing_date)
                    transactions.extend(form4_data)
                    
                    if len(transactions) >= 50:
                        break
                        
                except (ValueError, IndexError) as e:
                    logger.debug(f"[SEC EDGAR] Parse error: {e}")
                    continue
            
            self._cache[cache_key] = (transactions, time.time())
            return transactions
            
        except Exception as e:
            logger.error(f"[SEC EDGAR] Insider transactions error: {e}")
            return []
    
    def _parse_form4(
        self, 
        cik: str, 
        accession: str, 
        symbol: str,
        company_name: str,
        filing_date: datetime
    ) -> List[InsiderTransaction]:
        """Parse Form 4 XML to extract transaction details."""
        transactions = []
        
        try:
            url = f"{self.BASE_URL}/Archives/edgar/data/{cik.lstrip('0')}/{accession}/primary_doc.xml"
            self._rate_limit_wait()
            
            with httpx.Client(timeout=30.0, headers=self.HEADERS) as client:
                response = client.get(url)
                if response.status_code != 200:
                    return transactions
                
                content = response.text
            
            insider_match = re.search(r"<rptOwnerName>([^<]+)</rptOwnerName>", content)
            insider_name = insider_match.group(1) if insider_match else "Unknown"
            
            title_match = re.search(r"<officerTitle>([^<]+)</officerTitle>", content)
            insider_title = title_match.group(1) if title_match else "Officer"
            
            trans_pattern = re.compile(
                r"<transactionCode>([^<]+)</transactionCode>.*?"
                r"<transactionDate>.*?<value>([^<]+)</value>.*?"
                r"<transactionShares>.*?<value>([^<]+)</value>.*?"
                r"<transactionPricePerShare>.*?<value>([^<]*)</value>.*?"
                r"<sharesOwnedFollowingTransaction>.*?<value>([^<]+)</value>",
                re.DOTALL
            )
            
            for match in trans_pattern.finditer(content):
                try:
                    code = match.group(1)
                    trans_date_str = match.group(2)
                    shares = float(match.group(3))
                    price_str = match.group(4)
                    shares_after = float(match.group(5))
                    
                    price = float(price_str) if price_str else 0.0
                    
                    trans_type = self.FORM_4_TRANSACTION_CODES.get(code, "OTHER")
                    
                    if trans_type not in ["BUY", "SELL"]:
                        continue
                    
                    try:
                        trans_date = datetime.strptime(trans_date_str, "%Y-%m-%d")
                    except ValueError:
                        trans_date = filing_date
                    
                    transaction = InsiderTransaction(
                        symbol=symbol,
                        company_name=company_name,
                        insider_name=insider_name,
                        insider_title=insider_title,
                        transaction_date=trans_date,
                        transaction_type=trans_type,
                        shares=shares,
                        price_per_share=price,
                        total_value=shares * price,
                        shares_owned_after=shares_after,
                        filing_date=filing_date,
                        form_type="4",
                        accession_number=accession
                    )
                    transactions.append(transaction)
                    
                except (ValueError, AttributeError) as e:
                    logger.debug(f"[SEC EDGAR] Transaction parse error: {e}")
                    continue
            
        except Exception as e:
            logger.debug(f"[SEC EDGAR] Form 4 parse error: {e}")
        
        return transactions
    
    def get_institutional_holdings(self, symbol: str) -> List[InstitutionalHolding]:
        """
        Get institutional holdings (13F filings) for a symbol.
        
        This shows what hedge funds and institutions own.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            List of InstitutionalHolding objects
        """
        cache_key = f"13f_{symbol}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        holdings = []
        
        try:
            url = f"https://efts.sec.gov/LATEST/search-index"
            params = {
                "q": f'"{symbol}"',
                "dateRange": "custom",
                "startdt": (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d"),
                "enddt": datetime.now().strftime("%Y-%m-%d"),
                "forms": "13F-HR",
                "from": "0",
                "size": "20"
            }
            
            data = self._request(url, params)
            
            if not data or "hits" not in data:
                self._cache[cache_key] = (holdings, time.time())
                return holdings
            
            for hit in data.get("hits", {}).get("hits", [])[:10]:
                try:
                    source = hit.get("_source", {})
                    
                    holding = InstitutionalHolding(
                        institution_name=source.get("display_names", ["Unknown"])[0],
                        symbol=symbol,
                        company_name=symbol,
                        shares=0,
                        value=0.0,
                        share_change=0,
                        percent_change=0.0,
                        filing_date=datetime.strptime(
                            source.get("file_date", datetime.now().strftime("%Y-%m-%d")),
                            "%Y-%m-%d"
                        ),
                        quarter=source.get("period_of_report", "Unknown")
                    )
                    holdings.append(holding)
                    
                except Exception as e:
                    logger.debug(f"[SEC EDGAR] 13F parse error: {e}")
                    continue
            
            self._cache[cache_key] = (holdings, time.time())
            return holdings
            
        except Exception as e:
            logger.error(f"[SEC EDGAR] Institutional holdings error: {e}")
            return []
    
    def get_insider_summary(self, symbol: str, days: int = 90) -> Dict[str, Any]:
        """
        Get a summary of insider activity for a symbol.
        
        Returns aggregated buy/sell activity useful for catalyst scoring.
        """
        transactions = self.get_insider_transactions(symbol, days)
        
        if not transactions:
            return {
                "symbol": symbol,
                "period_days": days,
                "total_transactions": 0,
                "buy_count": 0,
                "sell_count": 0,
                "net_shares": 0,
                "buy_value": 0.0,
                "sell_value": 0.0,
                "net_value": 0.0,
                "insider_sentiment": "NEUTRAL",
                "confidence": 0.0,
                "notable_insiders": [],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        buy_transactions = [t for t in transactions if t.transaction_type == "BUY"]
        sell_transactions = [t for t in transactions if t.transaction_type == "SELL"]
        
        buy_shares = sum(t.shares for t in buy_transactions)
        sell_shares = sum(t.shares for t in sell_transactions)
        buy_value = sum(t.total_value for t in buy_transactions)
        sell_value = sum(t.total_value for t in sell_transactions)
        
        net_shares = buy_shares - sell_shares
        net_value = buy_value - sell_value
        
        if net_value > 100000:
            sentiment = "STRONG_BUY"
            confidence = min(0.9, 0.5 + (net_value / 1000000) * 0.4)
        elif net_value > 10000:
            sentiment = "BUY"
            confidence = 0.6
        elif net_value < -100000:
            sentiment = "STRONG_SELL"
            confidence = min(0.9, 0.5 + (abs(net_value) / 1000000) * 0.4)
        elif net_value < -10000:
            sentiment = "SELL"
            confidence = 0.6
        else:
            sentiment = "NEUTRAL"
            confidence = 0.3
        
        notable_insiders = []
        for t in sorted(transactions, key=lambda x: x.total_value, reverse=True)[:5]:
            if t.total_value > 10000:
                notable_insiders.append({
                    "name": t.insider_name,
                    "title": t.insider_title,
                    "type": t.transaction_type,
                    "shares": t.shares,
                    "value": t.total_value,
                    "date": t.transaction_date.strftime("%Y-%m-%d")
                })
        
        return {
            "symbol": symbol,
            "period_days": days,
            "total_transactions": len(transactions),
            "buy_count": len(buy_transactions),
            "sell_count": len(sell_transactions),
            "net_shares": net_shares,
            "buy_value": round(buy_value, 2),
            "sell_value": round(sell_value, 2),
            "net_value": round(net_value, 2),
            "insider_sentiment": sentiment,
            "confidence": round(confidence, 2),
            "notable_insiders": notable_insiders,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_material_events(self, symbol: str, days: int = 30) -> List[MaterialEvent]:
        """
        Get material events (8-K filings) for a symbol.
        
        8-K filings report significant corporate events like:
        - Earnings announcements
        - Executive changes
        - Acquisitions/mergers
        - Bankruptcy
        """
        cache_key = f"8k_{symbol}_{days}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        cik = self._get_cik(symbol)
        if not cik:
            return []
        
        events = []
        
        try:
            url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
            data = self._request(url, {})
            
            if not data:
                return []
            
            company_name = data.get("name", symbol)
            filings = data.get("filings", {}).get("recent", {})
            
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            descriptions = filings.get("primaryDocDescription", [])
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for i, form in enumerate(forms):
                if not form.startswith("8-K"):
                    continue
                    
                try:
                    filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
                    if filing_date < cutoff_date:
                        continue
                    
                    event = MaterialEvent(
                        symbol=symbol,
                        company_name=company_name,
                        event_type="8-K",
                        description=descriptions[i] if i < len(descriptions) else "Material Event",
                        filing_date=filing_date,
                        accession_number=accessions[i]
                    )
                    events.append(event)
                    
                    if len(events) >= 20:
                        break
                        
                except (ValueError, IndexError):
                    continue
            
            self._cache[cache_key] = (events, time.time())
            return events
            
        except Exception as e:
            logger.error(f"[SEC EDGAR] Material events error: {e}")
            return []


_edgar_adapter: Optional[SecEdgarAdapter] = None


def get_sec_edgar_adapter() -> SecEdgarAdapter:
    """Get singleton SEC EDGAR adapter instance."""
    global _edgar_adapter
    if _edgar_adapter is None:
        _edgar_adapter = SecEdgarAdapter()
    return _edgar_adapter
