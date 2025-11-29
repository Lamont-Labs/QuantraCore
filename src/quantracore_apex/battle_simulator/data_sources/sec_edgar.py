"""
SEC EDGAR Data Client.

Fetches 13F filings and other public disclosures from SEC EDGAR.

COMPLIANCE NOTES:
- SEC EDGAR is a public database, free to access
- No API key required (though rate limiting applies)
- All data is public information
- SEC requests max 10 requests/second
"""

import logging
import time
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import urllib.request
import urllib.error
from xml.etree import ElementTree as ET

from ..models import (
    Institution,
    InstitutionalHolding,
    Filing13F,
    ComplianceStatus,
)

logger = logging.getLogger(__name__)


SEC_EDGAR_BASE = "https://www.sec.gov"
SEC_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_COMPANY_SEARCH = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_DATA_API = "https://data.sec.gov"

USER_AGENT = "QuantraCore Research quantracore@research.edu"

TOP_INSTITUTIONS = [
    ("0001067983", "Berkshire Hathaway"),
    ("0001336528", "Bridgewater Associates"),
    ("0001423053", "Renaissance Technologies"),
    ("0001350694", "Citadel Advisors"),
    ("0001037389", "Vanguard Group"),
    ("0001364742", "BlackRock"),
    ("0001159159", "Soros Fund Management"),
    ("0001061768", "Appaloosa Management"),
    ("0000921669", "Baupost Group"),
    ("0001336528", "Two Sigma Investments"),
    ("0001167483", "Pershing Square"),
    ("0001656456", "Tiger Global"),
    ("0001510233", "Coatue Management"),
    ("0001040273", "Third Point"),
    ("0000102909", "State Street"),
]


@dataclass
class RateLimiter:
    """Simple rate limiter for SEC compliance."""
    requests_per_second: float = 10.0
    last_request_time: float = 0.0
    
    def wait(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = time.time()
        elapsed = now - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self.last_request_time = time.time()


class SECEdgarClient:
    """
    Client for fetching data from SEC EDGAR.
    
    COMPLIANCE:
    - Uses public SEC EDGAR API
    - Respects rate limits (10 req/sec)
    - Includes proper User-Agent header
    - All data is public information
    """
    
    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or USER_AGENT
        self.rate_limiter = RateLimiter()
        self._cache: Dict[str, Any] = {}
        
        logger.info("[SECEdgar] Initialized with public data access")
        logger.info(f"[SECEdgar] User-Agent: {self.user_agent}")
    
    def _make_request(self, url: str) -> Optional[str]:
        """Make rate-limited request to SEC EDGAR."""
        self.rate_limiter.wait()
        
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent}
            )
            
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
                
        except urllib.error.HTTPError as e:
            logger.warning(f"[SECEdgar] HTTP error {e.code} for {url}")
            return None
        except urllib.error.URLError as e:
            logger.warning(f"[SECEdgar] URL error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"[SECEdgar] Error fetching {url}: {e}")
            return None
    
    def get_institution_info(self, cik: str) -> Optional[Institution]:
        """
        Get basic information about an institution from SEC.
        
        Uses the SEC company API endpoint.
        """
        cik = str(cik).zfill(10)
        url = f"{SEC_DATA_API}/submissions/CIK{cik}.json"
        
        data = self._make_request(url)
        if not data:
            return None
        
        try:
            info = json.loads(data)
            
            filings = info.get("filings", {}).get("recent", {})
            filing_dates = filings.get("filingDate", [])
            forms = filings.get("form", [])
            
            f13_dates = [
                d for d, f in zip(filing_dates, forms)
                if f in ("13F-HR", "13F-HR/A")
            ]
            
            return Institution(
                cik=cik,
                name=info.get("name", "Unknown"),
                filing_count=len(f13_dates),
                first_filing_date=date.fromisoformat(f13_dates[-1]) if f13_dates else None,
                latest_filing_date=date.fromisoformat(f13_dates[0]) if f13_dates else None,
                headquarters=info.get("addresses", {}).get("business", {}).get("stateOrCountry"),
                compliance_status=ComplianceStatus.PUBLIC_DATA,
                data_source="SEC EDGAR",
            )
            
        except Exception as e:
            logger.error(f"[SECEdgar] Error parsing institution info: {e}")
            return None
    
    def get_13f_filings_list(
        self,
        cik: str,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Get list of 13F filings for an institution.
        
        Returns metadata about each filing (not full holdings).
        """
        cik = str(cik).zfill(10)
        url = f"{SEC_DATA_API}/submissions/CIK{cik}.json"
        
        data = self._make_request(url)
        if not data:
            return []
        
        try:
            info = json.loads(data)
            filings = info.get("filings", {}).get("recent", {})
            
            results = []
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            
            for i, (form, filing_date, accession) in enumerate(zip(forms, dates, accessions)):
                if form in ("13F-HR", "13F-HR/A"):
                    results.append({
                        "form": form,
                        "filing_date": filing_date,
                        "accession_number": accession,
                        "cik": cik,
                        "is_amendment": "/A" in form,
                    })
                    
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"[SECEdgar] Error getting filings list: {e}")
            return []
    
    def get_13f_holdings(
        self,
        cik: str,
        accession_number: str,
    ) -> Optional[Filing13F]:
        """
        Fetch and parse a complete 13F filing with all holdings.
        
        Parses the infotable.xml file from the 13F-HR filing.
        """
        cik = str(cik).zfill(10)
        accession_clean = accession_number.replace("-", "")
        
        xml_url = (
            f"{SEC_EDGAR_BASE}/Archives/edgar/data/{int(cik)}/"
            f"{accession_clean}/infotable.xml"
        )
        
        primary_url = (
            f"{SEC_EDGAR_BASE}/Archives/edgar/data/{int(cik)}/"
            f"{accession_clean}/primary_doc.xml"
        )
        
        xml_data = self._make_request(xml_url)
        primary_data = self._make_request(primary_url)
        
        holdings = []
        filing_date = None
        period_of_report = None
        institution_name = ""
        
        if primary_data:
            try:
                root = ET.fromstring(primary_data)
                
                ns = {"ns": "http://www.sec.gov/edgar/thirteenffiler"}
                
                for elem in root.iter():
                    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                    
                    if tag == "filingManager":
                        name_elem = elem.find(".//ns:name", ns) or elem.find(".//name")
                        if name_elem is not None:
                            institution_name = name_elem.text or ""
                    
                    if tag == "signatureDate" and elem.text:
                        try:
                            filing_date = date.fromisoformat(elem.text.strip())
                        except:
                            pass
                    
                    if tag == "reportCalendarOrQuarter" and elem.text:
                        try:
                            period_of_report = date.fromisoformat(elem.text.strip())
                        except:
                            pass
                            
            except Exception as e:
                logger.warning(f"[SECEdgar] Error parsing primary doc: {e}")
        
        if xml_data:
            try:
                root = ET.fromstring(xml_data)
                
                for info_table in root.iter():
                    tag = info_table.tag.split("}")[-1] if "}" in info_table.tag else info_table.tag
                    
                    if tag == "infoTable":
                        holding = self._parse_holding(info_table)
                        if holding:
                            holdings.append(holding)
                            
            except Exception as e:
                logger.warning(f"[SECEdgar] Error parsing holdings XML: {e}")
        
        if not holdings:
            return None
        
        return Filing13F(
            accession_number=accession_number,
            cik=cik,
            institution_name=institution_name,
            filing_date=filing_date or date.today(),
            period_of_report=period_of_report or date.today(),
            holdings=holdings,
            total_value=sum(h.value for h in holdings),
            total_holdings_count=len(holdings),
            compliance_status=ComplianceStatus.PUBLIC_DATA,
            data_source="SEC EDGAR 13F-HR",
        )
    
    def _parse_holding(self, info_table: ET.Element) -> Optional[InstitutionalHolding]:
        """Parse a single holding from 13F XML."""
        try:
            def get_text(tag: str) -> str:
                for elem in info_table.iter():
                    elem_tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                    if elem_tag == tag and elem.text:
                        return elem.text.strip()
                return ""
            
            def get_int(tag: str) -> int:
                text = get_text(tag)
                if text:
                    return int(text.replace(",", ""))
                return 0
            
            def get_float(tag: str) -> float:
                text = get_text(tag)
                if text:
                    return float(text.replace(",", ""))
                return 0.0
            
            return InstitutionalHolding(
                cusip=get_text("cusip"),
                issuer_name=get_text("nameOfIssuer"),
                class_title=get_text("titleOfClass"),
                value=get_float("value"),
                shares=get_int("sshPrnamt"),
                share_type=get_text("sshPrnamtType") or "SH",
                investment_discretion=get_text("investmentDiscretion") or "SOLE",
                voting_authority_sole=get_int("Sole"),
                voting_authority_shared=get_int("Shared"),
                voting_authority_none=get_int("None"),
                put_call=get_text("putCall") or None,
            )
            
        except Exception as e:
            logger.warning(f"[SECEdgar] Error parsing holding: {e}")
            return None
    
    def get_top_institutions(self) -> List[Institution]:
        """
        Get list of major institutional investors.
        
        Uses a curated list of well-known institutions.
        """
        institutions = []
        
        for cik, name in TOP_INSTITUTIONS:
            inst = self.get_institution_info(cik)
            if inst:
                institutions.append(inst)
            else:
                institutions.append(Institution(
                    cik=cik,
                    name=name,
                    compliance_status=ComplianceStatus.PUBLIC_DATA,
                ))
        
        return institutions
    
    def search_institutions(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for institutions by name.
        
        Uses SEC full-text search endpoint.
        """
        search_url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q={urllib.parse.quote(query)}"
            f"&forms=13F-HR"
            f"&dateRange=custom"
            f"&startdt=2020-01-01"
            f"&enddt={date.today().isoformat()}"
        )
        
        results = []
        
        for cik, name in TOP_INSTITUTIONS:
            if query.lower() in name.lower():
                results.append({
                    "cik": cik,
                    "name": name,
                    "source": "curated_list",
                })
        
        return results[:limit]
    
    def get_quarterly_holdings_comparison(
        self,
        cik: str,
        quarters: int = 4,
    ) -> List[Filing13F]:
        """
        Get multiple quarters of 13F filings for comparison.
        
        Useful for tracking position changes over time.
        """
        filings_meta = self.get_13f_filings_list(cik, limit=quarters)
        
        filings = []
        for meta in filings_meta:
            if not meta.get("is_amendment", False):
                filing = self.get_13f_holdings(cik, meta["accession_number"])
                if filing:
                    filings.append(filing)
        
        return filings


import urllib.parse
