"""
Trade Journal for QuantraCore Apex.

Maintains a persistent research journal in Google Docs, logging:
- Signals generated and their outcomes
- Protocol activations
- Omega directive alerts
- Backtesting results
- Research notes and observations
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from enum import Enum

from src.quantracore_apex.integrations.google_docs.client import google_docs_client

logger = logging.getLogger(__name__)


class JournalEntryType(str, Enum):
    """Types of journal entries."""
    SIGNAL = "signal"
    PROTOCOL = "protocol"
    OMEGA_ALERT = "omega_alert"
    BACKTEST = "backtest"
    OBSERVATION = "observation"
    RESEARCH_NOTE = "research_note"
    SCAN_RESULT = "scan_result"


class JournalEntry(BaseModel):
    """A journal entry record."""
    entry_type: JournalEntryType
    symbol: Optional[str] = None
    title: str
    content: str
    metadata: Dict[str, Any] = {}
    timestamp: datetime = None
    
    def __init__(self, **data):
        if data.get('timestamp') is None:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


class TradeJournal:
    """
    Persistent trade journal stored in Google Docs.
    
    Features:
    - Automatic daily journal creation
    - Structured entry formatting
    - Signal tracking with outcomes
    - Protocol activation logging
    - Omega directive alert tracking
    - Research observation notes
    """
    
    JOURNAL_PREFIX = "QuantraCore Trade Journal"
    
    def __init__(self):
        self.client = google_docs_client
        self._current_journal_id: Optional[str] = None
        self._current_journal_date: Optional[str] = None
    
    def _get_journal_title(self, date: Optional[datetime] = None) -> str:
        """Get journal title for a specific date."""
        date = date or datetime.utcnow()
        return f"{self.JOURNAL_PREFIX} - {date.strftime('%Y-%m-%d')}"
    
    async def _get_or_create_daily_journal(self, date: Optional[datetime] = None) -> str:
        """
        Get or create the journal document for a specific date.
        
        Args:
            date: Date for the journal (defaults to today)
            
        Returns:
            str: Document ID
        """
        date = date or datetime.utcnow()
        date_str = date.strftime('%Y-%m-%d')
        
        if self._current_journal_date == date_str and self._current_journal_id:
            return self._current_journal_id
        
        title = self._get_journal_title(date)
        
        try:
            docs = await self.client.search_documents(title, max_results=1)
            for doc in docs:
                if doc.get('name') == title:
                    self._current_journal_id = doc['id']
                    self._current_journal_date = date_str
                    return doc['id']
        except Exception as e:
            logger.warning(f"Error searching for journal: {e}")
        
        document = await self.client.create_document(title)
        document_id = document.get('documentId')
        
        header = f"""
================================================================================
                        QUANTRACORE APEX TRADE JOURNAL
================================================================================

Date: {date.strftime('%A, %B %d, %Y')}
Engine: v9.0-A Institutional Hardening
Mode: RESEARCH ONLY

This journal tracks all signals, protocol activations, and research observations
for the trading session. All entries are timestamped and categorized.

DISCLAIMER: This is a research journal. No entries constitute trading advice.

================================================================================

"""
        await self.client.insert_text(document_id, header)
        
        self._current_journal_id = document_id
        self._current_journal_date = date_str
        
        logger.info(f"Created new trade journal: {document_id}")
        return document_id
    
    def _format_entry(self, entry: JournalEntry) -> str:
        """Format a journal entry for insertion."""
        timestamp = entry.timestamp.strftime('%H:%M:%S UTC')
        type_label = entry.entry_type.value.upper().replace('_', ' ')
        
        formatted = f"\n[{timestamp}] {type_label}"
        if entry.symbol:
            formatted += f" | {entry.symbol}"
        formatted += f"\n{'-' * 60}\n"
        formatted += f"{entry.title}\n\n"
        formatted += f"{entry.content}\n"
        
        if entry.metadata:
            formatted += "\nMetadata:\n"
            for key, value in entry.metadata.items():
                formatted += f"  {key}: {value}\n"
        
        formatted += "\n"
        return formatted
    
    async def log_entry(self, entry: JournalEntry) -> Dict[str, Any]:
        """
        Log an entry to the daily journal.
        
        Args:
            entry: Journal entry to log
            
        Returns:
            dict: Entry logging result
        """
        journal_id = await self._get_or_create_daily_journal()
        formatted = self._format_entry(entry)
        
        await self.client.append_text(journal_id, formatted)
        
        logger.info(f"Logged {entry.entry_type.value} entry to journal")
        
        return {
            "journal_id": journal_id,
            "entry_type": entry.entry_type.value,
            "symbol": entry.symbol,
            "timestamp": entry.timestamp.isoformat(),
            "logged": True
        }
    
    async def log_signal(
        self,
        symbol: str,
        signal_data: Dict[str, Any],
        scan_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a trading signal to the journal.
        
        Args:
            symbol: Trading symbol
            signal_data: Signal generation result
            scan_result: Optional associated scan result
            
        Returns:
            dict: Logging result
        """
        signal = signal_data.get('signal', {})
        
        content = f"""Signal Type: {signal.get('signal_type', 'N/A')}
Direction: {signal.get('direction', 'N/A')}
Strength: {signal.get('strength', 'N/A')}
Entry Price: {signal.get('entry_price', 'N/A')}
Stop Loss: {signal.get('stop_loss', 'N/A')}
Take Profit: {signal.get('take_profit', 'N/A')}
Risk-Reward: {signal.get('risk_reward_ratio', 'N/A')}"""
        
        metadata = {
            "risk_tier": signal_data.get('risk_tier', 'N/A'),
        }
        
        if scan_result:
            metadata["quantrascore"] = scan_result.get('quantrascore', 'N/A')
            metadata["regime"] = scan_result.get('regime', 'N/A')
            metadata["window_hash"] = scan_result.get('window_hash', 'N/A')
        
        entry = JournalEntry(
            entry_type=JournalEntryType.SIGNAL,
            symbol=symbol,
            title=f"Signal Generated: {signal.get('direction', 'N/A')} {signal.get('signal_type', 'N/A')}",
            content=content,
            metadata=metadata
        )
        
        return await self.log_entry(entry)
    
    async def log_protocol_activation(
        self,
        symbol: str,
        protocols_fired: List[str],
        scan_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Log protocol activations to the journal.
        
        Args:
            symbol: Trading symbol
            protocols_fired: List of fired protocol IDs
            scan_result: Scan result data
            
        Returns:
            dict: Logging result
        """
        content = f"""Protocols Fired: {len(protocols_fired)} of 145

Active Protocols:
{chr(10).join(f'  - {p}' for p in protocols_fired[:20])}
{'... and {} more'.format(len(protocols_fired) - 20) if len(protocols_fired) > 20 else ''}

QuantraScore: {scan_result.get('quantrascore', 'N/A')}/100
Regime: {scan_result.get('regime', 'N/A')}
Verdict: {scan_result.get('verdict_action', 'N/A')}"""
        
        entry = JournalEntry(
            entry_type=JournalEntryType.PROTOCOL,
            symbol=symbol,
            title=f"Protocol Analysis Complete - {len(protocols_fired)} Fired",
            content=content,
            metadata={
                "total_fired": len(protocols_fired),
                "quantrascore": scan_result.get('quantrascore', 0),
                "window_hash": scan_result.get('window_hash', 'N/A')
            }
        )
        
        return await self.log_entry(entry)
    
    async def log_omega_alert(
        self,
        symbol: str,
        omega_alerts: List[str],
        scan_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Log Omega directive alerts to the journal.
        
        Args:
            symbol: Trading symbol
            omega_alerts: List of active Omega directives
            scan_result: Scan result data
            
        Returns:
            dict: Logging result
        """
        if not omega_alerts:
            return {"logged": False, "reason": "No omega alerts to log"}
        
        content = f"""OMEGA DIRECTIVE ALERTS ACTIVE

Active Directives:
{chr(10).join(f'  - {alert}' for alert in omega_alerts)}

These safety overrides have been triggered based on current market conditions.
Review carefully before any research decisions.

QuantraScore: {scan_result.get('quantrascore', 'N/A')}/100
Regime: {scan_result.get('regime', 'N/A')}"""
        
        entry = JournalEntry(
            entry_type=JournalEntryType.OMEGA_ALERT,
            symbol=symbol,
            title=f"OMEGA ALERT: {len(omega_alerts)} Directives Active",
            content=content,
            metadata={
                "alert_count": len(omega_alerts),
                "alerts": omega_alerts,
                "window_hash": scan_result.get('window_hash', 'N/A')
            }
        )
        
        return await self.log_entry(entry)
    
    async def log_scan_result(
        self,
        symbol: str,
        scan_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Log a scan result summary to the journal.
        
        Args:
            symbol: Trading symbol
            scan_result: Scan result data
            
        Returns:
            dict: Logging result
        """
        content = f"""QuantraScore: {scan_result.get('quantrascore', 'N/A')}/100 ({scan_result.get('score_bucket', 'N/A')})
Regime: {scan_result.get('regime', 'N/A')}
Risk Tier: {scan_result.get('risk_tier', 'N/A')}

States:
  Entropy: {scan_result.get('entropy_state', 'N/A')}
  Suppression: {scan_result.get('suppression_state', 'N/A')}
  Drift: {scan_result.get('drift_state', 'N/A')}

Verdict: {scan_result.get('verdict_action', 'N/A')} (Confidence: {scan_result.get('verdict_confidence', 0):.1%})
Protocols Fired: {scan_result.get('protocol_fired_count', 0)}
Omega Alerts: {len(scan_result.get('omega_alerts', []))}"""
        
        entry = JournalEntry(
            entry_type=JournalEntryType.SCAN_RESULT,
            symbol=symbol,
            title=f"Scan Complete - Score {scan_result.get('quantrascore', 0):.1f}",
            content=content,
            metadata={
                "quantrascore": scan_result.get('quantrascore', 0),
                "regime": scan_result.get('regime', 'N/A'),
                "window_hash": scan_result.get('window_hash', 'N/A')
            }
        )
        
        return await self.log_entry(entry)
    
    async def log_research_note(
        self,
        title: str,
        content: str,
        symbol: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a custom research note to the journal.
        
        Args:
            title: Note title
            content: Note content
            symbol: Optional associated symbol
            metadata: Optional additional metadata
            
        Returns:
            dict: Logging result
        """
        entry = JournalEntry(
            entry_type=JournalEntryType.RESEARCH_NOTE,
            symbol=symbol,
            title=title,
            content=content,
            metadata=metadata or {}
        )
        
        return await self.log_entry(entry)
    
    async def log_observation(
        self,
        observation: str,
        symbol: Optional[str] = None,
        category: str = "General"
    ) -> Dict[str, Any]:
        """
        Log a quick observation to the journal.
        
        Args:
            observation: Observation text
            symbol: Optional associated symbol
            category: Observation category
            
        Returns:
            dict: Logging result
        """
        entry = JournalEntry(
            entry_type=JournalEntryType.OBSERVATION,
            symbol=symbol,
            title=f"Observation: {category}",
            content=observation,
            metadata={"category": category}
        )
        
        return await self.log_entry(entry)
    
    async def get_journal_url(self, date: Optional[datetime] = None) -> str:
        """
        Get the URL for a journal document.
        
        Args:
            date: Date of the journal (defaults to today)
            
        Returns:
            str: Google Docs URL
        """
        journal_id = await self._get_or_create_daily_journal(date)
        return f"https://docs.google.com/document/d/{journal_id}/edit"
    
    async def list_journals(self, max_results: int = 30) -> List[Dict[str, Any]]:
        """
        List all trade journals.
        
        Args:
            max_results: Maximum journals to return
            
        Returns:
            list: Journal documents metadata
        """
        docs = await self.client.search_documents(self.JOURNAL_PREFIX, max_results)
        return [
            {
                "id": doc["id"],
                "name": doc["name"],
                "modified": doc.get("modifiedTime"),
                "url": f"https://docs.google.com/document/d/{doc['id']}/edit"
            }
            for doc in docs
        ]


trade_journal = TradeJournal()
