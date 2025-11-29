"""
Google Docs Integration for QuantraCore Apex.

Provides comprehensive document management capabilities:
- Research Report Generation
- Trade Journal Logging
- Investor Updates
- Notes Import
- Documentation Sync
"""

from src.quantracore_apex.integrations.google_docs.client import GoogleDocsClient, google_docs_client
from src.quantracore_apex.integrations.google_docs.research_report import ResearchReportGenerator, research_report_generator
from src.quantracore_apex.integrations.google_docs.trade_journal import TradeJournal, trade_journal
from src.quantracore_apex.integrations.google_docs.investor_updates import InvestorUpdates, investor_updates
from src.quantracore_apex.integrations.google_docs.notes_importer import NotesImporter, notes_importer
from src.quantracore_apex.integrations.google_docs.doc_sync import DocSync, doc_sync

__all__ = [
    "GoogleDocsClient",
    "google_docs_client",
    "ResearchReportGenerator",
    "research_report_generator",
    "TradeJournal",
    "trade_journal",
    "InvestorUpdates",
    "investor_updates",
    "NotesImporter",
    "notes_importer",
    "DocSync",
    "doc_sync",
]
