"""
QuantraCore Apex Integrations Module.

Provides third-party service integrations including Google Docs.
"""

from src.quantracore_apex.integrations.google_docs import (
    GoogleDocsClient,
    google_docs_client,
    ResearchReportGenerator,
    research_report_generator,
    TradeJournal,
    trade_journal,
    InvestorUpdates,
    investor_updates,
    NotesImporter,
    notes_importer,
    DocSync,
    doc_sync,
)

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
