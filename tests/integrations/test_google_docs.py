"""
Tests for Google Docs Integration.

Tests the core functionality of the Google Docs integration module
including research reports, trade journal, investor updates, notes importer,
and documentation sync.

Note: These tests mock the Google API calls to avoid requiring actual
Google Docs access during testing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.quantracore_apex.integrations.google_docs.client import GoogleDocsClient
from src.quantracore_apex.integrations.google_docs.research_report import (
    ResearchReportGenerator,
    ResearchReportConfig,
    ReportSection,
)
from src.quantracore_apex.integrations.google_docs.trade_journal import (
    TradeJournal,
    JournalEntry,
    JournalEntryType,
)
from src.quantracore_apex.integrations.google_docs.investor_updates import (
    InvestorUpdates,
    InvestorUpdateConfig,
)
from src.quantracore_apex.integrations.google_docs.notes_importer import (
    NotesImporter,
    ImportedNote,
    Watchlist,
)
from src.quantracore_apex.integrations.google_docs.doc_sync import (
    DocSync,
    DocSyncConfig,
    SyncResult,
)


class TestGoogleDocsClient:
    """Tests for the core Google Docs client."""
    
    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = GoogleDocsClient()
        assert client._cached_credentials is None
    
    @pytest.mark.asyncio
    async def test_check_connection_without_env(self):
        """Test connection check fails gracefully without environment."""
        client = GoogleDocsClient()
        with patch.dict('os.environ', {}, clear=True):
            status = await client.check_connection()
            assert status["connected"] is False
            assert status["status"] == "error"
            assert "error" in status


class TestResearchReportGenerator:
    """Tests for the Research Report Generator."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = ResearchReportConfig()
        assert config.include_protocols is True
        assert config.include_omega_alerts is True
        assert config.include_risk_assessment is True
        assert config.include_monster_runner is True
        assert config.include_compliance_note is True
    
    def test_generator_initialization(self):
        """Test generator initializes with default config."""
        generator = ResearchReportGenerator()
        assert generator.config is not None
        assert generator.config.include_protocols is True
    
    def test_generator_with_custom_config(self):
        """Test generator accepts custom configuration."""
        config = ResearchReportConfig(
            include_protocols=False,
            include_risk_assessment=False,
        )
        generator = ResearchReportGenerator(config=config)
        assert generator.config.include_protocols is False
        assert generator.config.include_risk_assessment is False
    
    def test_format_header(self):
        """Test report header formatting."""
        generator = ResearchReportGenerator()
        header = generator._format_header("AAPL", datetime(2025, 11, 29, 12, 0))
        
        assert "QUANTRACORE APEX RESEARCH REPORT" in header
        assert "AAPL" in header
        assert "2025-11-29" in header
        assert "COMPLIANCE NOTICE" in header
    
    def test_format_executive_summary(self):
        """Test executive summary formatting."""
        generator = ResearchReportGenerator()
        scan_result = {
            "quantrascore": 75.5,
            "score_bucket": "A",
            "regime": "BULLISH",
            "risk_tier": "LOW",
            "verdict_action": "HOLD",
            "verdict_confidence": 0.85,
            "entropy_state": "STABLE",
            "suppression_state": "NONE",
            "drift_state": "NEUTRAL",
            "protocol_fired_count": 42,
            "omega_alerts": ["O5_RSIExtreme"],
        }
        
        summary = generator._format_executive_summary(scan_result)
        
        assert "75.5/100" in summary
        assert "BULLISH" in summary
        assert "LOW" in summary
        assert "HOLD" in summary
        assert "42 of 145" in summary
    
    def test_format_monster_runner(self):
        """Test MonsterRunner section formatting."""
        generator = ResearchReportGenerator()
        monster_data = {
            "runner_probability": 0.75,
            "runner_state": "PRIMED",
            "rare_event_class": "POTENTIAL_RUNNER",
            "metrics": {
                "compression_trace": 0.8,
                "entropy_floor": 0.2,
                "volume_pulse": 1.5,
                "range_contraction": 0.3,
                "primed_confidence": 0.85,
            }
        }
        
        section = generator._format_monster_runner(monster_data)
        
        assert "MONSTERRUNNER ANALYSIS" in section
        assert "75.0%" in section
        assert "PRIMED" in section
        assert "POTENTIAL_RUNNER" in section
    
    def test_format_footer(self):
        """Test report footer formatting."""
        generator = ResearchReportGenerator()
        footer = generator._format_footer()
        
        assert "DISCLAIMER" in footer
        assert "NOT TRADING ADVICE" in footer
        assert "RESEARCH ONLY" in footer
        assert "Lamont Labs" in footer


class TestTradeJournal:
    """Tests for the Trade Journal."""
    
    def test_journal_initialization(self):
        """Test journal initializes correctly."""
        journal = TradeJournal()
        assert journal._current_journal_id is None
        assert journal._current_journal_date is None
    
    def test_journal_title_generation(self):
        """Test journal title format."""
        journal = TradeJournal()
        date = datetime(2025, 11, 29)
        title = journal._get_journal_title(date)
        
        assert "QuantraCore Trade Journal" in title
        assert "2025-11-29" in title
    
    def test_journal_entry_creation(self):
        """Test journal entry model."""
        entry = JournalEntry(
            entry_type=JournalEntryType.SIGNAL,
            symbol="AAPL",
            title="Test Signal",
            content="Test content",
        )
        
        assert entry.entry_type == JournalEntryType.SIGNAL
        assert entry.symbol == "AAPL"
        assert entry.timestamp is not None
    
    def test_format_entry(self):
        """Test entry formatting."""
        journal = TradeJournal()
        entry = JournalEntry(
            entry_type=JournalEntryType.RESEARCH_NOTE,
            symbol="MSFT",
            title="Research Note",
            content="This is test content.",
            metadata={"key": "value"},
        )
        
        formatted = journal._format_entry(entry)
        
        assert "RESEARCH NOTE" in formatted
        assert "MSFT" in formatted
        assert "Research Note" in formatted
        assert "This is test content." in formatted
        assert "key: value" in formatted


class TestInvestorUpdates:
    """Tests for the Investor Updates module."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = InvestorUpdateConfig()
        assert config.include_performance_metrics is True
        assert config.include_compliance_score is True
        assert config.include_system_capabilities is True
        assert config.confidentiality_level == "standard"
    
    def test_updates_initialization(self):
        """Test module initializes correctly."""
        updates = InvestorUpdates()
        assert updates.config is not None
    
    def test_format_header(self):
        """Test investor update header."""
        updates = InvestorUpdates()
        header = updates._format_header("November 2025", "Monthly Update")
        
        assert "LAMONT LABS" in header
        assert "QUANTRACORE APEX" in header
        assert "November 2025" in header
        assert "Monthly Update" in header
    
    def test_format_system_capabilities(self):
        """Test system capabilities section."""
        updates = InvestorUpdates()
        section = updates._format_system_capabilities()
        
        assert "80 Tier Protocols" in section
        assert "20 Omega Directives" in section
        assert "20 MonsterRunner Protocols" in section
        assert "25 Learning Protocols" in section
    
    def test_format_compliance_section(self):
        """Test compliance section formatting."""
        updates = InvestorUpdates()
        section = updates._format_compliance_section(None)
        
        assert "REGULATORY COMPLIANCE" in section
        assert "FINRA" in section
        assert "SEC" in section
        assert "MiFID II" in section


class TestNotesImporter:
    """Tests for the Notes Importer."""
    
    def test_importer_initialization(self):
        """Test importer initializes correctly."""
        importer = NotesImporter()
        assert importer.client is not None
    
    def test_symbol_extraction(self):
        """Test stock symbol extraction from text."""
        importer = NotesImporter()
        
        text = "Looking at AAPL and MSFT today. GOOGL might be interesting too. The CEO said something."
        symbols = importer._extract_symbols(text)
        
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" in symbols
        assert "CEO" not in symbols
        assert "THE" not in symbols
    
    def test_symbol_extraction_known_tickers(self):
        """Test that known tickers are prioritized."""
        importer = NotesImporter()
        
        text = "TSLA and NVDA are volatile. XYZ is unknown."
        symbols = importer._extract_symbols(text)
        
        assert "TSLA" in symbols
        assert "NVDA" in symbols
        assert "XYZ" in symbols
    
    def test_watchlist_parsing_comma_separated(self):
        """Test parsing comma-separated watchlist."""
        importer = NotesImporter()
        
        text = "My watchlist: AAPL, MSFT, GOOGL, NVDA"
        symbols = importer._parse_watchlist(text)
        
        assert len(symbols) >= 4
        assert "AAPL" in symbols
        assert "MSFT" in symbols
    
    def test_watchlist_parsing_line_separated(self):
        """Test parsing line-separated watchlist."""
        importer = NotesImporter()
        
        text = """- AAPL
- MSFT
- GOOGL
* NVDA"""
        symbols = importer._parse_watchlist(text)
        
        assert len(symbols) >= 4


class TestDocSync:
    """Tests for the Documentation Sync module."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = DocSyncConfig()
        assert config.docs_folder == "docs"
        assert "*.md" in config.include_patterns
        assert config.prefix == "QuantraCore Apex - "
    
    def test_sync_initialization(self):
        """Test sync module initializes correctly."""
        sync = DocSync()
        assert sync.config is not None
        assert sync._sync_registry == {}
    
    def test_markdown_to_text_conversion(self):
        """Test markdown to plain text conversion."""
        sync = DocSync()
        
        markdown = "# Heading\n\n**Bold text** and *italic*.\n\n```python\ncode block\n```"
        text = sync._markdown_to_text(markdown)
        
        assert "Bold text" in text
        assert "italic" in text
        assert "Heading" in text
        assert "**" not in text
        assert "*" not in text or text.count("*") == text.count(" * ")
    
    def test_sync_result_model(self):
        """Test SyncResult model."""
        result = SyncResult(
            document_id="doc123",
            title="Test Doc",
            local_path="docs/test.md",
            url="https://docs.google.com/document/d/doc123/edit",
            synced_at=datetime.utcnow(),
            direction="export",
            success=True,
        )
        
        assert result.document_id == "doc123"
        assert result.success is True
        assert result.error is None


class TestModuleExports:
    """Tests for module exports and singleton instances."""
    
    def test_google_docs_module_exports(self):
        """Test that all expected exports are available."""
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
        
        assert GoogleDocsClient is not None
        assert google_docs_client is not None
        assert isinstance(google_docs_client, GoogleDocsClient)
        
        assert ResearchReportGenerator is not None
        assert research_report_generator is not None
        assert isinstance(research_report_generator, ResearchReportGenerator)
        
        assert TradeJournal is not None
        assert trade_journal is not None
        assert isinstance(trade_journal, TradeJournal)
        
        assert InvestorUpdates is not None
        assert investor_updates is not None
        assert isinstance(investor_updates, InvestorUpdates)
        
        assert NotesImporter is not None
        assert notes_importer is not None
        assert isinstance(notes_importer, NotesImporter)
        
        assert DocSync is not None
        assert doc_sync is not None
        assert isinstance(doc_sync, DocSync)
    
    def test_integrations_module_exports(self):
        """Test top-level integrations module exports."""
        from src.quantracore_apex.integrations import (
            google_docs_client,
            research_report_generator,
            trade_journal,
            investor_updates,
            notes_importer,
            doc_sync,
        )
        
        assert google_docs_client is not None
        assert research_report_generator is not None
        assert trade_journal is not None
        assert investor_updates is not None
        assert notes_importer is not None
        assert doc_sync is not None


class TestIntegrationAPIEndpoints:
    """Tests for the API endpoint request/response models."""
    
    def test_imported_note_model(self):
        """Test ImportedNote model."""
        note = ImportedNote(
            document_id="doc123",
            title="Test Note",
            content="Note content",
            symbols=["AAPL", "MSFT"],
            extracted_at=datetime.utcnow(),
            word_count=100,
        )
        
        assert note.document_id == "doc123"
        assert len(note.symbols) == 2
        assert note.word_count == 100
    
    def test_watchlist_model(self):
        """Test Watchlist model."""
        watchlist = Watchlist(
            name="Tech Stocks",
            symbols=["AAPL", "MSFT", "GOOGL"],
            source_document_id="doc456",
            extracted_at=datetime.utcnow(),
        )
        
        assert watchlist.name == "Tech Stocks"
        assert len(watchlist.symbols) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
