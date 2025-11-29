"""
Notes Importer for QuantraCore Apex.

Pulls research notes, watchlists, and trading ideas from Google Docs
for processing within the QuantraCore system.
"""

import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from src.quantracore_apex.integrations.google_docs.client import google_docs_client

logger = logging.getLogger(__name__)


class ImportedNote(BaseModel):
    """An imported research note."""
    document_id: str
    title: str
    content: str
    symbols: List[str]
    extracted_at: datetime
    word_count: int
    metadata: Dict[str, Any] = {}


class Watchlist(BaseModel):
    """An imported watchlist."""
    name: str
    symbols: List[str]
    source_document_id: str
    extracted_at: datetime
    notes: Optional[str] = None


class NotesImporter:
    """
    Imports and parses research notes from Google Docs.
    
    Features:
    - Symbol extraction from document text
    - Watchlist parsing
    - Research note categorization
    - Bulk import from multiple documents
    """
    
    SYMBOL_PATTERN = re.compile(r'\b[A-Z]{1,5}\b')
    COMMON_WORDS = {
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
        'WAS', 'ONE', 'OUR', 'OUT', 'HAS', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW',
        'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY',
        'SHE', 'TOO', 'USE', 'BUY', 'SELL', 'HOLD', 'LONG', 'SHORT', 'BULL',
        'BEAR', 'HIGH', 'LOW', 'OPEN', 'CLOSE', 'USD', 'EUR', 'GBP', 'JPY',
        'ETF', 'IPO', 'CEO', 'CFO', 'COO', 'CTO', 'NYSE', 'SEC', 'FDA', 'FED',
        'IMF', 'GDP', 'CPI', 'PPI', 'EPS', 'PE', 'PB', 'ROE', 'ROA', 'ROI',
        'YTD', 'MTD', 'QTD', 'ATH', 'ATL', 'SMA', 'EMA', 'RSI', 'MACD', 'BB',
        'API', 'JSON', 'CSV', 'PDF', 'URL', 'HTTP', 'SQL', 'HTML', 'CSS'
    }
    
    KNOWN_TICKERS = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
        'BRK', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL',
        'BAC', 'XOM', 'NFLX', 'COST', 'PEP', 'ABBV', 'TMO', 'CSCO', 'ACN',
        'MRK', 'AVGO', 'ABT', 'CVX', 'KO', 'PFE', 'WMT', 'NKE', 'LLY',
        'MCD', 'DHR', 'VZ', 'ORCL', 'ADBE', 'TXN', 'PM', 'NEE', 'UNP',
        'BMY', 'RTX', 'QCOM', 'INTC', 'HON', 'LOW', 'SBUX', 'AMD', 'IBM',
        'GE', 'CAT', 'BA', 'GS', 'BLK', 'AMAT', 'INTU', 'AXP', 'ISRG',
        'MDLZ', 'BKNG', 'ADI', 'GILD', 'TJX', 'VRTX', 'SYK', 'CVS', 'LRCX',
        'ZTS', 'MMC', 'MO', 'CI', 'REGN', 'CME', 'PLD', 'NOW', 'SNPS',
        'EOG', 'APD', 'KLAC', 'DUK', 'CL', 'SHW', 'ITW', 'BDX', 'SO',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VXX', 'UVXY', 'SQQQ',
        'TQQQ', 'SPXL', 'SPXS', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY'
    }
    
    def __init__(self):
        self.client = google_docs_client
    
    def _extract_symbols(self, text: str) -> List[str]:
        """
        Extract stock symbols from text.
        
        Args:
            text: Document text content
            
        Returns:
            list: Extracted and validated stock symbols
        """
        potential_symbols = self.SYMBOL_PATTERN.findall(text)
        
        valid_symbols = []
        seen = set()
        
        for symbol in potential_symbols:
            if symbol in seen:
                continue
            seen.add(symbol)
            
            if symbol in self.COMMON_WORDS:
                continue
            
            if symbol in self.KNOWN_TICKERS:
                valid_symbols.append(symbol)
            elif len(symbol) >= 2 and len(symbol) <= 5:
                valid_symbols.append(symbol)
        
        return valid_symbols
    
    def _parse_watchlist(self, text: str) -> List[str]:
        """
        Parse a watchlist from document text.
        
        Supports formats:
        - Comma-separated: AAPL, MSFT, GOOGL
        - Line-separated: One symbol per line
        - Bulleted: - AAPL or * AAPL
        
        Args:
            text: Document text
            
        Returns:
            list: Parsed symbols
        """
        symbols = []
        
        if ',' in text:
            parts = text.split(',')
            for part in parts:
                matches = self.SYMBOL_PATTERN.findall(part.strip())
                for match in matches:
                    if match not in self.COMMON_WORDS:
                        symbols.append(match)
        else:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                line = re.sub(r'^[-*\u2022\d.)\]]+\s*', '', line)
                matches = self.SYMBOL_PATTERN.findall(line)
                if matches:
                    first_match = matches[0]
                    if first_match not in self.COMMON_WORDS:
                        symbols.append(first_match)
        
        return list(dict.fromkeys(symbols))
    
    async def import_document(self, document_id: str) -> ImportedNote:
        """
        Import a single document as a research note.
        
        Args:
            document_id: Google Doc ID
            
        Returns:
            ImportedNote: Parsed document content
        """
        document = await self.client.get_document(document_id)
        text = await self.client.get_document_text(document_id)
        
        title = document.get('title', 'Untitled')
        symbols = self._extract_symbols(text)
        word_count = len(text.split())
        
        logger.info(f"Imported document {document_id}: {len(symbols)} symbols found")
        
        return ImportedNote(
            document_id=document_id,
            title=title,
            content=text,
            symbols=symbols,
            extracted_at=datetime.utcnow(),
            word_count=word_count,
            metadata={
                "revision_id": document.get('revisionId'),
            }
        )
    
    async def import_watchlist(
        self,
        document_id: str,
        watchlist_name: Optional[str] = None
    ) -> Watchlist:
        """
        Import a document as a watchlist.
        
        Args:
            document_id: Google Doc ID
            watchlist_name: Optional custom name for the watchlist
            
        Returns:
            Watchlist: Parsed watchlist
        """
        document = await self.client.get_document(document_id)
        text = await self.client.get_document_text(document_id)
        
        title = document.get('title', 'Untitled Watchlist')
        symbols = self._parse_watchlist(text)
        
        logger.info(f"Imported watchlist from {document_id}: {len(symbols)} symbols")
        
        return Watchlist(
            name=watchlist_name or title,
            symbols=symbols,
            source_document_id=document_id,
            extracted_at=datetime.utcnow(),
            notes=text[:500] if len(text) > 500 else text
        )
    
    async def search_and_import(
        self,
        query: str,
        max_documents: int = 10
    ) -> List[ImportedNote]:
        """
        Search for documents and import matching ones.
        
        Args:
            query: Search query
            max_documents: Maximum documents to import
            
        Returns:
            list: Imported notes
        """
        docs = await self.client.search_documents(query, max_documents)
        notes = []
        
        for doc in docs:
            try:
                note = await self.import_document(doc['id'])
                notes.append(note)
            except Exception as e:
                logger.warning(f"Failed to import {doc['id']}: {e}")
        
        return notes
    
    async def import_recent_notes(
        self,
        max_documents: int = 10,
        min_symbols: int = 1
    ) -> List[ImportedNote]:
        """
        Import recent documents that contain stock symbols.
        
        Args:
            max_documents: Maximum documents to check
            min_symbols: Minimum symbols required to include
            
        Returns:
            list: Imported notes with symbols
        """
        docs = await self.client.list_documents(max_documents * 2)
        notes = []
        
        for doc in docs:
            if len(notes) >= max_documents:
                break
            
            try:
                note = await self.import_document(doc['id'])
                if len(note.symbols) >= min_symbols:
                    notes.append(note)
            except Exception as e:
                logger.warning(f"Failed to import {doc['id']}: {e}")
        
        return notes
    
    async def get_all_symbols_from_notes(
        self,
        max_documents: int = 20
    ) -> Dict[str, Any]:
        """
        Extract all unique symbols from recent notes.
        
        Args:
            max_documents: Maximum documents to scan
            
        Returns:
            dict: Symbol extraction summary
        """
        notes = await self.import_recent_notes(max_documents, min_symbols=0)
        
        symbol_counts: Dict[str, int] = {}
        symbol_sources: Dict[str, List[str]] = {}
        
        for note in notes:
            for symbol in note.symbols:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                if symbol not in symbol_sources:
                    symbol_sources[symbol] = []
                symbol_sources[symbol].append(note.title)
        
        sorted_symbols = sorted(
            symbol_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "total_symbols": len(symbol_counts),
            "documents_scanned": len(notes),
            "symbols": [
                {
                    "symbol": symbol,
                    "mentions": count,
                    "sources": symbol_sources[symbol]
                }
                for symbol, count in sorted_symbols
            ],
            "extracted_at": datetime.utcnow().isoformat()
        }
    
    async def find_watchlist_documents(self) -> List[Dict[str, Any]]:
        """
        Find documents that look like watchlists.
        
        Returns:
            list: Potential watchlist documents
        """
        keywords = ["watchlist", "watch list", "symbols", "tickers", "portfolio"]
        potential = []
        
        for keyword in keywords:
            try:
                docs = await self.client.search_documents(keyword, max_results=5)
                for doc in docs:
                    if doc['id'] not in [p['id'] for p in potential]:
                        potential.append({
                            "id": doc['id'],
                            "name": doc['name'],
                            "matched_keyword": keyword,
                            "url": f"https://docs.google.com/document/d/{doc['id']}/edit"
                        })
            except Exception as e:
                logger.warning(f"Error searching for {keyword}: {e}")
        
        return potential


notes_importer = NotesImporter()
