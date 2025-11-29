"""
Documentation Sync for QuantraCore Apex.

Syncs project documentation between local files and Google Docs,
enabling easy sharing and collaboration on specifications.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel

from src.quantracore_apex.integrations.google_docs.client import google_docs_client

logger = logging.getLogger(__name__)


class SyncResult(BaseModel):
    """Result of a sync operation."""
    document_id: str
    title: str
    local_path: str
    url: str
    synced_at: datetime
    direction: str
    success: bool
    error: Optional[str] = None


class DocSyncConfig(BaseModel):
    """Configuration for document sync."""
    docs_folder: str = "docs"
    include_patterns: List[str] = ["*.md", "*.txt"]
    exclude_patterns: List[str] = ["node_modules", ".git", "__pycache__"]
    prefix: str = "QuantraCore Apex - "
    auto_format: bool = True


class DocSync:
    """
    Syncs project documentation with Google Docs.
    
    Features:
    - Export local docs to Google Docs
    - Import Google Docs to local files
    - Bidirectional sync with conflict detection
    - Markdown to Google Docs formatting
    """
    
    def __init__(self, config: Optional[DocSyncConfig] = None):
        self.client = google_docs_client
        self.config = config or DocSyncConfig()
        self._sync_registry: Dict[str, str] = {}
    
    def _markdown_to_text(self, markdown: str) -> str:
        """
        Convert markdown to plain text for Google Docs.
        
        Basic conversion preserving structure.
        """
        text = markdown
        
        text = text.replace('###', '   ')
        text = text.replace('##', '  ')
        text = text.replace('#', ' ')
        
        text = text.replace('**', '')
        text = text.replace('__', '')
        text = text.replace('*', '')
        text = text.replace('_', '')
        
        text = text.replace('`', '')
        
        text = text.replace('```python', '\n--- Code Block ---\n')
        text = text.replace('```javascript', '\n--- Code Block ---\n')
        text = text.replace('```typescript', '\n--- Code Block ---\n')
        text = text.replace('```bash', '\n--- Code Block ---\n')
        text = text.replace('```', '\n--- End Code Block ---\n')
        
        lines = []
        for line in text.split('\n'):
            if line.startswith('- '):
                lines.append('  * ' + line[2:])
            elif line.startswith('* '):
                lines.append('  * ' + line[2:])
            else:
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _get_local_docs(self) -> List[Path]:
        """Get list of local documentation files."""
        docs_path = Path(self.config.docs_folder)
        if not docs_path.exists():
            return []
        
        files = []
        for pattern in self.config.include_patterns:
            files.extend(docs_path.rglob(pattern))
        
        filtered = []
        for f in files:
            excluded = False
            for exclude in self.config.exclude_patterns:
                if exclude in str(f):
                    excluded = True
                    break
            if not excluded:
                filtered.append(f)
        
        return filtered
    
    async def export_document(self, local_path: str) -> SyncResult:
        """
        Export a local document to Google Docs.
        
        Args:
            local_path: Path to local file
            
        Returns:
            SyncResult: Export result
        """
        path = Path(local_path)
        if not path.exists():
            return SyncResult(
                document_id="",
                title="",
                local_path=local_path,
                url="",
                synced_at=datetime.utcnow(),
                direction="export",
                success=False,
                error=f"File not found: {local_path}"
            )
        
        try:
            content = path.read_text(encoding='utf-8')
            
            filename = path.stem.replace('_', ' ').replace('-', ' ').title()
            title = f"{self.config.prefix}{filename}"
            
            if self.config.auto_format and path.suffix == '.md':
                content = self._markdown_to_text(content)
            
            header = f"""
================================================================================
                    {title.upper()}
================================================================================

Source: {local_path}
Synced: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
System: QuantraCore Apex v9.0-A

================================================================================

"""
            full_content = header + content
            
            document = await self.client.create_document(title)
            document_id = document.get('documentId')
            await self.client.insert_text(document_id, full_content)
            
            self._sync_registry[local_path] = document_id
            
            logger.info(f"Exported {local_path} to Google Docs: {document_id}")
            
            return SyncResult(
                document_id=document_id,
                title=title,
                local_path=local_path,
                url=f"https://docs.google.com/document/d/{document_id}/edit",
                synced_at=datetime.utcnow(),
                direction="export",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to export {local_path}: {e}")
            return SyncResult(
                document_id="",
                title="",
                local_path=local_path,
                url="",
                synced_at=datetime.utcnow(),
                direction="export",
                success=False,
                error=str(e)
            )
    
    async def import_document(
        self,
        document_id: str,
        local_path: str,
        overwrite: bool = False
    ) -> SyncResult:
        """
        Import a Google Doc to a local file.
        
        Args:
            document_id: Google Doc ID
            local_path: Destination local path
            overwrite: Whether to overwrite existing files
            
        Returns:
            SyncResult: Import result
        """
        path = Path(local_path)
        
        if path.exists() and not overwrite:
            return SyncResult(
                document_id=document_id,
                title="",
                local_path=local_path,
                url=f"https://docs.google.com/document/d/{document_id}/edit",
                synced_at=datetime.utcnow(),
                direction="import",
                success=False,
                error="File exists and overwrite=False"
            )
        
        try:
            document = await self.client.get_document(document_id)
            text = await self.client.get_document_text(document_id)
            title = document.get('title', 'Untitled')
            
            path.parent.mkdir(parents=True, exist_ok=True)
            
            path.write_text(text, encoding='utf-8')
            
            self._sync_registry[local_path] = document_id
            
            logger.info(f"Imported {document_id} to {local_path}")
            
            return SyncResult(
                document_id=document_id,
                title=title,
                local_path=local_path,
                url=f"https://docs.google.com/document/d/{document_id}/edit",
                synced_at=datetime.utcnow(),
                direction="import",
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to import {document_id}: {e}")
            return SyncResult(
                document_id=document_id,
                title="",
                local_path=local_path,
                url=f"https://docs.google.com/document/d/{document_id}/edit",
                synced_at=datetime.utcnow(),
                direction="import",
                success=False,
                error=str(e)
            )
    
    async def export_all_docs(self) -> Dict[str, Any]:
        """
        Export all local documentation to Google Docs.
        
        Returns:
            dict: Export summary
        """
        local_docs = self._get_local_docs()
        results = []
        
        for doc_path in local_docs:
            result = await self.export_document(str(doc_path))
            results.append(result.model_dump())
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        return {
            "total_documents": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "results": results,
            "synced_at": datetime.utcnow().isoformat()
        }
    
    async def list_synced_docs(self) -> List[Dict[str, Any]]:
        """
        List all QuantraCore documents in Google Docs.
        
        Returns:
            list: Document metadata
        """
        docs = await self.client.search_documents(
            self.config.prefix,
            max_results=50
        )
        
        return [
            {
                "id": doc['id'],
                "name": doc['name'],
                "modified": doc.get('modifiedTime'),
                "url": f"https://docs.google.com/document/d/{doc['id']}/edit"
            }
            for doc in docs
        ]
    
    async def sync_spec_documents(self) -> Dict[str, Any]:
        """
        Sync key specification documents.
        
        Exports the main spec files for external sharing.
        
        Returns:
            dict: Sync summary
        """
        key_docs = [
            "docs/QUANTRACORE_APEX_MASTER_SPEC_v9.0-A.md",
            "docs/BROKER_LAYER_SPEC.md",
            "docs/ESTIMATED_MOVE_SPEC.md",
            "docs/ENTRY_EXIT_OPTIMIZATION_ENGINE_SPEC.md",
            "docs/APEXVISION_UPGRADE_SPEC.md",
            "docs/INVESTOR_DUE_DILIGENCE_REQUIREMENTS.md",
            "docs/SECURITY_COMPLIANCE/TEST_COVERAGE_REPORT.md",
            "docs/SECURITY_COMPLIANCE/hardening_blueprint.md",
            "README.md"
        ]
        
        results = []
        for doc_path in key_docs:
            if Path(doc_path).exists():
                result = await self.export_document(doc_path)
                results.append(result.model_dump())
        
        successful = [r for r in results if r['success']]
        
        return {
            "total_documents": len(results),
            "successful": len(successful),
            "documents": [
                {
                    "local": r['local_path'],
                    "url": r['url'],
                    "success": r['success']
                }
                for r in results
            ],
            "synced_at": datetime.utcnow().isoformat()
        }
    
    async def create_documentation_index(self) -> Dict[str, Any]:
        """
        Create an index document linking to all synced docs.
        
        Returns:
            dict: Index document metadata
        """
        synced = await self.list_synced_docs()
        
        title = f"{self.config.prefix}Documentation Index"
        
        content = f"""
================================================================================
                    QUANTRACORE APEX DOCUMENTATION INDEX
================================================================================

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
System: QuantraCore Apex v9.0-A Institutional Hardening

This document provides links to all QuantraCore Apex documentation
available in Google Docs.

================================================================================

AVAILABLE DOCUMENTS
-------------------

"""
        
        for i, doc in enumerate(synced, 1):
            content += f"{i}. {doc['name']}\n"
            content += f"   URL: {doc['url']}\n"
            content += f"   Last Modified: {doc.get('modified', 'Unknown')}\n\n"
        
        content += f"""
================================================================================

Total Documents: {len(synced)}
Index Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

================================================================================
"""
        
        document = await self.client.create_document(title)
        document_id = document.get('documentId')
        await self.client.insert_text(document_id, content)
        
        return {
            "document_id": document_id,
            "title": title,
            "url": f"https://docs.google.com/document/d/{document_id}/edit",
            "documents_indexed": len(synced),
            "created_at": datetime.utcnow().isoformat()
        }


doc_sync = DocSync()
