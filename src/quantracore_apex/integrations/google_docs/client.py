"""
Google Docs Client for QuantraCore Apex.

Handles OAuth2 authentication via Replit's connector system and provides
a unified interface for Google Docs API operations.
"""

import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Dict, Any, List
import httpx
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


class GoogleDocsClient:
    """
    Google Docs API client with Replit connector authentication.
    
    Uses Replit's OAuth2 connector system to manage access tokens
    automatically, handling token refresh transparently.
    
    Note: Google API client is synchronous, so all API calls are
    wrapped to run in a thread pool executor.
    """
    
    def __init__(self):
        self._cached_credentials: Optional[Credentials] = None
        self._credentials_expiry: Optional[datetime] = None
    
    async def _get_access_token(self) -> str:
        """
        Fetch access token from Replit's connector system.
        
        Returns:
            str: Valid OAuth2 access token
            
        Raises:
            RuntimeError: If token cannot be retrieved
        """
        hostname = os.environ.get("REPLIT_CONNECTORS_HOSTNAME")
        
        repl_identity = os.environ.get("REPL_IDENTITY")
        web_repl_renewal = os.environ.get("WEB_REPL_RENEWAL")
        
        if repl_identity:
            x_replit_token = f"repl {repl_identity}"
        elif web_repl_renewal:
            x_replit_token = f"depl {web_repl_renewal}"
        else:
            raise RuntimeError("X_REPLIT_TOKEN not found - not running in Replit environment")
        
        if not hostname:
            raise RuntimeError("REPLIT_CONNECTORS_HOSTNAME not set - Google Docs not connected")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://{hostname}/api/v2/connection",
                params={
                    "include_secrets": "true",
                    "connector_names": "google-docs"
                },
                headers={
                    "Accept": "application/json",
                    "X_REPLIT_TOKEN": x_replit_token
                }
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to fetch Google Docs connection: {response.status_code}")
            
            data = response.json()
            items = data.get("items", [])
            
            if not items:
                raise RuntimeError("Google Docs not connected - please set up the connection first")
            
            connection = items[0]
            settings = connection.get("settings", {})
            
            access_token = settings.get("access_token")
            if not access_token:
                oauth_settings = settings.get("oauth", {}).get("credentials", {})
                access_token = oauth_settings.get("access_token")
            
            if not access_token:
                raise RuntimeError("No access token found in Google Docs connection")
            
            return access_token
    
    def _build_docs_service(self, access_token: str):
        """Build Google Docs service synchronously."""
        credentials = Credentials(token=access_token)
        return build("docs", "v1", credentials=credentials, cache_discovery=False)
    
    def _build_drive_service(self, access_token: str):
        """Build Google Drive service synchronously."""
        credentials = Credentials(token=access_token)
        return build("drive", "v3", credentials=credentials, cache_discovery=False)
    
    async def create_document(self, title: str) -> Dict[str, Any]:
        """
        Create a new Google Doc.
        
        Args:
            title: Document title
            
        Returns:
            dict: Created document metadata including documentId
        """
        access_token = await self._get_access_token()
        
        def _create():
            service = self._build_docs_service(access_token)
            return service.documents().create(body={"title": title}).execute()
        
        loop = asyncio.get_event_loop()
        document = await loop.run_in_executor(_executor, _create)
        logger.info(f"Created document: {document.get('documentId')}")
        return document
    
    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get a document by ID.
        
        Args:
            document_id: Google Doc ID
            
        Returns:
            dict: Full document content
        """
        access_token = await self._get_access_token()
        
        def _get():
            service = self._build_docs_service(access_token)
            return service.documents().get(documentId=document_id).execute()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _get)
    
    async def get_document_text(self, document_id: str) -> str:
        """
        Extract plain text content from a document.
        
        Args:
            document_id: Google Doc ID
            
        Returns:
            str: Plain text content of the document
        """
        document = await self.get_document(document_id)
        content = document.get("body", {}).get("content", [])
        
        text_parts = []
        for element in content:
            if "paragraph" in element:
                paragraph = element["paragraph"]
                for para_element in paragraph.get("elements", []):
                    if "textRun" in para_element:
                        text_parts.append(para_element["textRun"].get("content", ""))
        
        return "".join(text_parts)
    
    async def batch_update(self, document_id: str, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform batch updates on a document.
        
        Args:
            document_id: Google Doc ID
            requests: List of update requests
            
        Returns:
            dict: Update response
        """
        access_token = await self._get_access_token()
        
        def _update():
            service = self._build_docs_service(access_token)
            return service.documents().batchUpdate(
                documentId=document_id,
                body={"requests": requests}
            ).execute()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _update)
    
    async def insert_text(self, document_id: str, text: str, index: int = 1) -> Dict[str, Any]:
        """
        Insert text at a specific index.
        
        Args:
            document_id: Google Doc ID
            text: Text to insert
            index: Character index (1 = start of document)
            
        Returns:
            dict: Update response
        """
        requests = [
            {
                "insertText": {
                    "location": {"index": index},
                    "text": text
                }
            }
        ]
        return await self.batch_update(document_id, requests)
    
    async def append_text(self, document_id: str, text: str) -> Dict[str, Any]:
        """
        Append text to the end of a document.
        
        Args:
            document_id: Google Doc ID
            text: Text to append
            
        Returns:
            dict: Update response
        """
        document = await self.get_document(document_id)
        body = document.get("body", {})
        content = body.get("content", [])
        
        end_index = 1
        if content:
            last_element = content[-1]
            end_index = last_element.get("endIndex", 1) - 1
        
        return await self.insert_text(document_id, text, max(1, end_index))
    
    async def insert_heading(
        self, 
        document_id: str, 
        text: str, 
        heading_level: int = 1,
        index: int = 1
    ) -> Dict[str, Any]:
        """
        Insert a heading with specified level.
        
        Args:
            document_id: Google Doc ID
            text: Heading text
            heading_level: 1-6 for H1-H6
            index: Character index
            
        Returns:
            dict: Update response
        """
        heading_type = f"HEADING_{min(max(heading_level, 1), 6)}"
        
        requests = [
            {
                "insertText": {
                    "location": {"index": index},
                    "text": f"{text}\n"
                }
            },
            {
                "updateParagraphStyle": {
                    "range": {
                        "startIndex": index,
                        "endIndex": index + len(text) + 1
                    },
                    "paragraphStyle": {
                        "namedStyleType": heading_type
                    },
                    "fields": "namedStyleType"
                }
            }
        ]
        return await self.batch_update(document_id, requests)
    
    async def list_documents(self, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        List recent Google Docs.
        
        Args:
            max_results: Maximum number of documents to return
            
        Returns:
            list: List of document metadata
        """
        access_token = await self._get_access_token()
        
        def _list():
            service = self._build_drive_service(access_token)
            results = service.files().list(
                pageSize=max_results,
                fields="files(id, name, mimeType, modifiedTime, createdTime)",
                q="mimeType='application/vnd.google-apps.document'",
                orderBy="modifiedTime desc"
            ).execute()
            return results.get("files", [])
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _list)
    
    async def search_documents(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents by name or content.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            list: Matching documents
        """
        access_token = await self._get_access_token()
        
        def _search():
            service = self._build_drive_service(access_token)
            results = service.files().list(
                pageSize=max_results,
                fields="files(id, name, mimeType, modifiedTime)",
                q=f"mimeType='application/vnd.google-apps.document' and fullText contains '{query}'",
                orderBy="modifiedTime desc"
            ).execute()
            return results.get("files", [])
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _search)
    
    async def check_connection(self) -> Dict[str, Any]:
        """
        Verify Google Docs connection is working.
        
        Returns:
            dict: Connection status with details
        """
        try:
            await self._get_access_token()
            docs = await self.list_documents(max_results=1)
            return {
                "connected": True,
                "status": "operational",
                "document_count": len(docs),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "connected": False,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


google_docs_client = GoogleDocsClient()
