"""
Memory tools for the Midpoint agent system.

This module provides tools for interacting with the memory repository,
including storing and retrieving documents.
"""

import os
import json
import logging
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from .base import Tool
from .registry import ToolRegistry

# Try to import memory tools directly
try:
    from scripts.memory_tools import store_document as system_store_document
    from scripts.memory_tools import retrieve_documents as system_retrieve_documents
    from scripts.memory_tools import get_repo_path as system_get_repo_path
    from scripts.memory_tools import get_memory_for_code_hash as system_get_memory_for_code_hash
    from scripts.memory_tools import update_cross_reference as system_update_cross_reference
    HAS_MEMORY_TOOLS = True
except ImportError:
    # Use fallback implementations if needed
    logging.warning("Memory tools import failed. Using fallback implementations.")
    HAS_MEMORY_TOOLS = False
    
    def system_get_repo_path():
        """Fallback implementation to get memory repository path."""
        return os.environ.get("MEMORY_REPO_PATH", os.path.expanduser("~/.midpoint/memory"))
        
    def system_store_document(content, category, metadata=None, repo_path=None):
        """Fallback implementation to store a document."""
        logging.warning("Using fallback store_document implementation.")
        # Get repository path
        repo_path = repo_path or system_get_repo_path()
        repo_path = Path(repo_path)
        
        # Create basic directories
        docs_dir = repo_path / "documents" / category
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple filename (check if metadata has a custom_filename)
        if metadata and 'custom_filename' in metadata:
            filename = metadata['custom_filename']
        else:
            filename = f"doc_{int(time.time())}.md"
        doc_path = docs_dir / filename
        
        # Write content
        with open(doc_path, "w") as f:
            f.write(content)
            
        logging.info(f"Stored document at: {doc_path} (fallback implementation)")
        return str(doc_path.relative_to(repo_path) if repo_path in doc_path.parents else doc_path)
        
    def system_retrieve_documents(category=None, limit=10, repo_path=None):
        """Fallback implementation to retrieve documents."""
        logging.warning("Using fallback retrieve_documents implementation.")
        # Get repository path
        repo_path = repo_path or system_get_repo_path()
        repo_path = Path(repo_path)
        
        # Set search path
        if category:
            search_path = repo_path / "documents" / category
        else:
            search_path = repo_path / "documents"
        
        results = []
        
        if search_path.exists():
            # Find all .md files
            files = list(search_path.glob("**/*.md"))
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Limit results
            files = files[:limit]
            
            # Read contents
            for file in files:
                try:
                    with open(file, "r") as f:
                        results.append((str(file.relative_to(repo_path) if repo_path in file.parents else file), f.read()))
                except:
                    pass
        
        return results
        
    def system_update_cross_reference(code_hash, memory_hash, repo_path=None):
        """Fallback implementation to update cross-reference between code and memory."""
        logging.warning("Using fallback update_cross_reference implementation.")
        # Get repository path
        repo_path = repo_path or system_get_repo_path()
        repo_path = Path(repo_path)
        
        # Ensure metadata directory exists
        metadata_dir = repo_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to cross-reference file
        cross_ref_path = metadata_dir / "cross-reference.json"
        
        # Load existing cross-reference or create new
        if cross_ref_path.exists():
            try:
                with open(cross_ref_path, "r") as f:
                    cross_ref = json.load(f)
            except:
                cross_ref = {"mappings": [], "latest": {}}
        else:
            cross_ref = {"mappings": [], "latest": {}}
        
        # Update cross-reference
        timestamp = int(time.time())
        cross_ref["mappings"].append({
            "code_hash": code_hash,
            "memory_hash": memory_hash,
            "timestamp": timestamp
        })
        cross_ref["latest"][code_hash] = memory_hash
        
        # Write updated cross-reference
        with open(cross_ref_path, "w") as f:
            json.dump(cross_ref, f, indent=2)
        
        logging.debug(f"Updated cross-reference: {code_hash[:7]} → {memory_hash[:7]}")
        
    def system_get_memory_for_code_hash(code_hash, repo_path=None):
        """Fallback implementation to get memory hash for code hash."""
        logging.warning("Using fallback get_memory_for_code_hash implementation.")
        # Get repository path
        repo_path = repo_path or system_get_repo_path()
        repo_path = Path(repo_path)
        
        # Path to cross-reference file
        cross_ref_path = repo_path / "metadata" / "cross-reference.json"
        
        if not cross_ref_path.exists():
            return None
            
        # Load cross-reference
        try:
            with open(cross_ref_path, "r") as f:
                cross_ref = json.load(f)
        except:
            return None
            
        # Get memory hash for code hash
        return cross_ref.get("latest", {}).get(code_hash)
        
# Define tools

class StoreMemoryDocumentTool(Tool):
    """Tool for storing documents in the memory repository."""
    
    @property
    def name(self) -> str:
        return "store_memory_document"
    
    @property
    def description(self) -> str:
        return "Store a document in the memory repository"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to store in the document"
                },
                "category": {
                    "type": "string",
                    "description": "Category to store the document under",
                    "default": "general"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata to store with the document",
                    "additionalProperties": True
                },
                "memory_repo_path": {
                    "type": "string",
                    "description": "Path to the memory repository (optional)"
                }
            },
            "required": ["content"]
        }
    
    async def execute(self, content: str, category: str = "general", metadata: Dict[str, Any] = None, memory_repo_path: str = None) -> Dict[str, Any]:
        """Store a document in the memory repository."""
        try:
            # Call the implementation
            document_path = system_store_document(content, category, metadata, memory_repo_path)
            return {
                "success": True,
                "document_path": document_path
            }
        except Exception as e:
            logging.error(f"Error storing document: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

class RetrieveMemoryDocumentsTool(Tool):
    """Tool for retrieving documents from the memory repository."""
    
    @property
    def name(self) -> str:
        return "retrieve_memory_documents"
    
    @property
    def description(self) -> str:
        return "Retrieve documents from the memory repository"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category to retrieve documents from",
                    "default": None
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to retrieve",
                    "default": 10
                },
                "memory_repo_path": {
                    "type": "string",
                    "description": "Path to the memory repository (optional)"
                }
            },
            "required": []
        }
    
    async def execute(self, category: str = None, limit: int = 10, memory_repo_path: str = None) -> Dict[str, Any]:
        """Retrieve documents from the memory repository."""
        # Call the implementation
        documents = system_retrieve_documents(
            category=category,
            limit=limit,
            repo_path=memory_repo_path
        )
        
        # Format the results
        result_docs = []
        for path, content in documents:
            # Truncate content if too long
            truncated_content = content[:1000] + ("..." if len(content) > 1000 else "")
            result_docs.append({
                "path": path,
                "content": truncated_content,
                "full_content": content
            })
        
        return {
            "success": True,
            "documents": result_docs,
            "total": len(documents),
            "message": f"Retrieved {len(documents)} documents"
        }

# Instantiate and register the tools
store_memory_document_tool = StoreMemoryDocumentTool()
retrieve_memory_documents_tool = RetrieveMemoryDocumentsTool()

ToolRegistry.register_tool(store_memory_document_tool)
ToolRegistry.register_tool(retrieve_memory_documents_tool)

# Export tool functions
async def store_memory_document(content: str, category: str, metadata: Dict[str, Any] = None, memory_repo_path: str = None) -> Dict[str, Any]:
    """Store a document in the memory repository."""
    return await store_memory_document_tool.execute(
        content=content,
        category=category,
        metadata=metadata,
        memory_repo_path=memory_repo_path
    )

async def retrieve_memory_documents(category: str = None, limit: int = 10, memory_repo_path: str = None) -> Dict[str, Any]:
    """Retrieve documents from the memory repository."""
    return await retrieve_memory_documents_tool.execute(
        category=category,
        limit=limit,
        memory_repo_path=memory_repo_path
    ) 