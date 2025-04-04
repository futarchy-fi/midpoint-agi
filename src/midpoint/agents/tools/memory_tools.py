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
                },
                "memory_hash": {
                    "type": "string",
                    "description": "The memory hash to operate on - must check out this hash before making changes (required)"
                }
            },
            "required": ["content", "memory_hash"]
        }
    
    @property
    def required_parameters(self) -> List[str]:
        return ["content", "memory_hash"]
    
    async def execute(self, content: str, memory_hash: str, category: str = "general", metadata: Dict[str, Any] = None, memory_repo_path: str = None) -> Dict[str, Any]:
        """Store a document in the memory repository."""
        try:
            # Initialize metadata if None
            metadata = metadata or {}
            
            # Always ensure memory_hash is included in metadata
            if "memory_hash" not in metadata:
                metadata["memory_hash"] = memory_hash
            
            # Call the implementation with memory_hash
            document_path = system_store_document(
                content=content,
                category=category,
                metadata=metadata,
                repo_path=memory_repo_path,
                memory_hash=memory_hash
            )
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
                },
                "memory_hash": {
                    "type": "string",
                    "description": "The memory hash to operate on - must check out this hash before reading documents (required)"
                }
            },
            "required": ["memory_hash"]
        }
    
    @property
    def required_parameters(self) -> List[str]:
        return ["memory_hash"]
    
    async def execute(self, memory_hash: str, category: str = None, limit: int = 10, memory_repo_path: str = None) -> Dict[str, Any]:
        """Retrieve documents from the memory repository."""
        try:
            # Call the implementation with memory_hash
            documents = system_retrieve_documents(
                category=category,
                limit=limit,
                repo_path=memory_repo_path,
                memory_hash=memory_hash
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
        except Exception as e:
            logging.error(f"Error retrieving documents: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Instantiate and register the tools
store_memory_document_tool = StoreMemoryDocumentTool()
retrieve_memory_documents_tool = RetrieveMemoryDocumentsTool()

ToolRegistry.register_tool(store_memory_document_tool)
ToolRegistry.register_tool(retrieve_memory_documents_tool)

# Export tool functions
async def store_memory_document(content: str, category: str, metadata: Dict[str, Any] = None, memory_repo_path: str = None, memory_hash: str = None) -> Dict[str, Any]:
    """
    Store a document in the memory repository.
    
    Args:
        content: Content to store in the document
        category: Category to store the document under
        metadata: Additional metadata to store with the document
        memory_repo_path: Path to the memory repository
        memory_hash: REQUIRED hash to operate on - must check out this hash before making changes
        
    Returns:
        Dictionary with success status and document information
    """
    # Validate memory_hash is present either as parameter or in metadata
    if not memory_hash:
        if metadata and "memory_hash" in metadata:
            memory_hash = metadata["memory_hash"]
        else:
            raise ValueError("memory_hash is required - either as parameter or in metadata['memory_hash']")
            
    return await store_memory_document_tool.execute(
        content=content,
        category=category,
        metadata=metadata,
        memory_repo_path=memory_repo_path,
        memory_hash=memory_hash
    )

async def retrieve_memory_documents(category: str = None, limit: int = 10, memory_repo_path: str = None, memory_hash: str = None) -> Dict[str, Any]:
    """
    Retrieve documents from the memory repository.
    
    Args:
        category: Category to retrieve documents from
        limit: Maximum number of documents to retrieve
        memory_repo_path: Path to the memory repository
        memory_hash: REQUIRED hash to operate on - must check out this hash before reading documents
        
    Returns:
        Dictionary with success status and document information
    """
    # Validate memory_hash is present
    if not memory_hash:
        raise ValueError("memory_hash is required")
        
    return await retrieve_memory_documents_tool.execute(
        category=category,
        limit=limit,
        memory_repo_path=memory_repo_path,
        memory_hash=memory_hash
    )

def retrieve_recent_memory(memory_hash, char_limit=5000, repo_path=None):
    """
    Retrieve the most recent documents from the general_memory folder up to a specified character limit.
    
    Args:
        memory_hash: Required hash to operate on - must check out this hash before reading documents
        char_limit: Maximum total number of characters to retrieve
        repo_path: Path to the memory repository
        
    Returns:
        A tuple containing (total_chars, list of (path, content, timestamp) tuples)
    """
    # Get repository path
    repo_path = repo_path or get_repo_path()
    repo_path = Path(repo_path)
    
    # Validate memory_hash parameter
    if not memory_hash:
        raise ValueError("Memory hash is required")
    
    # Keep track of original position to restore later
    original_branch = None
    current_hash = None
    
    try:
        # Get current branch/hash
        try:
            current_hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            
            if current_hash_result.returncode == 0:
                current_hash = current_hash_result.stdout.strip()
                
                # Get current branch
                branch_result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                
                if branch_result.returncode == 0:
                    original_branch = branch_result.stdout.strip()
                    logging.debug(f"Memory: Current branch: {original_branch}, hash: {current_hash}")
            else:
                logging.error(f"Memory: Failed to get current hash: {current_hash_result.stderr}")
                raise ValueError(f"Failed to get current hash: {current_hash_result.stderr}")
                
        except Exception as e:
            logging.error(f"Memory: Error getting current hash: {str(e)}")
            raise ValueError(f"Error getting current hash: {str(e)}")
        
        # Check if we need to switch to the memory hash
        if current_hash != memory_hash:
            # Check if the target hash exists
            hash_check = subprocess.run(
                ["git", "cat-file", "-t", memory_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            
            if hash_check.returncode == 0:
                logging.debug(f"Memory: Checking out memory hash: {memory_hash}")
                # Checkout the memory hash
                checkout_result = subprocess.run(
                    ["git", "checkout", memory_hash],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                
                if checkout_result.returncode != 0:
                    error_msg = checkout_result.stderr
                    logging.error(f"Memory: Failed to checkout memory hash: {memory_hash}, error: {error_msg}")
                    raise ValueError(f"Cannot checkout memory hash: {memory_hash}. Error: {error_msg}")
            else:
                error_msg = hash_check.stderr
                logging.error(f"Memory: Memory hash does not exist: {memory_hash}, error: {error_msg}")
                raise ValueError(f"Memory hash does not exist: {memory_hash}. Error: {error_msg}")
        else:
            logging.debug(f"Memory: Already at memory hash: {memory_hash}")
        
        # Set search path to specifically target general_memory folder
        search_path = repo_path / "documents" / "../general_memory"
        
        if not search_path.exists():
            if hasattr(logging, 'debug'):
                logging.debug(f"Path does not exist: {search_path}")
            return 0, []
        
        # Find all .md files
        files = list(search_path.glob("**/*.md"))
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Start with newest files, process until we hit the limit
        temp_results = []  # To store results in newest-first order
        total_chars = 0
        
        for file in files:
            file_timestamp = file.stat().st_mtime
            file_size = file.stat().st_size  # Get file size before reading
            space_left = char_limit - total_chars
            
            if space_left <= 0:
                # No more space left, stop processing files
                break
                
            # Process the file: fully if there's space, or partially if needed
            if file_size <= space_left:
                # File fits completely
                with open(file, "r") as f:
                    content = f.read()
                temp_results.append((str(file.relative_to(repo_path)), content, file_timestamp))
                total_chars += len(content)
            else:
                # Only read the tail portion of the file that fits
                with open(file, "r") as f:
                    # If file is too large, seek to position where we can read the last 'space_left' characters
                    f.seek(max(0, file_size - space_left))
                    content = f.read(space_left)
                temp_results.append((str(file.relative_to(repo_path)), content, file_timestamp))
                total_chars += len(content)
                break  # No more space, we're done
        
        # Reverse the results so oldest appears first, newest last
        temp_results.reverse()
        
        return total_chars, temp_results
        
    finally:
        # Restore original branch if we changed it
        if original_branch and current_hash != memory_hash:
            try:
                logging.debug(f"Memory: Restoring original position: {original_branch}")
                subprocess.run(
                    ["git", "checkout", original_branch],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                logging.warning(f"Memory: Failed to restore original branch: {str(e)}") 

async def get_memory_diff(initial_hash: str, final_hash: str, memory_repo_path: str = None, max_size: int = 50000) -> Dict[str, Any]:
    """
    Get a diff between two memory repository states.
    
    Args:
        initial_hash: Starting memory hash
        final_hash: Ending memory hash
        memory_repo_path: Path to the memory repository (optional)
        max_size: Maximum diff size in characters
        
    Returns:
        Dictionary containing:
        - diff_summary: Summary of changes
        - changed_files: List of files changed
        - diff_content: Actual diff content (truncated if too large)
        - truncated: Whether the diff was truncated
    """
    result = {
        "diff_summary": "",
        "changed_files": [],
        "diff_content": "",
        "truncated": False
    }
    
    try:
        # Get repository path
        memory_repo_path = memory_repo_path or system_get_repo_path()
        
        # Save the current state
        try:
            current_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
        except Exception as e:
            current_hash = None
            logging.warning(f"Failed to get current memory hash: {str(e)}")
        
        try:
            # Get list of changed files
            cmd = ["git", "diff", "--name-only", initial_hash, final_hash]
            proc = subprocess.run(
                cmd,
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            result["changed_files"] = [
                file for file in proc.stdout.strip().split("\n") 
                if file.strip()
            ]
            
            # Get diff summary (stats)
            cmd = ["git", "diff", "--stat", initial_hash, final_hash]
            proc = subprocess.run(
                cmd,
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            result["diff_summary"] = proc.stdout.strip()
            
            # Get complete diff
            cmd = ["git", "diff", initial_hash, final_hash]
            proc = subprocess.run(
                cmd,
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            diff_content = proc.stdout
            
            # Truncate if too large
            if len(diff_content) > max_size:
                result["diff_content"] = diff_content[:max_size] + "\n[...TRUNCATED...]"
                result["truncated"] = True
            else:
                result["diff_content"] = diff_content
                
        finally:
            # Restore original state if needed
            if current_hash and current_hash != final_hash:
                try:
                    subprocess.run(
                        ["git", "checkout", current_hash],
                        cwd=memory_repo_path,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                except:
                    logging.warning(f"Failed to restore original memory hash state")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error getting memory diff: {str(e)}") from e 