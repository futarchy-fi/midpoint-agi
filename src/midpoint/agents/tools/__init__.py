"""
Tools for Midpoint agents.

This module provides tools that can be used by agents to interact with the 
environment, including filesystem operations, code search, and web interactions.
"""

from .registry import ToolRegistry
from .base import Tool

# Import tools from modules
from .filesystem_tools import list_directory, read_file, edit_file
from .code_tools import search_code
from .terminal_tools import run_terminal_cmd
from .git_tools import get_current_hash, check_repo_state, get_current_branch, create_branch, create_commit
from .web_tools import web_search, web_scrape
from .memory_tools import store_memory_document, retrieve_memory_documents

# Export public API
__all__ = [
    # Base classes
    'Tool',
    'ToolRegistry',
    
    # Filesystem tools
    'list_directory',
    'read_file',
    'edit_file',
    
    # Code tools
    'search_code',
    
    # Terminal tools
    'run_terminal_cmd',
    
    # Git tools
    'get_current_hash',
    'check_repo_state',
    'get_current_branch',
    'create_branch',
    'create_commit',
    
    # Web tools
    'web_search',
    'web_scrape',
    
    # Memory tools
    'store_memory_document',
    'retrieve_memory_documents',
    
    # Initialization function
    'initialize_all_tools'
]

def initialize_all_tools():
    """Initialize all tools by importing their modules."""
    # Import all tool modules to ensure they register with the ToolRegistry
    from . import filesystem_tools
    from . import code_tools
    from . import terminal_tools
    from . import git_tools
    from . import web_tools
    from . import memory_tools
    
    # Mark as initialized
    ToolRegistry._initialized = True
    
    # Return the number of registered tools
    return len(ToolRegistry.get_tools()) 