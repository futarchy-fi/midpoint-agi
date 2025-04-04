"""
Registry for storing and retrieving tools.

This module provides a registry for storing and retrieving tools.
The registry is used to keep track of all available tools.
"""

from typing import Dict, List, Any, Optional
import logging

class ToolRegistry:
    """Registry for storing and retrieving tools."""
    
    # Class variable to store all registered tools
    _tools = {}
    _initialized = False
    
    @classmethod
    def register_tool(cls, tool):
        """Register a tool with the registry."""
        if tool.name in cls._tools:
            # Only log a debug message instead of a warning when overwriting a tool
            logging.debug(f"Tool with name '{tool.name}' is already registered. Overwriting.")
        else:
            logging.debug(f"Registered tool: {tool.name}")
        
        cls._tools[tool.name] = tool
        
        return tool
    
    @classmethod
    def get_tool(cls, name: str):
        """Get a tool by name."""
        return cls._tools.get(name)
    
    @classmethod
    def get_tools(cls) -> List[Any]:
        """Get all registered tools."""
        return list(cls._tools.values())
    
    @classmethod
    def get_tool_schemas(cls) -> List[Dict[str, Any]]:
        """Get the schemas for all registered tools, excluding memory tools."""
        schemas = []
        
        for tool in cls._tools.values():
            # Skip memory-related tools as we handle memory automatically now
            if tool.name.startswith("store_memory") or tool.name.startswith("retrieve_memory"):
                logging.debug(f"Excluding memory tool from schemas: {tool.name}")
                continue
                
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            schemas.append(schema)
            
        return schemas

def initialize_all_tools():
    """Initialize and register all available tools."""
    if ToolRegistry._initialized:
        return
    
    # Import tool modules to register tools
    try:
        from midpoint.agents.tools import filesystem_tools
    except ImportError:
        logging.warning("Failed to import filesystem_tools")
        
    try:
        from midpoint.agents.tools import code_tools
    except ImportError:
        logging.warning("Failed to import code_tools")
        
    try:
        from midpoint.agents.tools import git_tools
    except ImportError:
        logging.warning("Failed to import git_tools")
        
    try:
        from midpoint.agents.tools import web_tools
    except ImportError:
        logging.warning("Failed to import web_tools")
        
    try:
        from midpoint.agents.tools import memory_tools
    except ImportError:
        logging.warning("Failed to import memory_tools")
    
    ToolRegistry._initialized = True
    logging.info(f"Initialized {len(ToolRegistry.get_tools())} tools") 