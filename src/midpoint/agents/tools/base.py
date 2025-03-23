"""
Base tool class definition.

This module defines the base Tool class that all tools must inherit from.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, Union, List, Type

class Tool(ABC):
    """Base class for all tools used by Midpoint agents."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tool."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass
        
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """JSON Schema for the parameters this tool accepts."""
        pass
        
    @property
    def schema(self) -> Dict[str, Any]:
        """Get the OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_parameters
                }
            }
        }
    
    @property
    def required_parameters(self) -> List[str]:
        """List of required parameter names."""
        return []
    
    @abstractmethod
    async def execute(self, **kwargs) -> Union[str, Dict[str, Any]]:
        """Execute the tool with the provided parameters."""
        pass
    
    async def log_execution(self, **kwargs) -> None:
        """Log relevant information about the tool execution."""
        logging.debug(f"Executing tool: {self.name} with parameters: {kwargs}") 