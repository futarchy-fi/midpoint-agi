"""
Configuration management for the Midpoint system.
"""

import os
import json
from pathlib import Path
from typing import Optional

def get_openai_api_key() -> Optional[str]:
    """Get the OpenAI API key from environment variables."""
    # Check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        if not api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
        return api_key
    return None

def get_tavily_api_key() -> Optional[str]:
    """Get the Tavily API key from environment variables."""
    return os.getenv("TAVILY_API_KEY")

def get_model_name() -> str:
    """Get the OpenAI model name to use."""
    return os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

def get_max_tokens() -> int:
    """Get the maximum tokens to use for completions."""
    return int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

def get_temperature() -> float:
    """Get the temperature setting for completions."""
    return float(os.getenv("OPENAI_TEMPERATURE", "0.7")) 