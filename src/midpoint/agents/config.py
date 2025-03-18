"""
Configuration management for the Midpoint system.
"""

import os
from typing import Optional

def get_openai_api_key() -> Optional[str]:
    """Get the OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
        
    # Basic validation of API key format
    if not api_key.startswith("sk-"):
        raise ValueError("Invalid OpenAI API key format")
        
    return api_key

def get_model_name() -> str:
    """Get the OpenAI model name to use."""
    return os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

def get_max_tokens() -> int:
    """Get the maximum tokens to use for completions."""
    return int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

def get_temperature() -> float:
    """Get the temperature setting for completions."""
    return float(os.getenv("OPENAI_TEMPERATURE", "0.7")) 