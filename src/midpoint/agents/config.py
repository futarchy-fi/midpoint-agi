"""
Configuration management for the Midpoint system.
"""

import os
import json
from pathlib import Path
from typing import Optional

def get_openai_api_key() -> Optional[str]:
    """Get the OpenAI API key from environment variables or config file."""
    # First check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        if not api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
        return api_key
        
    # Then check config file
    config_path = Path.home() / ".midpoint" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            api_key = config.get('openai', {}).get('api_key')
            if api_key:
                if not api_key.startswith("sk-"):
                    raise ValueError("Invalid OpenAI API key format")
                return api_key
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass
            
    return None

def get_model_name() -> str:
    """Get the OpenAI model name to use."""
    return os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

def get_max_tokens() -> int:
    """Get the maximum tokens to use for completions."""
    return int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

def get_temperature() -> float:
    """Get the temperature setting for completions."""
    return float(os.getenv("OPENAI_TEMPERATURE", "0.7")) 