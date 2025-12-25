"""
Configuration management for the Midpoint system.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

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

def _load_llm_config() -> Dict[str, Any]:
    """Load the LLM configuration from llm_config.json."""
    # Get config directory
    src_dir = Path(__file__).parent.parent
    config_dir = src_dir / "config"
    
    if not config_dir.exists():
        # Fallback path
        config_dir = Path(__file__).parent.parent.parent.parent / "src" / "midpoint" / "config"
    
    config_path = config_dir / "llm_config.json"
    
    if not config_path.exists():
        # Return defaults if config file doesn't exist
        return {
            "agents": {
                "task_executor": {"model": "gpt-4o-mini", "max_tokens": 3000},
                "goal_decomposer": {"model": "gpt-4o-mini", "max_tokens": 3000},
                "goal_analyzer": {"model": "gpt-4o-mini", "max_tokens": 1000},
                "goal_validator": {"model": "gpt-4o-mini", "max_tokens": 2000},
            }
        }
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific agent from llm_config.json.
    
    Args:
        agent_name: Name of the agent (e.g., "task_executor", "goal_decomposer")
        
    Returns:
        Dictionary with model and max_tokens for the agent
    """
    config = _load_llm_config()
    agents_config = config.get("agents", {})
    agent_config = agents_config.get(agent_name, {})
    
    # Return with defaults if missing
    return {
        "model": agent_config.get("model", "gpt-4o-mini"),
        "max_tokens": agent_config.get("max_tokens", 2000),
    } 