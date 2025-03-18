"""
Configuration management for the Midpoint system.

This module handles loading and managing configuration from various sources:
1. Local config file (~/.midpoint/config.json)
2. Environment variables
3. Default values
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

CONFIG_DIR = Path.home() / ".midpoint"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "points_budget": {
        "planning": 1000,
        "execution": 2000,
        "validation": 500,
        "analysis": 500
    }
}

def load_config() -> Dict[str, Any]:
    """Load configuration from all sources."""
    config = DEFAULT_CONFIG.copy()
    
    # Load from config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"Warning: Error loading config file: {e}")
    
    # Load from environment variables
    env_mapping = {
        "OPENAI_API_KEY": ("openai", "api_key"),
        "OPENAI_ORG_ID": ("openai", "org_id"),
        "OPENAI_PROJECT_ID": ("openai", "project_id"),
        "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
        "POINTS_BUDGET_PLANNING": ("points_budget", "planning"),
        "POINTS_BUDGET_EXECUTION": ("points_budget", "execution"),
        "POINTS_BUDGET_VALIDATION": ("points_budget", "validation"),
        "POINTS_BUDGET_ANALYSIS": ("points_budget", "analysis")
    }
    
    for env_var, (section, key) in env_mapping.items():
        if value := os.environ.get(env_var):
            if section not in config:
                config[section] = {}
            try:
                # Convert string values to int for budget settings
                if section == "points_budget":
                    value = int(value)
                config[section][key] = value
            except ValueError:
                print(f"Warning: Invalid value for {env_var}: {value}")
    
    return config

def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to the config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def get_openai_api_key() -> Optional[str]:
    """Get the OpenAI API key from config or environment."""
    config = load_config()
    # First try environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Then try config file
        api_key = config.get("openai", {}).get("api_key")
    if not api_key:
        print("Warning: OpenAI API key not found. Please set it in your .env file or ~/.midpoint/config.json")
    return api_key

def get_anthropic_api_key() -> Optional[str]:
    """Get the Anthropic API key from config or environment."""
    config = load_config()
    api_key = config.get("anthropic", {}).get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: Anthropic API key not found. Please set it in your .env file or ~/.midpoint/config.json")
    return api_key

def set_api_key(provider: str, api_key: str, org_id: Optional[str] = None, project_id: Optional[str] = None) -> None:
    """Set the API key and optional organization/project IDs for a specific provider in the config file."""
    if provider not in ["openai", "anthropic"]:
        raise ValueError(f"Unsupported provider: {provider}")
    
    config = load_config()
    if provider not in config:
        config[provider] = {}
    config[provider]["api_key"] = api_key
    if org_id:
        config[provider]["org_id"] = org_id
    if project_id:
        config[provider]["project_id"] = project_id
    save_config(config)

def get_points_budget() -> Dict[str, int]:
    """Get the points budget configuration."""
    config = load_config()
    return config.get("points_budget", DEFAULT_CONFIG["points_budget"]) 