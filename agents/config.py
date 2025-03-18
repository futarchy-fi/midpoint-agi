"""
Configuration management for the Midpoint system.

This module handles loading and managing configuration from:
1. Environment variables (loaded from .env file if present)
2. Default values
"""

import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

DEFAULT_CONFIG = {
    "points_budget": {
        "planning": 1000,
        "execution": 2000,
        "validation": 500,
        "analysis": 500
    }
}

def validate_api_key(api_key: str) -> bool:
    """Validate API key format."""
    # OpenAI API keys start with 'sk-' and are 51 characters long
    return bool(re.match(r'^sk-[A-Za-z0-9]{48}$', api_key))

def get_openai_api_key() -> str:
    """Get the OpenAI API key from environment.
    
    Returns:
        str: The OpenAI API key.
        
    Raises:
        ValueError: If API key is not found or invalid.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "\nOpenAI API key not found!\n\n"
            "To configure your API key:\n"
            "1. Copy .env.example to .env:\n"
            "   cp .env.example .env\n\n"
            "2. Edit .env and add your API key:\n"
            "   OPENAI_API_KEY=your-api-key\n\n"
            "Or set it directly in your environment:\n"
            "   export OPENAI_API_KEY=your-api-key"
        )
    
    if not validate_api_key(api_key):
        raise ValueError(
            "\nInvalid OpenAI API key format!\n\n"
            "API key should:\n"
            "- Start with 'sk-'\n"
            "- Be 51 characters long\n"
            "- Contain only letters and numbers after 'sk-'\n\n"
            "Please check your API key in .env or environment variables."
        )
    
    return api_key

def get_anthropic_api_key() -> Optional[str]:
    """Get the Anthropic API key from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "\nNote: Anthropic API key not found.\n"
            "If you plan to use Claude, set ANTHROPIC_API_KEY in your .env file."
        )
    return api_key

def validate_org_id(org_id: str) -> bool:
    """Validate OpenAI organization ID format."""
    # OpenAI org IDs are 20 characters long and contain only letters and numbers
    return bool(re.match(r'^[A-Za-z0-9]{20}$', org_id))

def get_openai_org_id() -> Optional[str]:
    """Get the OpenAI organization ID from environment.
    
    Returns:
        Optional[str]: The OpenAI organization ID if configured.
        
    Raises:
        ValueError: If organization ID is invalid.
    """
    org_id = os.environ.get("OPENAI_ORG_ID")
    if org_id and not validate_org_id(org_id):
        raise ValueError(
            "\nInvalid OpenAI organization ID format!\n\n"
            "Organization ID should:\n"
            "- Be 20 characters long\n"
            "- Contain only letters and numbers\n\n"
            "Please check your organization ID in .env or environment variables."
        )
    return org_id

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables and defaults."""
    config = DEFAULT_CONFIG.copy()
    
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

def get_points_budget() -> Dict[str, int]:
    """Get the points budget configuration."""
    config = load_config()
    return config.get("points_budget", DEFAULT_CONFIG["points_budget"]) 