#!/usr/bin/env python
"""
Helper module to load API keys from the local config.json file.

This module is used by example scripts to load the API key without
modifying the core midpoint package.
"""

import json
import os
from pathlib import Path
from typing import Tuple, Optional

def load_api_key_from_local_config() -> Tuple[bool, Optional[str]]:
    """
    Load the OpenAI API key from the local config.json file.
    
    Returns:
        Tuple[bool, Optional[str]]: (Success, API Key)
    """
    local_config_path = Path("config.json")
    
    if not local_config_path.exists():
        print("Local config.json not found.")
        print("Please run setup_config.py to set up your API key.")
        return False, None
    
    try:
        with open(local_config_path, 'r') as f:
            config = json.load(f)
        
        api_key = config.get('openai', {}).get('api_key')
        if not api_key:
            print("No API key found in local config.json.")
            return False, None
        
        # Set environment variable for immediate use
        os.environ["OPENAI_API_KEY"] = api_key
        
        return True, api_key
    except Exception as e:
        print(f"Error loading API key from local config: {str(e)}")
        return False, None

def get_points_budget() -> dict:
    """
    Get the points budget from the local config.json file.
    
    Returns:
        dict: Points budget configuration
    """
    local_config_path = Path("config.json")
    
    if not local_config_path.exists():
        return {
            "planning": 1000,
            "execution": 2000,
            "validation": 500,
            "analysis": 500
        }
    
    try:
        with open(local_config_path, 'r') as f:
            config = json.load(f)
        
        return config.get('points_budget', {})
    except Exception:
        return {
            "planning": 1000,
            "execution": 2000,
            "validation": 500,
            "analysis": 500
        }

if __name__ == "__main__":
    # If run directly, try loading the API key and print status
    success, api_key = load_api_key_from_local_config()
    
    if success:
        # Mask the API key for display
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        print(f"API key loaded successfully: {masked_key}")
        print("Environment variable OPENAI_API_KEY has been set for this session.")
    else:
        print("Failed to load API key.")
        print("Please run setup_config.py to set up your API key.") 