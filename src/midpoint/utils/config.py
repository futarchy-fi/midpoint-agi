"""
Configuration utilities for Midpoint.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

def get_config_dir() -> Path:
    """
    Get the directory containing configuration files.
    
    Returns:
        Path: Path to the configuration directory
    """
    # Check if we're running from the installed package or from the source directory
    src_dir = Path(__file__).parent.parent
    config_dir = src_dir / "config"
    
    if not config_dir.exists():
        # Fallback to project root
        config_dir = Path(__file__).parent.parent.parent.parent / "src" / "midpoint" / "config"
    
    return config_dir

def load_config(config_name: str = "default.json") -> Dict[str, Any]:
    """
    Load a configuration file.
    
    Args:
        config_name (str): Name of the config file to load. Defaults to "default.json".
        
    Returns:
        Dict[str, Any]: Configuration data
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    config_path = get_config_dir() / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f) 