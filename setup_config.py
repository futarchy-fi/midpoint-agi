#!/usr/bin/env python
"""
Script to copy the config.json file from ~/.midpoint to the local repository.

This ensures the API key is available for local development but won't be committed to git.
"""

import json
import os
import shutil
import sys
from pathlib import Path

def setup_config():
    """Copy config.json from ~/.midpoint to the local repository."""
    # Define paths
    home_config_path = Path.home() / ".midpoint" / "config.json"
    local_config_path = Path("config.json")
    gitignore_path = Path(".gitignore")
    
    # Check if ~/.midpoint/config.json exists
    if not home_config_path.exists():
        print(f"Error: Config file not found at {home_config_path}")
        print("Please make sure you have set up your API key in ~/.midpoint/config.json")
        return False
    
    # Read the config file
    try:
        with open(home_config_path, "r") as f:
            config = json.load(f)
        
        # Verify the API key exists
        api_key = config.get("openai", {}).get("api_key")
        if not api_key:
            print("Error: No OpenAI API key found in config.json")
            return False
        
        # Mask the API key for display
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        print(f"Found API key: {masked_key}")
        
        # Copy the config file to the local repository
        with open(local_config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Copied config.json to {local_config_path}")
        
        # Ensure config.json is in .gitignore
        add_to_gitignore(gitignore_path, "config.json")
        
        # Set environment variable for immediate use
        os.environ["OPENAI_API_KEY"] = api_key
        print("Set OPENAI_API_KEY environment variable for current session")
        
        return True
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def add_to_gitignore(gitignore_path, pattern):
    """Add a pattern to .gitignore if not already present."""
    # Read existing gitignore
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []
    
    # Check if pattern is already in gitignore
    if any(line.strip() == pattern for line in lines):
        print(f"{pattern} is already in .gitignore")
        return
    
    # Add pattern to gitignore
    with open(gitignore_path, "a") as f:
        # Add a newline if the file doesn't end with one
        if lines and not lines[-1].endswith("\n"):
            f.write("\n")
        
        # Add comment and pattern
        f.write(f"\n# Local config file with API keys\n{pattern}\n")
    
    print(f"Added {pattern} to .gitignore")

def check_config_setup():
    """Check if the config is properly set up."""
    local_config_path = Path("config.json")
    
    if not local_config_path.exists():
        print("Local config.json not found. Running setup...")
        return setup_config()
    
    # Check if config.json contains a valid API key
    try:
        with open(local_config_path, "r") as f:
            config = json.load(f)
        
        api_key = config.get("openai", {}).get("api_key")
        if not api_key:
            print("No API key found in local config.json. Re-running setup...")
            return setup_config()
        
        # Mask the API key for display
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        print(f"Found existing API key in local config.json: {masked_key}")
        
        # Set environment variable for immediate use
        os.environ["OPENAI_API_KEY"] = api_key
        print("Set OPENAI_API_KEY environment variable for current session")
        
        return True
    
    except Exception as e:
        print(f"Error reading local config.json: {str(e)}")
        print("Re-running setup...")
        return setup_config()

if __name__ == "__main__":
    print("Setting up config.json from ~/.midpoint...")
    success = check_config_setup()
    
    if success:
        print("\nConfig setup successful!")
        print("You can now run examples without setting OPENAI_API_KEY manually.")
        
        # Instructions for using the config
        print("\nTo use the config in your code:")
        print("1. Import the config function:")
        print("   from midpoint.agents.config import get_openai_api_key")
        print("2. Get the API key:")
        print("   api_key = get_openai_api_key()")
        print("3. Or load from the config file:")
        print("   from setup_config import load_api_key_from_local_config")
        print("   success, api_key = load_api_key_from_local_config()")
        
        sys.exit(0)
    else:
        print("\nConfig setup failed.")
        print("Please ensure ~/.midpoint/config.json exists with a valid API key.")
        sys.exit(1) 