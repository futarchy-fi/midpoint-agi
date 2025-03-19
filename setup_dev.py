#!/usr/bin/env python
"""Setup script for midpoint development environment."""
import os
import subprocess
import sys
import json
from pathlib import Path

def get_existing_api_key():
    """Check for existing API key in config file."""
    config_path = Path.home() / ".midpoint" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('openai', {}).get('api_key')
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass
    return None

def main():
    """Set up the development environment."""
    # Create and activate virtual environment
    if not os.path.exists(".venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    
    # Get the path to activate script
    if sys.platform == "win32":
        activate_script = ".venv\\Scripts\\activate"
        pip_cmd = [".venv\\Scripts\\pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
        pip_install_cmd = [".venv\\Scripts\\pip", "install", "-e", ".[dev]"]
        python_exe = ".venv\\Scripts\\python"
    else:
        activate_script = ".venv/bin/activate"
        pip_cmd = [".venv/bin/pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
        pip_install_cmd = [".venv/bin/pip", "install", "-e", ".[dev]"]
        python_exe = ".venv/bin/python"
    
    # Upgrade pip and other build tools
    print("Upgrading pip, setuptools, and wheel...")
    subprocess.run(pip_cmd, check=True)
    
    # Install dependencies
    print("Installing development dependencies...")
    subprocess.run(pip_install_cmd, check=True)
    
    print("\nSetup complete! To activate the environment:")
    print(f"source {activate_script}" if sys.platform != "win32" else activate_script)
    
    # Ask about creating test repo
    setup_test_repo = input("\nCreate test repository now? (y/n): ")
    if setup_test_repo.lower() == "y":
        subprocess.run([python_exe, "examples/setup_test_repo.py"])
        
    # Check for existing API key
    existing_key = get_existing_api_key()
    if existing_key:
        print("\nFound existing API key in config file. No need to set it up again.")
    else:
        # Ask about setting up OpenAI API key
        setup_api_key = input("\nSet up OpenAI API key now? (y/n): ")
        if setup_api_key.lower() == "y":
            api_key = input("Enter your OpenAI API key: ")
            
            # Create config directory if it doesn't exist
            config_dir = Path.home() / ".midpoint"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create or update config file
            config_path = config_dir / "config.json"
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Update API key
            if 'openai' not in config:
                config['openai'] = {}
            config['openai']['api_key'] = api_key
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("API key saved to config file.")
    
    print("\nDevelopment environment setup complete!")
    print("\nNext steps:")
    print("1. Activate your virtual environment")
    print("2. Try running the test script: python examples/test_goal_decomposer.py")

if __name__ == "__main__":
    main() 