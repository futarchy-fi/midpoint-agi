#!/usr/bin/env python3
"""
Test runner that checks for virtual environment and provides helpful suggestions.
"""

import sys
import os
from pathlib import Path
import subprocess
import shutil

def get_venv_path() -> str:
    """Get the path of the current virtual environment."""
    if hasattr(sys, 'real_prefix'):
        return sys.real_prefix
    elif hasattr(sys, 'base_prefix'):
        return sys.base_prefix
    return None

def is_in_project_virtualenv() -> bool:
    """Check if running in the project's virtual environment."""
    if not (hasattr(sys, 'real_prefix') or hasattr(sys, 'base_prefix')):
        return False
    
    # Get the current working directory
    cwd = Path.cwd()
    
    # Get the virtual environment path
    venv_path = Path(sys.prefix)
    
    # Check if the venv is in a subdirectory of the current project
    try:
        venv_path.relative_to(cwd)
        return True
    except ValueError:
        return False

def cleanup_old_venvs():
    """Clean up any old/incorrect virtual environments."""
    old_venvs = ['venv']  # List of old venv names to clean up
    for venv in old_venvs:
        if Path(venv).exists():
            try:
                shutil.rmtree(venv)
                print(f"\033[1;33mCleaned up old virtual environment: {venv}\033[0m")
            except Exception as e:
                print(f"\033[1;31mFailed to clean up {venv}: {str(e)}\033[0m")

def setup_virtualenv():
    """Set up the virtual environment if it doesn't exist."""
    venv_dir = '.venv'
    if not Path(venv_dir).exists():
        print(f"\033[1;33mSetting up virtual environment in {venv_dir}...\033[0m")
        try:
            subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)
            print("\033[1;32m✓ Virtual environment created successfully\033[0m")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\033[1;31mFailed to create virtual environment: {str(e)}\033[0m")
            return False
    return True

def get_venv_python():
    """Get the Python executable path in the virtual environment."""
    if os.name == 'nt':  # Windows
        return '.venv/Scripts/python.exe'
    return '.venv/bin/python'

def get_venv_activate():
    """Get the activation script path for the virtual environment."""
    if os.name == 'nt':  # Windows
        return '.venv\\Scripts\\activate.bat'
    return 'source .venv/bin/activate'

def check_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    # Clean up any old virtual environments
    cleanup_old_venvs()

    # Check if we're in the project's virtual environment
    if not is_in_project_virtualenv():
        # Try to set up the virtual environment if it doesn't exist
        if setup_virtualenv():
            print("\n\033[1;33mTo activate the virtual environment and install dependencies:\033[0m")
            print(f"\033[1;32m1. {get_venv_activate()}\033[0m")
            print("\033[1;32m2. pip install -e .\033[0m")
            print("\033[1;32m3. pip install pytest pytest-asyncio\033[0m")
            print("\nThen run this script again.")
            sys.exit(1)
        else:
            print("\033[1;31mFailed to set up virtual environment.\033[0m")
            sys.exit(1)

    # Check if midpoint package is installed
    if not check_package_installed('midpoint'):
        print("\033[1;31mError: midpoint package not installed!\033[0m")
        print("\nTo install the package in development mode, run:")
        print("\033[1;32mpip install -e .\033[0m")
        sys.exit(1)

    # Check if pytest is installed
    if not check_package_installed('pytest'):
        print("\033[1;31mError: pytest not installed!\033[0m")
        print("\nTo install pytest, run:")
        print("\033[1;32mpip install pytest pytest-asyncio\033[0m")
        sys.exit(1)

    # Run pytest with the provided arguments
    test_args = sys.argv[1:] if len(sys.argv) > 1 else ['tests/']
    cmd = ['pytest', '-v'] + test_args
    
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\033[1;31mError: pytest command not found!\033[0m")
        print("\nTo install pytest, run:")
        print("\033[1;32mpip install pytest pytest-asyncio\033[0m")
        sys.exit(1)

if __name__ == '__main__':
    main() 