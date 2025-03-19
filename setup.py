#!/usr/bin/env python
"""Setup script for midpoint package."""
import sys
from setuptools import setup

def check_venv():
    """Check if we're in a virtual environment."""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("\033[91mError: Installation must be performed within a virtual environment.\033[0m")
        print("\nPlease follow these steps:")
        print("1. Create a virtual environment:")
        print("   python -m venv .venv")
        print("\n2. Activate the virtual environment:")
        print("   source .venv/bin/activate  # On Unix/macOS")
        print("   # or")
        print("   .venv\\Scripts\\activate  # On Windows")
        print("\n3. Then try installing again:")
        print("   pip install -e .")
        sys.exit(1)

if __name__ == "__main__":
    check_venv()
    setup() 