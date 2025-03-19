#!/usr/bin/env python3
"""Run a single test with the current configuration."""

import os
import sys
from midpoint.agents.config import get_openai_api_key

def main():
    """Main function."""
    api_key = get_openai_api_key()
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment")
        sys.exit(1)
        
    # Run the test
    os.system("python -m pytest " + " ".join(sys.argv[1:]))

if __name__ == "__main__":
    main() 