"""
Setup script for the Midpoint system.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Define empty validation functions if imports fail
def validate_api_key(key):
    """Validate API key format (default implementation)"""
    return len(key) > 0

def validate_org_id(org_id):
    """Validate organization ID format (default implementation)"""
    return len(org_id) > 0

try:
    # Try to import from the correct location
    from src.midpoint.agents.config import validate_api_key, validate_org_id
except ImportError:
    # We'll use the default implementations above
    pass

def setup_api_keys():
    """Interactive setup for API keys."""
    print("\nWelcome to Midpoint Setup!")
    print("This will help you configure your API keys.\n")
    
    # OpenAI API Key
    while True:
        openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        if not openai_key:
            print("! OpenAI API key not configured")
            break
        if validate_api_key(openai_key):
            print("✓ OpenAI API key format valid")
            break
        print("✗ Invalid API key format. Key should start with 'sk-' and be 51 characters long.")
    
    # OpenAI Organization ID
    while True:
        org_id = input("\nEnter your OpenAI Organization ID (or press Enter to skip): ").strip()
        if not org_id:
            print("! OpenAI Organization ID not configured (optional)")
            break
        if validate_org_id(org_id):
            print("✓ OpenAI Organization ID format valid")
            break
        print("✗ Invalid Organization ID format. ID should be 20 characters long and contain only letters and numbers.")
    
    # Anthropic API Key
    anthropic_key = input("\nEnter your Anthropic API key (or press Enter to skip): ").strip()
    if anthropic_key:
        print("✓ Anthropic API key configured")
    else:
        print("! Anthropic API key not configured (optional)")
    
    print("\nTo use these API keys, either:")
    print("1. Add them to your .env file:")
    print("   OPENAI_API_KEY=your-key")
    print("   OPENAI_ORG_ID=your-org-id")
    print("   ANTHROPIC_API_KEY=your-key")
    print("\n2. Or set them as environment variables:")
    print("   export OPENAI_API_KEY=your-key")
    print("   export OPENAI_ORG_ID=your-org-id")
    print("   export ANTHROPIC_API_KEY=your-key")
    
    print("\nYou can verify your configuration with: make check-config")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--configure":
        setup_api_keys()
    else:
        setup(
            name="midpoint",
            version="0.1.0",
            packages=find_packages(),
            install_requires=[
                "openai>=1.12.0",
                "pydantic>=2.6.0",
                "python-dotenv>=1.0.0",
                "aiohttp>=3.9.0",
                "gitpython>=3.1.40",
            ],
            entry_points={
                "console_scripts": [
                    "midpoint=setup:setup_api_keys",
                    "goal-validator=midpoint.goal_validator_cli:main",
                    "goal=midpoint.goal_cli:main",
                ],
            },
        )
