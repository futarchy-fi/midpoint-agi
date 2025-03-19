import sys
import os
import argparse
from typing import List, Optional

class EnvironmentError(Exception):
    """Custom exception for environment-related errors."""
    pass

def check_environment(force: bool = False, skip_checks: Optional[List[str]] = None) -> bool:
    """
    Check if the development environment is properly set up.
    
    Args:
        force: If True, will only warn instead of failing
        skip_checks: List of check names to skip
    
    Returns:
        bool: True if all checks pass or force=True
    """
    if skip_checks is None:
        skip_checks = []
    
    issues = []
    warnings = []
    
    def add_issue(check_name: str, message: str, is_warning: bool = False):
        if check_name not in skip_checks:
            if is_warning:
                warnings.append(f"⚠️  {message}")
            else:
                issues.append(f"❌ {message}")
    
    # Check if running in virtual environment
    if not hasattr(sys, 'real_prefix') and not hasattr(sys, 'base_prefix'):
        add_issue(
            "venv",
            "Not running in a virtual environment. Please run: source .venv/bin/activate",
            force
        )
    
    # Check if midpoint package is installed
    try:
        import midpoint
    except ImportError:
        add_issue(
            "package",
            "midpoint package not found. Please run: python setup_dev.py",
            force
        )
    
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        add_issue(
            "api_key",
            "OPENAI_API_KEY not set. Please set it in your .env file",
            force
        )
    
    # Check if running in a test directory
    if not os.path.exists("tests") or not os.path.exists("src"):
        add_issue(
            "directory",
            "Not running from project root directory",
            force
        )
    
    # Check if git is initialized
    if not os.path.exists(".git"):
        add_issue(
            "git",
            "Git repository not initialized",
            force
        )
    
    # Print issues and warnings
    if issues or warnings:
        print("\033[1;31m=== Environment Issues Found ===\033[0m")
        for issue in issues:
            print(f"\033[1;31m{issue}\033[0m")
        for warning in warnings:
            print(f"\033[1;33m{warning}\033[0m")
        
        print("\n\033[1;33mTo fix these issues:\033[0m")
        print("1. Run: source .venv/bin/activate")
        print("2. Run: python setup_dev.py")
        print("3. Set up your .env file with required API keys")
        print("\nTo force run (not recommended):")
        print("python check_env.py --force")
        print("\nTo skip specific checks:")
        print("python check_env.py --skip venv,package,api_key")
        
        if not force:
            raise EnvironmentError("Environment checks failed")
    
    print("\033[1;32m=== Environment Check Passed ===\033[0m")
    print("✅ Virtual environment active")
    print("✅ midpoint package installed")
    print("✅ Environment variables configured")
    print("✅ Project structure verified")
    return True

def main():
    parser = argparse.ArgumentParser(description="Check development environment setup")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even with environment issues (not recommended)"
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Comma-separated list of checks to skip (e.g., venv,package,api_key)"
    )
    
    args = parser.parse_args()
    
    skip_checks = args.skip.split(",") if args.skip else None
    
    try:
        check_environment(force=args.force, skip_checks=skip_checks)
    except EnvironmentError:
        sys.exit(1)

if __name__ == "__main__":
    main() 