import os
import sys
import pytest
from pathlib import Path

def pytest_configure(config):
    """Configure pytest with environment checks."""
    # Skip checks if explicitly requested
    if config.getoption("--no-env-check", False):
        return
        
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent.parent.absolute()
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        pytest.exit("Tests must be run within a virtual environment. Please activate the correct virtual environment first:\n"
                   "1. cd " + str(project_root) + "\n"
                   "2. source .venv/bin/activate  # On Unix/macOS\n"
                   "   # or\n"
                   "   .venv\\Scripts\\activate  # On Windows\n")
    
    # Check if we're in the correct virtual environment
    venv_path = os.path.join(project_root, ".venv")
    current_venv = sys.prefix
    
    if not current_venv.startswith(str(project_root)):
        pytest.exit(f"Tests must be run from the project's virtual environment at {venv_path}.\n"
                   f"Current virtual environment: {current_venv}\n"
                   "Please activate the correct virtual environment:\n"
                   "1. cd " + str(project_root) + "\n"
                   "2. source .venv/bin/activate  # On Unix/macOS\n"
                   "   # or\n"
                   "   .venv\\Scripts\\activate  # On Windows\n")

def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--no-env-check",
        action="store_true",
        default=False,
        help="Skip virtual environment checks"
    ) 