#!/usr/bin/env python3
"""
Script to run the Midpoint CLI with proper path setup.
This replaces the run_midpoint.sh script.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory where the project root is located
    project_root = Path(__file__).parent.parent.absolute()
    
    # Add the project root to PYTHONPATH
    os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    
    # Construct the path to the CLI script
    cli_path = project_root / "src" / "midpoint" / "cli.py"
    
    # Run the CLI script with all arguments passed through
    result = subprocess.run([sys.executable, str(cli_path)] + sys.argv[1:])
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 