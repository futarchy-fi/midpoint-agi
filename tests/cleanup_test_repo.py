"""
Script to clean up the test repository by resetting any changes and ensuring it's in a clean state.
This script should be run before running tests that use the test repository.
"""

import os
import subprocess
from pathlib import Path

def cleanup_test_repo():
    """Reset the test repository to a clean state."""
    # Get the absolute path to the test repository
    repo_path = Path(__file__).parent.parent / "test-repo"
    
    if not repo_path.exists():
        print(f"Test repository not found at {repo_path}")
        return False
    
    try:
        # Change to the test repository directory
        os.chdir(repo_path)
        
        # Reset any changes
        subprocess.run(["git", "reset", "--hard"], check=True)
        
        # Clean all untracked files and directories, including .gitignored ones
        subprocess.run(["git", "clean", "-fdx"], check=True)
        
        # Ensure we're on the main branch
        subprocess.run(["git", "checkout", "main"], check=True)
        
        print("Test repository cleaned successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error cleaning test repository: {e}")
        return False
    finally:
        # Change back to the original directory
        os.chdir(Path(__file__).parent.parent)

if __name__ == "__main__":
    cleanup_test_repo() 