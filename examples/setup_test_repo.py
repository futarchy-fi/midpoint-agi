"""
Script to set up a test repository for testing git operations.

This script creates a test repository with some initial content and branches
that can be used for testing the Midpoint system's git operations.
"""

import os
import subprocess
import argparse
from pathlib import Path

def setup_test_repo(repo_path: str):
    """Set up a test repository with some initial content."""
    repo_path = Path(repo_path)
    
    # Create directory if it doesn't exist
    repo_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize git repository
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    
    # Create directories first
    (repo_path / "src").mkdir(exist_ok=True)
    (repo_path / "tests").mkdir(exist_ok=True)
    
    # Create some test files
    (repo_path / "README.md").write_text("""# Test Repository

This is a test repository for Midpoint AGI. It contains:
- A simple Python application
- Test files
- Multiple branches for testing
""")
    
    (repo_path / "src" / "main.py").write_text("""def main():
    print('Hello, World!')

if __name__ == '__main__':
    main()
""")
    
    (repo_path / "tests" / "test_main.py").write_text("""def test_main():
    from src.main import main
    # Add your tests here
    assert True
""")
    
    # Add and commit files
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)
    
    # Create a feature branch
    subprocess.run(["git", "checkout", "-b", "feature/test-feature"], cwd=repo_path, check=True)
    (repo_path / "src" / "feature.py").write_text("""def new_feature():
    print('New feature!')
""")
    subprocess.run(["git", "add", "src/feature.py"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Add new feature"], cwd=repo_path, check=True)
    
    # Go back to main
    subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True)
    
    print(f"\nTest repository created at: {repo_path}")
    print("\nRepository structure:")
    print("├── README.md")
    print("├── src/")
    print("│   ├── main.py")
    print("│   └── feature.py (in feature/test-feature branch)")
    print("└── tests/")
    print("    └── test_main.py")
    
    print("\nBranches:")
    subprocess.run(["git", "branch", "-a"], cwd=repo_path)
    
    print("\nNext steps:")
    print("1. Use this repository for testing:")
    print(f"   python examples/test_repo_operations.py {repo_path} --state")
    print("2. Create new branches:")
    print(f"   python examples/test_repo_operations.py {repo_path} --branch feature/new-feature")
    print("3. When done, you can remove the repository:")
    print(f"   rm -rf {repo_path}")

def main():
    parser = argparse.ArgumentParser(description="Set up a test repository for Midpoint AGI")
    parser.add_argument("--path", help="Path where to create the test repository", 
                       default=os.path.expanduser("~/midpoint-test-repo"))
    args = parser.parse_args()
    
    setup_test_repo(args.path)

if __name__ == "__main__":
    main() 