#!/usr/bin/env python
"""
Initialize a memory repository for Midpoint.

This script creates or connects to a memory repository for storing agent memory.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

def init_memory_repo(repo_path, remote_url=None, branch=None):
    """Initialize a memory repository at the given path."""
    repo_path = Path(repo_path)
    
    # Create directory if it doesn't exist
    if not repo_path.exists():
        print(f"Creating directory {repo_path}")
        repo_path.mkdir(parents=True)
    
    # Check if it's already a git repository
    git_dir = repo_path / ".git"
    if git_dir.exists():
        print(f"Repository already exists at {repo_path}")
        # Check for untracked files
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout:
            untracked_files = [line[3:] for line in result.stdout.splitlines() if line.startswith("??")]
            if untracked_files:
                error_msg = f"Cannot proceed: Found untracked files in memory repository:\n{chr(10).join(untracked_files)}\nPlease commit or remove these files before proceeding."
                raise RuntimeError(error_msg)
    else:
        # Initialize git repository
        print(f"Initializing git repository at {repo_path}")
        subprocess.run(["git", "init"], cwd=repo_path, check=True)
    
    # Create basic structure
    for directory in ["documents", "metadata"]:
        dir_path = repo_path / directory
        if not dir_path.exists():
            print(f"Creating directory {dir_path}")
            dir_path.mkdir(exist_ok=True)
    
    # Create cross-reference file if it doesn't exist
    cross_ref_path = repo_path / "metadata" / "cross-reference.json"
    if not cross_ref_path.exists():
        print(f"Creating cross-reference file at {cross_ref_path}")
        with open(cross_ref_path, "w") as f:
            json.dump({}, f, indent=2)
    
    # Create .gitignore file to exclude cross-reference.json
    gitignore_path = repo_path / ".gitignore"
    if not gitignore_path.exists():
        print(f"Creating .gitignore file to exclude cross-reference.json")
        with open(gitignore_path, "w") as f:
            f.write("# Ignore cross-reference file which changes frequently\n")
            f.write("metadata/cross-reference.json\n")
    
    # Add files to git
    print("Adding files to git")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    
    # Create initial commit if needed
    try:
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        print("Repository already has commits")
    except subprocess.CalledProcessError:
        print("Creating initial commit")
        subprocess.run(
            ["git", "commit", "-m", "Initialize memory repository"],
            cwd=repo_path,
            check=True
        )
    
    # Add remote if specified
    if remote_url:
        print(f"Adding remote: {remote_url}")
        try:
            # Check if remote exists
            result = subprocess.run(
                ["git", "remote"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            if "origin" in result.stdout:
                # Update existing remote
                subprocess.run(
                    ["git", "remote", "set-url", "origin", remote_url],
                    cwd=repo_path,
                    check=True
                )
            else:
                # Add new remote
                subprocess.run(
                    ["git", "remote", "add", "origin", remote_url],
                    cwd=repo_path,
                    check=True
                )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to add remote: {e}")
    
    # Create branch if specified
    if branch and branch != "main":
        print(f"Creating branch: {branch}")
        try:
            # Check if branch exists
            result = subprocess.run(
                ["git", "branch", "--list", branch],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            if branch not in result.stdout:
                # Create branch
                subprocess.run(
                    ["git", "checkout", "-b", branch],
                    cwd=repo_path,
                    check=True
                )
            else:
                # Checkout existing branch
                subprocess.run(
                    ["git", "checkout", branch],
                    cwd=repo_path,
                    check=True
                )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to create/checkout branch: {e}")
    
    print(f"Memory repository initialized at {repo_path}")
    
    # Return basic info about the repository
    return {
        "path": str(repo_path),
        "remote": remote_url,
        "branch": branch or "main"
    }

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Initialize a memory repository")
    parser.add_argument(
        "--path",
        default=os.getenv("MEMORY_REPO_PATH", os.path.expanduser("~/.midpoint/memory")),
        help="Path to memory repository"
    )
    parser.add_argument(
        "--remote",
        default=os.getenv("MEMORY_REPO_URL", ""),
        help="Remote URL for memory repository"
    )
    parser.add_argument(
        "--branch",
        default=os.getenv("MEMORY_REPO_BRANCH", "main"),
        help="Branch to use"
    )
    args = parser.parse_args()
    
    # Initialize repository
    repo_info = init_memory_repo(args.path, args.remote, args.branch)
    
    # Print repository info
    print("\nRepository information:")
    print(f"  Path: {repo_info['path']}")
    print(f"  Branch: {repo_info['branch']}")
    if repo_info['remote']:
        print(f"  Remote: {repo_info['remote']}")
    else:
        print("  Remote: None")

if __name__ == "__main__":
    main() 