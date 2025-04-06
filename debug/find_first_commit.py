#!/usr/bin/env python
"""
Script to find the very first commit in the memory repository.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import system get_repo_path if available
try:
    from scripts.memory_tools import get_repo_path as system_get_repo_path
    print(f"Successfully imported system_get_repo_path")
except ImportError as e:
    print(f"Error importing system_get_repo_path: {e}")
    system_get_repo_path = lambda: os.environ.get("MEMORY_REPO_PATH", os.path.join(os.path.expanduser("~"), ".midpoint", "memory"))

def find_first_commit():
    """Find the root commit(s) in the memory repository."""
    
    # Get memory repository path
    memory_repo_path = system_get_repo_path()
    
    if not memory_repo_path:
        print("WARNING: Failed to get memory repository path")
        print("Using ~/.midpoint/memory as fallback")
        memory_repo_path = os.path.expanduser("~/.midpoint/memory")
        
    print(f"Using memory repository path: {memory_repo_path}")
    
    # Verify git repository
    try:
        # Check if git repository exists
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=memory_repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Confirmed {memory_repo_path} is a git repository")
    except subprocess.CalledProcessError:
        print(f"ERROR: {memory_repo_path} is not a git repository")
        return None
    
    # Find root commit(s)
    try:
        print("Finding root commit(s)...")
        cmd = ["git", "rev-list", "--max-parents=0", "HEAD"]
        proc = subprocess.run(
            cmd,
            cwd=memory_repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        root_commits = proc.stdout.strip().split("\n")
        if not root_commits or not root_commits[0]:
            print("No root commits found")
            return None
        
        print(f"Found {len(root_commits)} root commit(s):")
        for i, commit in enumerate(root_commits):
            # Get commit details
            details_cmd = ["git", "show", "--no-patch", "--format=%h %an %ad %s", commit]
            details_proc = subprocess.run(
                details_cmd,
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            commit_details = details_proc.stdout.strip()
            print(f"{i+1}. {commit} - {commit_details}")
            
            # List files in this commit
            files_cmd = ["git", "ls-tree", "-r", "--name-only", commit]
            files_proc = subprocess.run(
                files_cmd,
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            files = [file for file in files_proc.stdout.strip().split("\n") if file.strip()]
            if not files:
                print(f"  This commit contains no files (empty repository)")
            else:
                print(f"  This commit contains {len(files)} files:")
                for file in files[:5]:  # Show first 5 files
                    print(f"    - {file}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")
            
            print()
        
        return root_commits
    except Exception as e:
        print(f"Error finding root commit: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_intermediate_commits(target_hashes):
    """Find intermediate commits between root and target hashes."""
    
    # Get memory repository path
    memory_repo_path = system_get_repo_path()
    
    if not memory_repo_path:
        print("Using ~/.midpoint/memory as fallback")
        memory_repo_path = os.path.expanduser("~/.midpoint/memory")
    
    print("\n\nChecking for 'empty-like' commits in history...")
    
    for target_hash in target_hashes:
        try:
            # Get the commit history from root to the target hash
            cmd = ["git", "log", "--reverse", "--format=%H %h %s", target_hash]
            proc = subprocess.run(
                cmd,
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = proc.stdout.strip().split("\n")
            if not commits:
                print(f"No commit history found for {target_hash}")
                continue
            
            print(f"Found {len(commits)} commits in history for {target_hash[:8]}")
            print("Checking first 10 commits for file count...")
            
            # Check first 10 commits
            for i, commit_line in enumerate(commits[:10]):
                if not commit_line.strip():
                    continue
                    
                parts = commit_line.split(' ', 2)
                if len(parts) < 2:
                    continue
                    
                commit_hash = parts[0]
                short_hash = parts[1]
                message = parts[2] if len(parts) > 2 else "No commit message"
                
                # Count files in this commit
                files_cmd = ["git", "ls-tree", "-r", "--name-only", commit_hash]
                files_proc = subprocess.run(
                    files_cmd,
                    cwd=memory_repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                files = [file for file in files_proc.stdout.strip().split("\n") if file.strip()]
                file_count = len(files)
                
                print(f"Commit {i+1}: {short_hash} - {message} - {file_count} files")
                
                # If this is a very small commit, show the files
                if 0 < file_count <= 3:
                    print(f"  This commit has just {file_count} files:")
                    for file in files:
                        print(f"    - {file}")
                    print()
            
            print("\n")
            
        except Exception as e:
            print(f"Error checking commit history: {e}")
            continue

if __name__ == "__main__":
    print("Finding the first commit in memory repository...")
    root_commits = find_first_commit()
    
    if root_commits:
        # Target hashes from previous analysis
        target_hashes = [
            "30ce45829825b9d679760ca0fd5526866e7877d5",  # G1 (reverted)
            "a00babccb37b6ddbf70d7b178bbabaca221a33e4",  # S1/T1
        ]
        
        # Check intermediate commits
        check_intermediate_commits(target_hashes)
    else:
        print("Failed to find root commit.")
        sys.exit(1) 