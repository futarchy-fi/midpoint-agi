"""
Git operations and utility functions for the Midpoint system.
"""

import os
import asyncio
import random
import string
from pathlib import Path
from typing import Dict, Any, Optional

async def check_repo_state(repo_path: str) -> Dict[str, bool]:
    """Check the current state of the repository."""
    repo_path = Path(repo_path)
    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
        
    # Check if it's a git repository
    if not (repo_path / ".git").exists():
        raise ValueError(f"Not a git repository: {repo_path}")
        
    # Get git status
    result = await asyncio.create_subprocess_exec(
        "git", "status", "--porcelain",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Git status failed: {stderr.decode()}")
        
    status = stdout.decode()
    
    # Check for various states
    has_uncommitted = bool(status)
    has_untracked = any(line.startswith("??") for line in status.splitlines())
    
    # Check for merge conflicts
    merge_result = await asyncio.create_subprocess_exec(
        "git", "diff", "--check",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, merge_stderr = await merge_result.communicate()
    has_merge_conflicts = "conflict" in merge_stderr.decode().lower()
    
    # Check for rebase conflicts
    rebase_result = await asyncio.create_subprocess_exec(
        "git", "rebase", "--show-current-patch",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, rebase_stderr = await rebase_result.communicate()
    has_rebase_conflicts = rebase_result.returncode != 0
    
    return {
        "is_clean": not any([has_uncommitted, has_untracked, has_merge_conflicts, has_rebase_conflicts]),
        "has_uncommitted": has_uncommitted,
        "has_untracked": has_untracked,
        "has_merge_conflicts": has_merge_conflicts,
        "has_rebase_conflicts": has_rebase_conflicts
    }

async def create_branch(repo_path: str, base_name: str) -> str:
    """Create a new branch with a random suffix."""
    # Generate random suffix
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    branch_name = f"{base_name}-{suffix}"
    
    # Create and checkout branch
    result = await asyncio.create_subprocess_exec(
        "git", "checkout", "-b", branch_name,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create branch: {stderr.decode()}")
        
    return branch_name

async def revert_to_hash(repo_path: str, git_hash: str) -> None:
    """Revert the repository to a specific git hash."""
    # First check if we have uncommitted changes
    state = await check_repo_state(repo_path)
    if not state["is_clean"]:
        raise RuntimeError("Cannot revert: repository has uncommitted changes")
        
    # Hard reset to the hash
    result = await asyncio.create_subprocess_exec(
        "git", "reset", "--hard", git_hash,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to revert to hash: {stderr.decode()}")

async def create_commit(repo_path: str, message: str) -> str:
    """Create a commit with the given message."""
    # Add all changes
    add_result = await asyncio.create_subprocess_exec(
        "git", "add", ".",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, add_stderr = await add_result.communicate()
    
    if add_result.returncode != 0:
        raise RuntimeError(f"Failed to add changes: {add_stderr.decode()}")
        
    # Create commit
    commit_result = await asyncio.create_subprocess_exec(
        "git", "commit", "-m", message,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, commit_stderr = await commit_result.communicate()
    
    if commit_result.returncode != 0:
        raise RuntimeError(f"Failed to create commit: {commit_stderr.decode()}")
        
    # Get the commit hash
    hash_result = await asyncio.create_subprocess_exec(
        "git", "rev-parse", "HEAD",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, hash_stderr = await hash_result.communicate()
    
    if hash_result.returncode != 0:
        raise RuntimeError(f"Failed to get commit hash: {hash_stderr.decode()}")
        
    return stdout.decode().strip()

async def get_current_hash(repo_path: str) -> str:
    """Get the current git hash."""
    result = await asyncio.create_subprocess_exec(
        "git", "rev-parse", "HEAD",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get current hash: {stderr.decode()}")
        
    return stdout.decode().strip()

async def get_current_branch(repo_path: str) -> str:
    """Get the current branch name."""
    result = await asyncio.create_subprocess_exec(
        "git", "rev-parse", "--abbrev-ref", "HEAD",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get current branch: {stderr.decode()}")
        
    return stdout.decode().strip()

async def checkout_branch(repo_path: str, branch_name: str) -> None:
    """Checkout a specific branch."""
    # First check if we have uncommitted changes
    state = await check_repo_state(repo_path)
    if not state["is_clean"]:
        raise RuntimeError("Cannot checkout: repository has uncommitted changes")
        
    result = await asyncio.create_subprocess_exec(
        "git", "checkout", branch_name,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to checkout branch: {stderr.decode()}")

async def track_points(operation: str, points: int) -> None:
    """Track points consumed by an operation."""
    # In a real implementation, this would store points in a database
    # or other persistent storage. For now, we just print it.
    print(f"Points consumed by {operation}: {points}") 