"""
Git operations for the Midpoint system.

This module provides safe git operations using the RepoContext for state management
and safety checks.
"""

from typing import Optional, Dict, Any, Callable
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime
import asyncio
import functools
import random
import string
from .repo_context import RepoContext, GitError

def function_tool(func: Callable) -> Callable:
    """Decorator to mark a function as a tool.
    
    This is a placeholder for the actual function_tool decorator from OpenAI.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    
    # Add metadata to the function
    wrapper.is_tool = True
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    
    return wrapper

def _generate_random_suffix(length: int = 6) -> str:
    """Generate a random suffix for branch names."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

@function_tool
async def check_repo_state(repo_path: str) -> Dict[str, bool]:
    """Check repository state for uncommitted changes and untracked files."""
    async with RepoContext(repo_path) as repo:
        # Check for uncommitted changes
        status = await repo._run_git("status", "--porcelain")
        has_uncommitted = bool(status)
        
        # Check for untracked files
        has_untracked = any(line.startswith("??") for line in status.splitlines())
        
        # Check for other git problems
        has_merge_conflicts = "UU" in status
        has_rebase_conflicts = (repo.repo_path / ".git" / "rebase-apply").exists()
        
        return {
            "has_uncommitted": has_uncommitted,
            "has_untracked": has_untracked,
            "has_merge_conflicts": has_merge_conflicts,
            "has_rebase_conflicts": has_rebase_conflicts,
            "is_clean": not any([
                has_uncommitted,
                has_untracked,
                has_merge_conflicts,
                has_rebase_conflicts
            ])
        }

@function_tool
async def create_branch(repo_path: str, base_name: str) -> str:
    """Create a new branch with random suffix."""
    async with RepoContext(repo_path) as repo:
        # Generate branch name with random suffix
        branch_name = f"{base_name}-{_generate_random_suffix()}"
        
        # Create and checkout new branch
        await repo._run_git("checkout", "-b", branch_name)
        
        return branch_name

@function_tool
async def revert_to_hash(repo_path: str, hash: str) -> None:
    """Revert repository to specific hash."""
    async with RepoContext(repo_path) as repo:
        # Check if hash exists
        try:
            await repo._run_git("rev-parse", "--verify", hash)
        except GitError:
            raise GitError(f"Hash {hash} does not exist in repository")
            
        # Hard reset to hash
        await repo._run_git("reset", "--hard", hash)

@function_tool
async def create_commit(repo_path: str, message: str) -> str:
    """Create a commit and return its hash."""
    async with RepoContext(repo_path) as repo:
        # Stage all changes
        await repo._run_git("add", ".")
        
        # Create commit
        await repo._run_git("commit", "-m", message)
        
        # Get commit hash
        return (await repo._run_git("rev-parse", "HEAD")).strip()

@function_tool
async def get_current_hash(repo_path: str) -> str:
    """Get current commit hash."""
    async with RepoContext(repo_path) as repo:
        return (await repo._run_git("rev-parse", "HEAD")).strip()

@function_tool
async def get_current_branch(repo_path: str) -> str:
    """Get current branch name."""
    async with RepoContext(repo_path) as repo:
        return (await repo._run_git("rev-parse", "--abbrev-ref", "HEAD")).strip()

@function_tool
async def checkout_branch(repo_path: str, branch_name: str) -> None:
    """Checkout a specific branch."""
    async with RepoContext(repo_path) as repo:
        # Check if branch exists
        try:
            await repo._run_git("rev-parse", "--verify", branch_name)
        except GitError:
            raise GitError(f"Branch {branch_name} does not exist")
            
        await repo._run_git("checkout", branch_name)

@function_tool
async def git_commit(message: str) -> str:
    """Commit changes to git with a descriptive message"""
    try:
        # Stage all changes
        process = await asyncio.create_subprocess_exec(
            "git", "add", ".",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return f"Error staging changes: {stderr.decode()}"
        
        # Commit with message
        process = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return f"Error committing changes: {stderr.decode()}"
        
        # Get the commit hash
        process = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return f"Error getting commit hash: {stderr.decode()}"
        
        return stdout.decode().strip()
    except Exception as e:
        return f"Error committing changes: {str(e)}"

@function_tool
async def git_checkout(hash: str) -> str:
    """Checkout a specific git commit"""
    try:
        process = await asyncio.create_subprocess_exec(
            "git", "checkout", hash,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return f"Error checking out commit: {stderr.decode()}"
        
        return f"Successfully checked out {hash}"
    except Exception as e:
        return f"Error checking out commit: {str(e)}"

@function_tool
async def read_file(file_path: str) -> str:
    """Read the contents of a file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@function_tool
async def write_file(file_path: str, content: str) -> str:
    """Write content to a file"""
    try:
        # Create parent directories if they don't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@function_tool
async def run_command(command: str) -> str:
    """Run a shell command and return its output"""
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return f"Error running command: {stderr.decode()}"
        
        return stdout.decode()
    except Exception as e:
        return f"Error running command: {str(e)}"

@function_tool
async def list_directory(path: str) -> str:
    """List contents of a directory"""
    try:
        items = os.listdir(path)
        return json.dumps(items, indent=2)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

# Points tracking tool
@function_tool
async def track_points(action: str, points: int) -> Dict[str, Any]:
    """Track points consumed by an action"""
    return {
        "action": action,
        "points_consumed": points,
        "timestamp": str(datetime.now())
    } 