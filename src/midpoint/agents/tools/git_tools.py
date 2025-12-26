"""
Git tools for Midpoint agents.

This module contains tools for interacting with Git repositories.
"""

import os
import subprocess
import random
import string
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple

from midpoint.agents.tools.base import Tool
from midpoint.agents.tools.registry import ToolRegistry

class GetCurrentHashTool(Tool):
    """Tool for getting the current Git hash."""
    
    @property
    def name(self) -> str:
        return "get_current_hash"
    
    @property
    def description(self) -> str:
        return "Gets the current Git hash of the repository"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the Git repository (optional, defaults to current directory)"
                }
            }
        }
    
    @property
    def required_parameters(self) -> List[str]:
        return []
    
    def execute(self, repo_path: Optional[str] = None) -> str:
        """Get the current git hash."""
        # If no repo_path provided, use current directory
        if not repo_path:
            repo_path = os.getcwd()
            
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            return result.stdout.decode().strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get current hash: {e.stderr.decode()}")

class CheckRepoStateTool(Tool):
    """Tool for checking the state of a Git repository."""
    
    @property
    def name(self) -> str:
        return "check_repo_state"
    
    @property
    def description(self) -> str:
        return "Checks the state of the Git repository"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the Git repository"
                }
            },
            "required": ["repo_path"]
        }
    
    @property
    def required_parameters(self) -> List[str]:
        return ["repo_path"]
    
    def execute(self, repo_path: str) -> Dict[str, bool]:
        """Check the current state of the repository."""
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
            
        # Check if it's a git repository
        if not (repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {repo_path}")
            
        # Get git status
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            status = result.stdout.decode()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git status failed: {e.stderr.decode()}")
        
        # Check for various states
        has_uncommitted = bool(status)
        has_untracked = any(line.startswith("??") for line in status.splitlines())
        
        return {
            "is_clean": not any([has_uncommitted, has_untracked]),
            "has_uncommitted": has_uncommitted,
            "has_untracked": has_untracked,
            "has_merge_conflicts": False,
            "has_rebase_conflicts": False
        }

class GetCurrentBranchTool(Tool):
    """Tool for getting the current Git branch."""
    
    @property
    def name(self) -> str:
        return "get_current_branch"
    
    @property
    def description(self) -> str:
        return "Gets the name of the current Git branch"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the Git repository (optional, defaults to current directory)"
                }
            }
        }
    
    @property
    def required_parameters(self) -> List[str]:
        return []
    
    def execute(self, repo_path: Optional[str] = None) -> str:
        """Get the current branch name."""
        # If no repo_path provided or empty string, use current directory
        if not repo_path:
            repo_path = os.getcwd()
        
        # Validate that the path exists
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            return result.stdout.decode().strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get current branch: {e.stderr.decode()}")

class CreateBranchTool(Tool):
    """Tool for creating a new Git branch."""
    
    @property
    def name(self) -> str:
        return "create_branch"
    
    @property
    def description(self) -> str:
        return "Creates a new Git branch in the repository"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the Git repository"
                },
                "branch_name": {
                    "type": "string",
                    "description": "Name of the branch to create"
                },
                "from_branch": {
                    "type": "string",
                    "description": "Name of the branch to create from (default: current branch)"
                }
            },
            "required": ["repo_path", "branch_name"]
        }
    
    def execute(self, repo_path: str, branch_name: str, from_branch: Optional[str] = None) -> Dict[str, Any]:
        """Create a new Git branch."""
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        if not os.path.exists(os.path.join(repo_path, ".git")):
            raise ValueError(f"Not a Git repository: {repo_path}")
        
        # Make the branch name safe
        safe_branch_name = "".join(c if c.isalnum() or c in ['-', '_', '/'] else '_' for c in branch_name)
        
        # Check if branch exists
        result = subprocess.run(
            ["git", "branch", "--list", safe_branch_name],
            cwd=repo_path,
            capture_output=True,
            check=False
        )
        
        if result.stdout.decode().strip():
            # Branch exists, append timestamp
            import time
            timestamp = int(time.time())
            safe_branch_name = f"{safe_branch_name}-{timestamp}"
        
        # Create branch command
        cmd = ["git", "checkout", "-b", safe_branch_name]
        if from_branch:
            cmd.append(from_branch)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            return {
                "branch_name": safe_branch_name,
                "message": result.stdout.decode().strip()
            }
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode().strip()
            raise ValueError(f"Failed to create branch: {error_msg}")

class CreateCommitTool(Tool):
    """Tool for creating a Git commit."""
    
    @property
    def name(self) -> str:
        return "create_commit"
    
    @property
    def description(self) -> str:
        return "Creates a Git commit with the given message"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the Git repository"
                },
                "message": {
                    "type": "string",
                    "description": "Commit message"
                },
                "add_all": {
                    "type": "boolean",
                    "description": "Whether to add all files before committing",
                    "default": True
                }
            },
            "required": ["repo_path", "message"]
        }
    
    def execute(self, repo_path: str, message: str, add_all: bool = True) -> Dict[str, Any]:
        """Create a Git commit."""
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        if not os.path.exists(os.path.join(repo_path, ".git")):
            raise ValueError(f"Not a Git repository: {repo_path}")
        
        # Add all files if requested
        if add_all:
            try:
                subprocess.run(
                    ["git", "add", "."],
                    cwd=repo_path,
                    capture_output=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode().strip()
                raise ValueError(f"Failed to add files: {error_msg}")
        
        # Create commit
        try:
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_path,
                capture_output=True,
                check=False  # Don't raise error for commit failure
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode().strip()
                if "nothing to commit" in error_msg:
                    return {
                        "success": False,
                        "error": "Nothing to commit",
                        "hash": None
                    }
                raise ValueError(f"Failed to create commit: {error_msg}")
            
            # Get the commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            commit_hash = hash_result.stdout.decode().strip()
            
            return {
                "success": True,
                "hash": commit_hash,
                "message": message
            }
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode().strip()
            raise ValueError(f"Failed to get commit hash: {error_msg}")

# Instantiate and register the tools
get_current_hash_tool = GetCurrentHashTool()
check_repo_state_tool = CheckRepoStateTool()
get_current_branch_tool = GetCurrentBranchTool()
create_branch_tool = CreateBranchTool()
create_commit_tool = CreateCommitTool()

ToolRegistry.register_tool(get_current_hash_tool)
ToolRegistry.register_tool(check_repo_state_tool)
ToolRegistry.register_tool(get_current_branch_tool)
ToolRegistry.register_tool(create_branch_tool)
ToolRegistry.register_tool(create_commit_tool)

# Export tool functions
def get_current_hash(repo_path: Optional[str] = None) -> str:
    """Get the current git hash of the repository.
    
    Args:
        repo_path: Optional path to the git repository. If not provided, uses current directory.
        
    Returns:
        The current git hash as a string.
    """
    return get_current_hash_tool.execute(repo_path=repo_path or os.getcwd())

def check_repo_state(repo_path: Optional[str] = None) -> Dict[str, bool]:
    """Check the current state of the git repository."""
    return check_repo_state_tool.execute(repo_path=(repo_path or os.getcwd()))

def get_current_branch(repo_path: Optional[str] = None) -> str:
    """Get the current git branch name."""
    return get_current_branch_tool.execute(repo_path=(repo_path or os.getcwd()))

def create_branch(repo_path: str, branch_name: str, from_branch: Optional[str] = None) -> Dict[str, Any]:
    """Create a new Git branch."""
    return create_branch_tool.execute(
        repo_path=repo_path, 
        branch_name=branch_name, 
        from_branch=from_branch
    )

def create_commit(repo_path: str, message: str, add_all: bool = True) -> Dict[str, Any]:
    """Create a Git commit."""
    return create_commit_tool.execute(
        repo_path=repo_path, 
        message=message, 
        add_all=add_all
    )

def get_repository_diff(repo_path: str, start_hash: str, end_hash: str, 
                        max_size: int = 50000) -> Dict[str, Any]:
    """
    Get a diff between two repository states.
    
    Args:
        repo_path: Path to the Git repository
        start_hash: Starting commit hash
        end_hash: Ending commit hash
        max_size: Maximum diff size in characters
        
    Returns:
        Dictionary containing:
        - diff_summary: Summary of changes
        - changed_files: List of files changed
        - diff_content: Actual diff content (truncated if too large)
        - truncated: Whether the diff was truncated
    """
    result = {
        "diff_summary": "",
        "changed_files": [],
        "diff_content": "",
        "truncated": False
    }
    
    try:
        # Get list of changed files
        proc = subprocess.run(
            ["git", "diff", "--name-only", start_hash, end_hash],
            cwd=repo_path,
            capture_output=True,
            check=False
        )
        
        if proc.returncode != 0:
            error_msg = proc.stderr.decode().strip()
            raise ValueError(f"Failed to get changed files: {error_msg}")
        
        result["changed_files"] = [
            file for file in proc.stdout.decode().strip().split("\n") 
            if file.strip()
        ]
        
        # Get diff summary (stats)
        proc = subprocess.run(
            ["git", "diff", "--stat", start_hash, end_hash],
            cwd=repo_path,
            capture_output=True,
            check=False
        )
        
        if proc.returncode != 0:
            error_msg = proc.stderr.decode().strip()
            raise ValueError(f"Failed to get diff stats: {error_msg}")
        
        result["diff_summary"] = proc.stdout.decode().strip()
        
        # Get complete diff
        proc = subprocess.run(
            ["git", "diff", start_hash, end_hash],
            cwd=repo_path,
            capture_output=True,
            check=False
        )
        
        if proc.returncode != 0:
            error_msg = proc.stderr.decode().strip()
            raise ValueError(f"Failed to get diff: {error_msg}")
        
        diff_content = proc.stdout.decode()
        
        # Truncate if too large
        if len(diff_content) > max_size:
            result["diff_content"] = diff_content[:max_size] + "\n[...TRUNCATED...]"
            result["truncated"] = True
        else:
            result["diff_content"] = diff_content
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error getting repository diff: {str(e)}") from e 