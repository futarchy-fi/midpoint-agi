"""
Git tools for Midpoint agents.

This module contains tools for interacting with Git repositories.
"""

import os
import subprocess
import asyncio
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
                    "description": "Path to the Git repository"
                }
            },
            "required": ["repo_path"]
        }
    
    @property
    def required_parameters(self) -> List[str]:
        return ["repo_path"]
    
    async def execute(self, repo_path: str) -> str:
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
    
    async def execute(self, repo_path: str) -> Dict[str, bool]:
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
                    "description": "Path to the Git repository"
                }
            },
            "required": ["repo_path"]
        }
    
    @property
    def required_parameters(self) -> List[str]:
        return ["repo_path"]
    
    async def execute(self, repo_path: str) -> str:
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
    
    async def execute(self, repo_path: str, branch_name: str, from_branch: Optional[str] = None) -> Dict[str, Any]:
        """Create a new Git branch."""
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        if not os.path.exists(os.path.join(repo_path, ".git")):
            raise ValueError(f"Not a Git repository: {repo_path}")
        
        # Make the branch name safe
        safe_branch_name = "".join(c if c.isalnum() or c in ['-', '_', '/'] else '_' for c in branch_name)
        
        # Check if branch exists
        process = await asyncio.create_subprocess_exec(
            "git", "branch", "--list", safe_branch_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path
        )
        
        stdout, _ = await process.communicate()
        
        if stdout.decode().strip():
            # Branch exists, append timestamp
            import time
            timestamp = int(time.time())
            safe_branch_name = f"{safe_branch_name}-{timestamp}"
        
        # Create branch command
        cmd = ["git", "checkout", "-b", safe_branch_name]
        if from_branch:
            cmd.append(from_branch)
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            raise ValueError(f"Failed to create branch: {error_msg}")
        
        return {
            "branch_name": safe_branch_name,
            "message": stdout.decode().strip()
        }

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
    
    async def execute(self, repo_path: str, message: str, add_all: bool = True) -> Dict[str, Any]:
        """Create a Git commit."""
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        if not os.path.exists(os.path.join(repo_path, ".git")):
            raise ValueError(f"Not a Git repository: {repo_path}")
        
        # Add all files if requested
        if add_all:
            process = await asyncio.create_subprocess_exec(
                "git", "add", ".",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_path
            )
            
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                raise ValueError(f"Failed to add files: {error_msg}")
        
        # Create commit
        process = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            if "nothing to commit" in error_msg:
                return {
                    "success": False,
                    "error": "Nothing to commit",
                    "hash": None
                }
            raise ValueError(f"Failed to create commit: {error_msg}")
        
        # Get the commit hash
        process = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_path
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            raise ValueError(f"Failed to get commit hash: {error_msg}")
        
        commit_hash = stdout.decode().strip()
        
        return {
            "success": True,
            "hash": commit_hash,
            "message": message
        }

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
async def get_current_hash(repo_path: str) -> str:
    """Get the current git hash of the repository."""
    return await get_current_hash_tool.execute(repo_path=repo_path)

async def check_repo_state(repo_path: str) -> Dict[str, bool]:
    """Check the current state of the git repository."""
    return await check_repo_state_tool.execute(repo_path=repo_path)

async def get_current_branch(repo_path: str) -> str:
    """Get the current git branch name."""
    return await get_current_branch_tool.execute(repo_path=repo_path)

async def create_branch(repo_path: str, branch_name: str, from_branch: Optional[str] = None) -> Dict[str, Any]:
    """Create a new Git branch."""
    return await create_branch_tool.execute(
        repo_path=repo_path, 
        branch_name=branch_name, 
        from_branch=from_branch
    )

async def create_commit(repo_path: str, message: str, add_all: bool = True) -> Dict[str, Any]:
    """Create a Git commit."""
    return await create_commit_tool.execute(
        repo_path=repo_path, 
        message=message, 
        add_all=add_all
    ) 