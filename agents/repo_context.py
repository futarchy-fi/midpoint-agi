"""
Repository context management for safe git operations.

This module provides a safe way to perform git operations on specified repositories
by ensuring operations are scoped to the correct repository and maintaining state
for rollback capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import asyncio
from dataclasses import dataclass
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class RepoConfig:
    """Configuration for repository operations."""
    allowed_repos: List[Path]
    safe_commands: List[str] = None
    require_confirmation: bool = True
    dry_run: bool = False
    
    def __post_init__(self):
        if self.safe_commands is None:
            self.safe_commands = [
                "status", "log", "diff", "show", "rev-parse",
                "branch", "checkout", "add", "commit", "reset"
            ]

class GitError(Exception):
    """Base exception for git-related errors."""
    pass

class RepoContext:
    """Context manager for safe git operations on a specific repository."""
    
    def __init__(self, repo_path: str, config: Optional[RepoConfig] = None):
        self.repo_path = Path(repo_path).resolve()
        self.config = config or RepoConfig(allowed_repos=[self.repo_path])
        self._original_branch: Optional[str] = None
        self._original_hash: Optional[str] = None
        self._operation_log: List[Dict] = []
        
    async def _validate_repo(self) -> None:
        """Validate that the repository is safe to operate on."""
        if not self.repo_path.exists():
            raise GitError(f"Repository path does not exist: {self.repo_path}")
            
        if not (self.repo_path / ".git").exists():
            raise GitError(f"Not a git repository: {self.repo_path}")
            
        # Check if repo is in allowed list
        if self.repo_path not in self.config.allowed_repos:
            raise GitError(f"Repository not in allowed list: {self.repo_path}")
            
        # Check for dangerous states
        await self._check_repo_state()
        
    async def _check_repo_state(self) -> None:
        """Check for dangerous repository states."""
        # Check for detached HEAD
        result = await self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        if result.strip() == "HEAD":
            raise GitError("Repository is in detached HEAD state")
            
        # Check for ongoing merge/rebase
        if (self.repo_path / ".git" / "MERGE_HEAD").exists():
            raise GitError("Repository has ongoing merge")
        if (self.repo_path / ".git" / "rebase-apply").exists():
            raise GitError("Repository has ongoing rebase")
            
    async def _run_git(self, *args: str, capture_output: bool = True) -> str:
        """Run git command with safety checks."""
        # Check if command is safe
        command = args[0]
        if command not in self.config.safe_commands:
            raise GitError(f"Unsafe git command attempted: {command}")
            
        # Log the operation
        self._operation_log.append({
            "command": "git " + " ".join(args),
            "repo": str(self.repo_path),
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Build command with repo path
        cmd = ["git", "-C", str(self.repo_path)] + list(args)
        
        # Run command
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise GitError(f"Git command failed: {error_msg}")
                
            return stdout.decode() if stdout else ""
            
        except Exception as e:
            raise GitError(f"Failed to run git command: {str(e)}")
            
    async def __aenter__(self):
        """Save current state before operations."""
        await self._validate_repo()
        self._original_branch = (await self._run_git("rev-parse", "--abbrev-ref", "HEAD")).strip()
        self._original_hash = (await self._run_git("rev-parse", "HEAD")).strip()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Restore original state if something went wrong."""
        if exc_type is not None:  # If an exception occurred
            logger.warning(f"Error occurred, restoring repository state: {exc_val}")
            try:
                await self._run_git("reset", "--hard", self._original_hash)
                await self._run_git("checkout", self._original_branch)
            except Exception as e:
                logger.error(f"Failed to restore repository state: {e}")
                
    def get_operation_log(self) -> List[Dict]:
        """Get the log of git operations performed."""
        return self._operation_log.copy()
        
    def save_operation_log(self, path: str) -> None:
        """Save operation log to file."""
        with open(path, 'w') as f:
            json.dump(self._operation_log, f, indent=2) 