"""
Tests for repository context safety features.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import asyncio
from agents.repo_context import RepoContext, RepoConfig, GitError

@pytest.fixture
def temp_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Initialize git repo
        asyncio.run(asyncio.create_subprocess_exec(
            "git", "init",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        ))
        
        # Create a test file
        (repo_path / "test.txt").write_text("test")
        
        # Add and commit
        asyncio.run(asyncio.create_subprocess_exec(
            "git", "add", "test.txt",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        ))
        
        asyncio.run(asyncio.create_subprocess_exec(
            "git", "commit", "-m", "Initial commit",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        ))
        
        yield repo_path
        
        # Cleanup
        shutil.rmtree(tmpdir)

@pytest.mark.asyncio
async def test_repo_validation(temp_repo):
    """Test repository validation."""
    # Valid repo should work
    async with RepoContext(str(temp_repo)) as repo:
        assert repo.repo_path == temp_repo
        
    # Non-existent repo should fail
    with pytest.raises(GitError):
        async with RepoContext("/nonexistent/repo"):
            pass
            
    # Non-git directory should fail
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(GitError):
            async with RepoContext(tmpdir):
                pass

@pytest.mark.asyncio
async def test_safe_commands(temp_repo):
    """Test safe command execution."""
    async with RepoContext(str(temp_repo)) as repo:
        # Safe command should work
        result = repo._run_git("status")
        assert "test.txt" in result
        
        # Unsafe command should fail
        with pytest.raises(GitError):
            repo._run_git("push")

@pytest.mark.asyncio
async def test_state_restoration(temp_repo):
    """Test state restoration on error."""
    original_branch = None
    original_hash = None
    
    async with RepoContext(str(temp_repo)) as repo:
        original_branch = repo._original_branch
        original_hash = repo._original_hash
        
        # Create a new branch
        repo._run_git("checkout", "-b", "test-branch")
        
        # Simulate an error
        raise Exception("Test error")
        
    # Check that we're back on original branch
    result = asyncio.run(asyncio.create_subprocess_exec(
        "git", "rev-parse", "--abbrev-ref", "HEAD",
        cwd=temp_repo,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    ))
    current_branch = result[0].decode().strip()
    assert current_branch == original_branch

@pytest.mark.asyncio
async def test_operation_logging(temp_repo):
    """Test operation logging."""
    async with RepoContext(str(temp_repo)) as repo:
        # Perform some operations
        repo._run_git("status")
        repo._run_git("log", "--oneline")
        
        # Check log
        log = repo.get_operation_log()
        assert len(log) == 2
        assert all("command" in entry for entry in log)
        assert all("repo" in entry for entry in log)
        assert all("timestamp" in entry for entry in log)
        
        # Test log saving
        with tempfile.NamedTemporaryFile() as tmp:
            repo.save_operation_log(tmp.name)
            # Verify file was created and contains log data
            assert Path(tmp.name).exists()
            log_data = Path(tmp.name).read_text()
            assert "command" in log_data
            assert "repo" in log_data
            assert "timestamp" in log_data 