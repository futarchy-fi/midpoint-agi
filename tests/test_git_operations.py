"""
Tests for git operations.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import asyncio
from agents.tools import (
    check_repo_state,
    create_branch,
    revert_to_hash,
    create_commit,
    get_current_hash,
    get_current_branch,
    checkout_branch
)

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
async def test_check_repo_state(temp_repo):
    """Test repository state checking."""
    # Check clean state
    state = await check_repo_state(str(temp_repo))
    assert state["is_clean"]
    assert not state["has_uncommitted"]
    assert not state["has_untracked"]
    
    # Create untracked file
    (temp_repo / "untracked.txt").write_text("untracked")
    state = await check_repo_state(str(temp_repo))
    assert not state["is_clean"]
    assert state["has_untracked"]
    
    # Create uncommitted changes
    (temp_repo / "test.txt").write_text("modified")
    state = await check_repo_state(str(temp_repo))
    assert not state["is_clean"]
    assert state["has_uncommitted"]

@pytest.mark.asyncio
async def test_branch_operations(temp_repo):
    """Test branch creation and management."""
    # Create new branch
    branch_name = await create_branch(str(temp_repo), "test")
    assert branch_name.startswith("test-")
    assert len(branch_name) > len("test-") + 6  # Check random suffix length
    
    # Verify we're on new branch
    current_branch = await get_current_branch(str(temp_repo))
    assert current_branch == branch_name
    
    # Checkout back to main
    await checkout_branch(str(temp_repo), "main")
    current_branch = await get_current_branch(str(temp_repo))
    assert current_branch == "main"

@pytest.mark.asyncio
async def test_hash_operations(temp_repo):
    """Test hash-based operations."""
    # Get initial hash
    initial_hash = await get_current_hash(str(temp_repo))
    
    # Create new commit
    (temp_repo / "test.txt").write_text("modified")
    new_hash = await create_commit(str(temp_repo), "Test commit")
    assert new_hash != initial_hash
    
    # Revert to initial hash
    await revert_to_hash(str(temp_repo), initial_hash)
    current_hash = await get_current_hash(str(temp_repo))
    assert current_hash == initial_hash
    
    # Verify file content reverted
    assert (temp_repo / "test.txt").read_text() == "test"

@pytest.mark.asyncio
async def test_error_handling(temp_repo):
    """Test error handling in git operations."""
    # Test non-existent branch
    with pytest.raises(Exception):
        await checkout_branch(str(temp_repo), "nonexistent")
        
    # Test non-existent hash
    with pytest.raises(Exception):
        await revert_to_hash(str(temp_repo), "nonexistent")
        
    # Test invalid repo path
    with pytest.raises(Exception):
        await check_repo_state("/nonexistent/path") 