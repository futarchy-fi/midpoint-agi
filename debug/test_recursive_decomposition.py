"""
Tests for the recursive goal decomposition functionality.
"""

import pytest
import pytest_asyncio
import tempfile
import shutil
import os
from pathlib import Path
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from midpoint.agents.models import State, Goal, SubgoalPlan, TaskContext
from midpoint.agents.goal_decomposer import GoalDecomposer, validate_repository_state
from midpoint.agents.tools import get_current_hash, track_points

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    old_env = dict(os.environ)
    # Use test key
    os.environ["OPENAI_API_KEY"] = "sk-" + "a" * 48
    yield
    os.environ.clear()
    os.environ.update(old_env)

@pytest_asyncio.fixture
async def temp_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Initialize git repo
        process = await asyncio.create_subprocess_exec(
            "git", "init",
            cwd=str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        # Create a test file
        (repo_path / "test.txt").write_text("test")
        
        # Add and commit
        process = await asyncio.create_subprocess_exec(
            "git", "add", "test.txt",
            cwd=str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        process = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", "Initial commit",
            cwd=str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        yield repo_path
        
        # Cleanup - force remove git directory
        git_dir = repo_path / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir, ignore_errors=True)
        shutil.rmtree(tmpdir)

@pytest.fixture
def goal_decomposer():
    """Create a GoalDecomposer instance for testing."""
    return GoalDecomposer()

@pytest.fixture
def sample_goal():
    """Create a sample goal for testing."""
    return Goal(
        description="Add a new feature to handle user authentication",
        validation_criteria=[
            "User can register with email and password",
            "User can login with credentials",
            "User can logout",
            "User session is maintained"
        ],
        success_threshold=0.8
    )

@pytest_asyncio.fixture
async def sample_context(temp_repo, sample_goal):
    """Create a sample task context for testing."""
    git_hash = await get_current_hash(str(temp_repo))
    return TaskContext(
        state=State(
            git_hash=git_hash,
            description="Initial state with test file",
            repository_path=str(temp_repo)
        ),
        goal=sample_goal,
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )

@pytest.mark.asyncio
async def test_repository_validation(temp_repo, sample_goal):
    """Test repository validation functionality."""
    # Get current hash
    git_hash = await get_current_hash(str(temp_repo))
    
    # Valid state should pass
    assert await validate_repository_state(str(temp_repo), git_hash) == True
    
    # Invalid hash should raise ValueError
    with pytest.raises(ValueError, match="Repository hash mismatch"):
        await validate_repository_state(str(temp_repo), "invalid_hash")
    
    # Create an uncommitted change
    (temp_repo / "uncommitted.txt").write_text("uncommitted")
    
    # Uncommitted changes should raise ValueError
    with pytest.raises(ValueError, match="Repository has uncommitted changes"):
        await validate_repository_state(str(temp_repo), git_hash)

@pytest.mark.asyncio
async def test_recursive_decomposition_basic(goal_decomposer, sample_context):
    """Test basic recursive decomposition functionality."""
    # Create a temporary log file
    log_file = "test_hierarchy.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Mock the determine_next_step method to return predefined results
    with patch.object(goal_decomposer, "determine_next_step", new_callable=AsyncMock) as mock_next_step:
        # First level: requires further decomposition
        first_subgoal = SubgoalPlan(
            next_step="Implement authentication system",
            validation_criteria=["Auth works", "Tests pass"],
            reasoning="Auth is needed first",
            requires_further_decomposition=True,
            relevant_context={"auth_type": "JWT"}
        )
        
        # Second level: ready for execution
        second_subgoal = SubgoalPlan(
            next_step="Implement user registration endpoint",
            validation_criteria=["Endpoint works", "Validates input"],
            reasoning="Registration is needed first",
            requires_further_decomposition=False,
            relevant_context={}
        )
        
        # Configure the mock to return different values on consecutive calls
        mock_next_step.side_effect = [
            first_subgoal,  # First call returns first_subgoal
            second_subgoal  # Second call returns second_subgoal
        ]
        
        # Call decompose_recursively
        result = await goal_decomposer.decompose_recursively(sample_context, log_file)
        
        # Verify results
        assert len(result) == 2
        assert result[0].next_step == "Implement authentication system"
        assert result[1].next_step == "Implement user registration endpoint"
        assert result[0].requires_further_decomposition == True
        assert result[1].requires_further_decomposition == False
        
        # Verify mock was called twice
        assert mock_next_step.call_count == 2
        
        # Verify log file was created
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            log_content = f.read()
            assert "Goal: Add a new feature to handle user authentication" in log_content
            assert "Subgoal: Implement authentication system" in log_content
            assert "READY FOR EXECUTION: Implement user registration endpoint" in log_content
        
        # Clean up log file
        os.remove(log_file)

@pytest.mark.asyncio
async def test_recursive_decomposition_multiple_levels(goal_decomposer, sample_context):
    """Test recursive decomposition with multiple levels."""
    # Create a temporary log file
    log_file = "test_hierarchy.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Mock the determine_next_step method to return predefined results
    with patch.object(goal_decomposer, "determine_next_step", new_callable=AsyncMock) as mock_next_step:
        # First level: requires further decomposition
        level1_subgoal = SubgoalPlan(
            next_step="Implement authentication system",
            validation_criteria=["Auth works", "Tests pass"],
            reasoning="Auth is needed first",
            requires_further_decomposition=True,
            relevant_context={"auth_type": "JWT"}
        )
        
        # Second level: requires further decomposition
        level2_subgoal = SubgoalPlan(
            next_step="Implement user management",
            validation_criteria=["User CRUD works", "Permissions work"],
            reasoning="User management is needed for auth",
            requires_further_decomposition=True,
            relevant_context={"db_type": "SQL"}
        )
        
        # Third level: ready for execution
        level3_subgoal = SubgoalPlan(
            next_step="Implement user registration endpoint",
            validation_criteria=["Endpoint works", "Validates input"],
            reasoning="Registration is needed first",
            requires_further_decomposition=False,
            relevant_context={}
        )
        
        # Configure the mock to return different values on consecutive calls
        mock_next_step.side_effect = [
            level1_subgoal,  # First call
            level2_subgoal,  # Second call
            level3_subgoal   # Third call
        ]
        
        # Call decompose_recursively
        result = await goal_decomposer.decompose_recursively(sample_context, log_file)
        
        # Verify results
        assert len(result) == 3
        assert result[0].next_step == "Implement authentication system"
        assert result[1].next_step == "Implement user management"
        assert result[2].next_step == "Implement user registration endpoint"
        assert result[0].requires_further_decomposition == True
        assert result[1].requires_further_decomposition == True
        assert result[2].requires_further_decomposition == False
        
        # Verify mock was called three times
        assert mock_next_step.call_count == 3
        
        # Verify metadata passing
        # This is passed as the second argument to the second call to determine_next_step
        second_call_args = mock_next_step.call_args_list[1][0][0]
        assert "parent_goal" in second_call_args.metadata
        assert second_call_args.metadata["parent_goal"] == sample_context.goal.description
        assert "parent_context" in second_call_args.metadata
        assert second_call_args.metadata["parent_context"]["auth_type"] == "JWT"
        
        # Verify log file was created and has proper indentation
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            log_content = f.read()
            assert "Goal: Add a new feature to handle user authentication" in log_content
            assert "└── Subgoal: Implement authentication system" in log_content
            assert "  Goal: Implement authentication system" in log_content
            assert "  └── Subgoal: Implement user management" in log_content
            assert "    Goal: Implement user management" in log_content
            assert "    └── Subgoal: Implement user registration endpoint" in log_content
            assert "      ✓ READY FOR EXECUTION: Implement user registration endpoint" in log_content
        
        # Clean up log file
        os.remove(log_file)

@pytest.mark.asyncio
async def test_decomposition_depth_calculation(goal_decomposer):
    """Test the calculation of decomposition depth."""
    # Create contexts with different depths
    root_context = TaskContext(
        state=State(git_hash="hash", description="desc", repository_path="/tmp"),
        goal=Goal(description="Root goal", validation_criteria=["Test"]),
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )
    
    # Context with depth 1
    depth1_context = TaskContext(
        state=State(git_hash="hash", description="desc", repository_path="/tmp"),
        goal=Goal(description="Level 1 goal", validation_criteria=["Test"]),
        iteration=0,
        points_consumed=0,
        total_budget=900,
        execution_history=[],
        metadata={
            "parent_goal": "Root goal",
            "parent_context": {"key": "value"}
        }
    )
    
    # Context with depth 2
    depth2_context = TaskContext(
        state=State(git_hash="hash", description="desc", repository_path="/tmp"),
        goal=Goal(description="Level 2 goal", validation_criteria=["Test"]),
        iteration=0,
        points_consumed=0,
        total_budget=800,
        execution_history=[],
        metadata={
            "parent_goal": "Level 1 goal",
            "parent_context": {
                "key": "value2",
                "parent_goal": "Root goal",
                "parent_context": {"original_key": "original_value"}
            }
        }
    )
    
    # Test depth calculations
    assert goal_decomposer._get_decomposition_depth(root_context) == 0
    assert goal_decomposer._get_decomposition_depth(depth1_context) == 1
    assert goal_decomposer._get_decomposition_depth(depth2_context) == 2 