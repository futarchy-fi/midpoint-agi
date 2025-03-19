"""
Tests for the Goal Decomposition agent.
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
from midpoint.agents.models import State, Goal, StrategyPlan, SubgoalPlan, TaskContext
from midpoint.agents.goal_decomposer import GoalDecomposer
from midpoint.agents.tools import (
    check_repo_state, 
    create_branch, 
    create_commit,
    get_current_hash,
    track_points
)
from midpoint.agents.config import get_openai_api_key

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    old_env = dict(os.environ)
    # Get API key from config if available, otherwise use test key
    api_key = get_openai_api_key() or "sk-" + "a" * 48
    os.environ["OPENAI_API_KEY"] = api_key
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
async def test_next_step_determination_basic(goal_decomposer, sample_context):
    """Test basic next step determination functionality."""
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        # Mock the chat completion response
        mock_message = MagicMock()
        mock_message.content = """```json
{
  "next_step": "Implement user registration with email and password",
  "validation_criteria": [
    "User can register with a valid email and password",
    "System validates email format",
    "System enforces password strength requirements",
    "Registration endpoint returns appropriate success/error responses"
  ],
  "reasoning": "User authentication is the foundation for the system and should be implemented first. Registration is the first step in the authentication flow."
}
```"""
        mock_message.tool_calls = None
        
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=mock_message)]
        )
        
        subgoal = await goal_decomposer.determine_next_step(sample_context)
        assert subgoal.next_step == "Implement user registration with email and password"
        assert len(subgoal.validation_criteria) == 4
        assert "User authentication is the foundation" in subgoal.reasoning

@pytest.mark.asyncio
async def test_next_step_with_tools(goal_decomposer, sample_context):
    """Test next step determination with tool usage."""
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        # First response calls a tool
        first_message = MagicMock()
        first_message.content = None
        first_message.tool_calls = [
            MagicMock(
                id="call_1",
                function=MagicMock(
                    name="list_directory",
                    arguments=json.dumps({"repo_path": sample_context.state.repository_path})
                )
            )
        ]
        
        # Second response provides the final output
        second_message = MagicMock()
        second_message.content = """```json
{
  "next_step": "Set up user registration with email/password",
  "validation_criteria": [
    "Registration endpoint accepts email and password",
    "Validation for email format is implemented",
    "Password hashing is properly implemented",
    "User data is stored securely"
  ],
  "reasoning": "Based on repository exploration, user authentication functionality needs to be implemented first to enable the core features."
}
```"""
        second_message.tool_calls = None
        
        # Configure the mock to return different responses on consecutive calls
        mock_create.side_effect = [
            MagicMock(choices=[MagicMock(message=first_message)]),
            MagicMock(choices=[MagicMock(message=second_message)])
        ]
        
        # Instead of mocking list_directory, track the tool calls directly
        tracked_tools = []
        
        # Define a simple mock for track_points to capture the tool usage
        async def mock_track_points(operation, points):
            if operation == "tool_use":
                tracked_tools.append(operation)
                
        # We're patching an existing method in the goal_decomposer instance to track tool usage
        with patch("midpoint.agents.goal_decomposer.track_points", mock_track_points):
            # Also patch the actual list_directory function to return a dummy result
            with patch("midpoint.agents.goal_decomposer.list_directory", return_value={"files": [], "directories": []}):
                # And also patch read_file and search_code to be safe
                with patch("midpoint.agents.goal_decomposer.read_file", return_value="File content"):
                    with patch("midpoint.agents.goal_decomposer.search_code", return_value="Search results"):
                        subgoal = await goal_decomposer.determine_next_step(sample_context)
                        
                        # Verify tools were tracked by checking the tracked_tools list
                        assert len(tracked_tools) > 0
                        
                        # Verify tool usage is in metadata
                        assert "tool_usage" in subgoal.metadata
                        assert len(subgoal.metadata["tool_usage"]) > 0
                        assert "list_directory" in subgoal.metadata["tool_usage"][0]
                        
                        # Verify the result
                        assert subgoal.next_step == "Set up user registration with email/password"
                        assert len(subgoal.validation_criteria) == 4
                        assert "Based on repository exploration" in subgoal.reasoning

@pytest.mark.asyncio
async def test_next_step_complex_goal(goal_decomposer, temp_repo):
    """Test next step determination for a complex goal."""
    complex_goal = Goal(
        description="Implement a full-stack e-commerce system with user authentication, product catalog, shopping cart, and checkout process",
        validation_criteria=[
            "Users can browse products",
            "Users can add items to cart",
            "Users can checkout",
            "Orders are processed",
            "Inventory is updated",
            "Users receive confirmation"
        ],
        success_threshold=0.8
    )
    
    context = TaskContext(
        state=State(
            git_hash="initial",
            description="Empty repository",
            repository_path=str(temp_repo)
        ),
        goal=complex_goal,
        iteration=0,
        points_consumed=0,
        total_budget=5000,
        execution_history=[]
    )
    
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_message = MagicMock()
        mock_message.content = """```json
{
  "next_step": "Set up the basic project structure with frontend and backend directories",
  "validation_criteria": [
    "Frontend directory created with React setup",
    "Backend directory created with API scaffolding",
    "Basic routing system implemented",
    "Project can be built and run without errors"
  ],
  "reasoning": "Before implementing specific features, it's important to set up the project structure to organize the codebase properly."
}
```"""
        mock_message.tool_calls = None
        
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=mock_message)]
        )
        
        subgoal = await goal_decomposer.determine_next_step(context)
        assert subgoal.next_step == "Set up the basic project structure with frontend and backend directories"
        assert len(subgoal.validation_criteria) == 4
        assert "project structure" in subgoal.reasoning

@pytest.mark.asyncio
async def test_iteration_handling(goal_decomposer, sample_context):
    """Test handling of multiple iterations in next step determination."""
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        # First iteration
        first_message = MagicMock()
        first_message.content = """```json
{
  "next_step": "Implement user registration system",
  "validation_criteria": [
    "User registration endpoint created",
    "Input validation implemented",
    "User data stored securely",
    "Registration confirmation provided"
  ],
  "reasoning": "User registration is the first step in authentication flow"
}
```"""
        first_message.tool_calls = None
        
        # Second iteration
        second_message = MagicMock()
        second_message.content = """```json
{
  "next_step": "Implement user login functionality",
  "validation_criteria": [
    "Login endpoint created",
    "Credentials validation works",
    "Session management implemented",
    "Login tokens generated"
  ],
  "reasoning": "After registration is complete, login is the next logical step"
}
```"""
        second_message.tool_calls = None
        
        # Configure the mock for first call
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=first_message)]
        )
        
        # First iteration
        subgoal1 = await goal_decomposer.determine_next_step(sample_context)
        
        # Update context for second iteration
        sample_context.iteration = 1
        sample_context.points_consumed = 50
        sample_context.execution_history.append({"action": "Implemented user registration"})
        
        # Configure the mock for second call
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=second_message)]
        )
        
        # Second iteration
        subgoal2 = await goal_decomposer.determine_next_step(sample_context)
        
        # Verify different next steps for different iterations
        assert subgoal1.next_step != subgoal2.next_step
        assert "registration" in subgoal1.next_step.lower()
        assert "login" in subgoal2.next_step.lower()

@pytest.mark.asyncio
async def test_error_handling(goal_decomposer):
    """Test error handling in next step determination."""
    # Test with invalid goal
    invalid_context = TaskContext(
        state=State(
            git_hash="invalid", 
            description="Invalid state",
            repository_path="/tmp"
        ),
        goal=None,  # Invalid goal
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )
    
    with pytest.raises(ValueError, match="No goal provided in context"):
        await goal_decomposer.determine_next_step(invalid_context)
    
    # Test with missing repository path
    invalid_context.goal = Goal(
        description="Test",
        validation_criteria=["Test"],
        success_threshold=0.8
    )
    invalid_context.state.repository_path = None
    
    with pytest.raises(ValueError, match="Repository path not provided in state"):
        await goal_decomposer.determine_next_step(invalid_context)

@pytest.mark.asyncio
async def test_user_prompt_creation(goal_decomposer, sample_context):
    """Test user prompt creation."""
    prompt = goal_decomposer._create_user_prompt(sample_context)
    
    assert "Goal: Add a new feature to handle user authentication" in prompt
    assert "User can register with email and password" in prompt
    assert "User can login with credentials" in prompt
    assert "User can logout" in prompt
    assert "User session is maintained" in prompt
    assert "Git Hash:" in prompt
    assert "Repository Path:" in prompt
    assert "Iteration: 0" in prompt
    assert "Previous Steps: 0" in prompt

@pytest.mark.asyncio
async def test_api_error_handling(goal_decomposer, sample_context):
    """Test handling of API errors."""
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock, side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="Error during next step determination: API Error"):
            await goal_decomposer.determine_next_step(sample_context)

@pytest.mark.asyncio
async def test_initialization():
    """Test GoalDecomposer initialization and API key handling."""
    # Test successful initialization with valid key format
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-" + "a" * 48}, clear=True):
        decomposer = GoalDecomposer()
        assert decomposer.client is not None

    # Test missing API key
    with patch.dict(os.environ, {}, clear=True):
        with patch("midpoint.agents.goal_decomposer.get_openai_api_key", return_value=None):
            with pytest.raises(ValueError, match="OpenAI API key not found in config or environment"):
                GoalDecomposer()

    # Test invalid API key format
    with patch.dict(os.environ, {}, clear=True):
        with patch("midpoint.agents.goal_decomposer.get_openai_api_key", return_value="invalid-key"):
            with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
                GoalDecomposer()

@pytest.mark.asyncio
async def test_points_tracking(goal_decomposer, sample_context):
    """Test points tracking during next step determination."""
    points_tracked = []
    
    async def mock_track_points(operation: str, points: int):
        points_tracked.append((operation, points))
    
    with patch("midpoint.agents.goal_decomposer.track_points", new=mock_track_points), \
         patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        
        mock_message = MagicMock()
        mock_message.content = """```json
{
  "next_step": "Implement user registration",
  "validation_criteria": ["Test validation"],
  "reasoning": "Test reasoning"
}
```"""
        mock_message.tool_calls = None
        
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=mock_message)]
        )
        
        await goal_decomposer.determine_next_step(sample_context)
        
        assert len(points_tracked) == 1
        assert points_tracked[0][0] == "goal_decomposition"
        assert points_tracked[0][1] == 10

@pytest.mark.asyncio
async def test_tool_usage_tracking(goal_decomposer, sample_context):
    """Test tracking of tool usage in metadata."""
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        # First response calls a tool
        first_message = MagicMock()
        first_message.content = None
        first_message.tool_calls = [
            MagicMock(
                id="call_1",
                function=MagicMock(
                    name="list_directory",
                    arguments=json.dumps({"repo_path": sample_context.state.repository_path})
                )
            )
        ]
        
        # Second response provides the final output
        second_message = MagicMock()
        second_message.content = """```json
{
  "next_step": "Implement user registration",
  "validation_criteria": ["Test validation"],
  "reasoning": "Test reasoning"
}
```"""
        second_message.tool_calls = None
        
        # Configure the mock to return different responses on consecutive calls
        mock_create.side_effect = [
            MagicMock(choices=[MagicMock(message=first_message)]),
            MagicMock(choices=[MagicMock(message=second_message)])
        ]
        
        # Track tool usage events
        tracked_tools = []
        
        # Define a mock for track_points to capture tool usage
        async def mock_track_points(operation, points):
            if operation == "tool_use":
                tracked_tools.append(operation)
        
        # Use patching to mock necessary functions
        with patch("midpoint.agents.goal_decomposer.track_points", mock_track_points):
            with patch("midpoint.agents.goal_decomposer.list_directory", return_value={"files": [], "directories": []}):
                with patch("midpoint.agents.goal_decomposer.read_file", return_value="File content"):
                    with patch("midpoint.agents.goal_decomposer.search_code", return_value="Search results"):
                        
                        subgoal = await goal_decomposer.determine_next_step(sample_context)
                        
                        # Verify tools were tracked
                        assert len(tracked_tools) > 0
                        
                        # Verify tool usage is tracked in metadata
                        assert "tool_usage" in subgoal.metadata
                        assert len(subgoal.metadata["tool_usage"]) > 0
                        assert "list_directory" in subgoal.metadata["tool_usage"][0]

@pytest.mark.asyncio
async def test_subgoal_validation(goal_decomposer, sample_context):
    """Test validation of the subgoal plan."""
    # Test missing next_step
    invalid_subgoal = SubgoalPlan(
        next_step="",
        validation_criteria=["Test"],
        reasoning="Test reasoning"
    )
    with pytest.raises(ValueError, match="Subgoal has no next step defined"):
        goal_decomposer._validate_subgoal(invalid_subgoal, sample_context)
    
    # Test missing validation criteria
    invalid_subgoal = SubgoalPlan(
        next_step="Test step",
        validation_criteria=[],
        reasoning="Test reasoning"
    )
    with pytest.raises(ValueError, match="Subgoal has no validation criteria"):
        goal_decomposer._validate_subgoal(invalid_subgoal, sample_context)
    
    # Test missing reasoning
    invalid_subgoal = SubgoalPlan(
        next_step="Test step",
        validation_criteria=["Test"],
        reasoning=""
    )
    with pytest.raises(ValueError, match="Subgoal has no reasoning"):
        goal_decomposer._validate_subgoal(invalid_subgoal, sample_context) 