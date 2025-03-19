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
from unittest.mock import patch, MagicMock, AsyncMock
from midpoint.agents.models import State, Goal, StrategyPlan, TaskContext
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
            description="Initial state with test file"
        ),
        goal=sample_goal,
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )

@pytest.mark.asyncio
async def test_goal_decomposition_basic(goal_decomposer, sample_context):
    """Test basic goal decomposition functionality."""
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="""Strategy: Test
Steps:
- Implement user registration with email and password
- Add login functionality for user authentication
- Create logout endpoint
- Implement session management for user state
Reasoning: Test reasoning
Points: 100"""))]
        )
        
        strategy = await goal_decomposer.decompose_goal(sample_context)
        assert len(strategy.steps) == 4
        assert strategy.estimated_points == 100
        assert "Test reasoning" in strategy.reasoning

@pytest.mark.asyncio
async def test_goal_decomposition_with_state(goal_decomposer, temp_repo, sample_context):
    """Test goal decomposition with repository state."""
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="""Strategy: Test with state
Steps:
- Set up user registration with email/password
- Implement user login functionality
- Add logout endpoint for users
- Set up session management
Reasoning: Test reasoning with state
Points: 150"""))]
        )
        
        # Update context with repo path
        sample_context.state.repository_path = str(temp_repo)
        strategy = await goal_decomposer.decompose_goal(sample_context)
        
        assert len(strategy.steps) == 4
        assert strategy.estimated_points == 150
        assert "Test reasoning with state" in strategy.reasoning

@pytest.mark.asyncio
async def test_goal_decomposition_complex(goal_decomposer):
    """Test decomposition of a complex goal."""
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
            description="Empty repository"
        ),
        goal=complex_goal,
        iteration=0,
        points_consumed=0,
        total_budget=5000,
        execution_history=[]
    )
    
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="""Strategy: Test
Steps:
- Set up product browsing functionality for users
- Implement shopping cart with add items feature
- Create checkout process with order processing
- Add inventory management and updates
- Set up order confirmation system
Reasoning: Test reasoning
Points: 1000"""))]
        )
        
        strategy = await goal_decomposer.decompose_goal(context)
        assert len(strategy.steps) == 5
        assert strategy.estimated_points == 1000
        assert "Test reasoning" in strategy.reasoning

@pytest.mark.asyncio
async def test_goal_decomposition_points_budget(goal_decomposer, sample_context):
    """Test points budget handling in goal decomposition."""
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        # Test with very low budget
        sample_context.total_budget = 100
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="""Strategy: Test
Steps:
- Implement user registration with email
- Add login functionality
- Create logout endpoint
- Set up session management
Reasoning: Test reasoning
Points: 50"""))]
        )
        strategy = await goal_decomposer.decompose_goal(sample_context)
        assert strategy.estimated_points <= 100
        
        # Test with high budget
        sample_context.total_budget = 10000
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="""Strategy: Test
Steps:
- Set up comprehensive user registration
- Implement secure login system
- Add logout functionality
- Create robust session management
Reasoning: Test reasoning
Points: 5000"""))]
        )
        strategy = await goal_decomposer.decompose_goal(sample_context)
        assert strategy.estimated_points <= 10000

@pytest.mark.asyncio
async def test_goal_decomposition_iteration_handling(goal_decomposer, sample_context):
    """Test handling of multiple iterations in goal decomposition."""
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        # First iteration
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="""Strategy: Test
Steps:
- Implement user registration system
- Add login functionality
- Create logout endpoint
- Set up session management
Reasoning: Test reasoning
Points: 100"""))]
        )
        strategy1 = await goal_decomposer.decompose_goal(sample_context)
        
        # Update context for second iteration
        sample_context.iteration = 1
        sample_context.points_consumed = 100
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="""Strategy: Test
Steps:
- Enhance user registration with validation
- Improve login security
- Add session timeout
- Implement remember me functionality
Reasoning: Test reasoning
Points: 50"""))]
        )
        strategy2 = await goal_decomposer.decompose_goal(sample_context)
        
        # Verify strategies are different
        assert strategy1.steps != strategy2.steps
        assert strategy2.estimated_points <= (sample_context.total_budget - sample_context.points_consumed)

@pytest.mark.asyncio
async def test_goal_decomposition_error_handling(goal_decomposer):
    """Test error handling in goal decomposition."""
    # Test with invalid goal
    invalid_context = TaskContext(
        state=State(git_hash="invalid", description="Invalid state"),
        goal=None,  # Invalid goal
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )
    
    with pytest.raises(ValueError, match="No goal provided in context"):
        await goal_decomposer.decompose_goal(invalid_context)
    
    # Test with negative budget
    invalid_context.goal = Goal(
        description="Test",
        validation_criteria=["Test"],
        success_threshold=0.8
    )
    invalid_context.total_budget = -100
    
    with pytest.raises(ValueError, match="Invalid points budget"):
        await goal_decomposer.decompose_goal(invalid_context)

@pytest.mark.asyncio
async def test_response_parsing(goal_decomposer):
    """Test parsing of OpenAI API responses."""
    sample_response = """Strategy: Implement authentication system

Steps:
- Create user model
- Add authentication endpoints
- Implement session management
- Add security middleware

Reasoning: This approach follows best practices for auth systems

Points: 500"""

    strategy = goal_decomposer._parse_response(sample_response)
    
    assert len(strategy.steps) == 4
    assert "Create user model" in strategy.steps
    assert strategy.reasoning == "This approach follows best practices for auth systems"
    assert strategy.estimated_points == 500
    assert strategy.metadata["strategy_description"] == "Strategy: Implement authentication system"
    assert strategy.metadata["raw_response"] == sample_response

@pytest.mark.asyncio
async def test_strategy_validation(goal_decomposer, sample_context):
    """Test strategy validation logic."""
    # Test empty steps
    invalid_strategy = StrategyPlan(
        steps=[],
        reasoning="Test",
        estimated_points=100,
        metadata={"strategy_description": "Test", "raw_response": "Test"}
    )
    with pytest.raises(ValueError, match="Strategy has no steps"):
        goal_decomposer._validate_strategy(invalid_strategy, sample_context)
    
    # Test empty reasoning
    invalid_strategy = StrategyPlan(
        steps=["Test step"],
        reasoning="",
        estimated_points=100,
        metadata={"strategy_description": "Test", "raw_response": "Test"}
    )
    with pytest.raises(ValueError, match="Strategy has no reasoning"):
        goal_decomposer._validate_strategy(invalid_strategy, sample_context)
    
    # Test invalid points
    invalid_strategy = StrategyPlan(
        steps=["Test step"],
        reasoning="Test",
        estimated_points=0,
        metadata={"strategy_description": "Test", "raw_response": "Test"}
    )
    with pytest.raises(ValueError, match="Invalid point estimate"):
        goal_decomposer._validate_strategy(invalid_strategy, sample_context)
    
    # Test exceeding budget
    invalid_strategy = StrategyPlan(
        steps=["Test step"],
        reasoning="Test",
        estimated_points=2000,
        metadata={"strategy_description": "Test", "raw_response": "Test"}
    )
    with pytest.raises(ValueError, match="Strategy exceeds available budget"):
        goal_decomposer._validate_strategy(invalid_strategy, sample_context)
    
    # Test insufficient validation criteria coverage
    invalid_strategy = StrategyPlan(
        steps=["Test step"],  # Only covers one criterion
        reasoning="Test",
        estimated_points=100,
        metadata={"strategy_description": "Test", "raw_response": "Test"}
    )
    with pytest.raises(ValueError, match="Strategy does not cover enough validation criteria"):
        goal_decomposer._validate_strategy(invalid_strategy, sample_context)

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
    assert "Iteration: 0" in prompt
    assert "Points Consumed: 0" in prompt
    assert "Total Budget: 1000" in prompt

@pytest.mark.asyncio
async def test_api_error_handling(goal_decomposer, sample_context):
    """Test handling of API errors."""
    with patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock, side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="Error during goal decomposition: API Error"):
            await goal_decomposer.decompose_goal(sample_context)

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
    """Test points tracking during goal decomposition."""
    points_tracked = []
    
    async def mock_track_points(operation: str, points: int):
        points_tracked.append((operation, points))
    
    with patch("midpoint.agents.goal_decomposer.track_points", new=mock_track_points), \
         patch.object(goal_decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="""Strategy: Test
Steps:
- Implement user registration with email
- Add login functionality
- Create logout endpoint
- Set up session management
Reasoning: Test reasoning
Points: 100"""))]
        )
        
        await goal_decomposer.decompose_goal(sample_context)
        
        assert len(points_tracked) == 1
        assert points_tracked[0][0] == "goal_decomposition"
        assert points_tracked[0][1] == 10 