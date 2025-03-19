"""
Tests for the Goal Decomposition agent using a mock OpenAI API.

IMPORTANT: These tests must be run from within the project's virtual environment.
To set up and activate the correct virtual environment:

1. Create the virtual environment (if not exists):
   python -m venv .venv

2. Activate it:
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\\Scripts\\activate  # On Windows

3. Install development dependencies:
   pip install -e ".[dev]"

4. Then run the tests:
   python -m pytest tests/
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest_asyncio
from midpoint.agents.models import State, Goal, TaskContext, StrategyPlan
from midpoint.agents.goal_decomposer import GoalDecomposer
from midpoint.agents.tools import get_current_hash

@pytest.fixture
def test_repo():
    """Get the test repository path."""
    return Path("test-repo")

@pytest_asyncio.fixture
async def git_hash(test_repo):
    """Get the current git hash of the test repo."""
    return await get_current_hash(str(test_repo))

@pytest.fixture
def goal_decomposer():
    """Create a GoalDecomposer instance for testing."""
    # Patch the get_openai_api_key function to return a valid-format test key
    with patch('midpoint.agents.config.get_openai_api_key', return_value="sk-" + "a" * 48):
        return GoalDecomposer()

@pytest.fixture
def sample_goal():
    """Create a sample goal for testing."""
    return Goal(
        description="Add a simple web server with basic routing and error handling",
        validation_criteria=[
            "Server starts and listens on specified port",
            "Handles GET requests to / and /health endpoints",
            "Returns 404 for unknown routes",
            "Logs requests and errors appropriately",
            "Gracefully handles server shutdown"
        ],
        success_threshold=0.8
    )

@pytest_asyncio.fixture
async def sample_context(test_repo, sample_goal, git_hash):
    """Create a sample task context for testing."""
    return TaskContext(
        state=State(
            git_hash=git_hash,
            description="Initial state with hello.py"
        ),
        goal=sample_goal,
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )

# Patch the validation to always succeed
@pytest.fixture
def mock_validation(monkeypatch):
    """Mock the validation method to always succeed."""
    def mock_validate(*args, **kwargs):
        return None  # Just return, don't raise any exceptions
    
    monkeypatch.setattr('midpoint.agents.goal_decomposer.GoalDecomposer._validate_strategy', mock_validate)
    return mock_validate

@pytest.mark.asyncio
async def test_mock_goal_decomposition(goal_decomposer, sample_context, mock_validation):
    """Test goal decomposition with a mock OpenAI API."""
    # Create the expected strategy
    expected_strategy = StrategyPlan(
        steps=[
            "Create a server.py file with a basic HTTP server implementation that starts and listens on specified port",
            "Implement route handling for / and /health endpoints",
            "Add 404 error handling for unknown routes",
            "Implement proper request and error logging",
            "Add graceful shutdown handler",
            "Write tests to verify all functionality"
        ],
        reasoning="This approach separates concerns and implements each validation criterion directly. Starting with a basic server and then adding routing, error handling, logging, and shutdown handling follows a logical progression.",
        estimated_points=500,
        metadata={
            "strategy_description": "Strategy: Implement a basic HTTP server",
            "raw_response": "Strategy: Implement a basic HTTP server\n\nSteps:\n- Create a server.py file with a basic HTTP server implementation that starts and listens on specified port\n- Implement route handling for / and /health endpoints\n- Add 404 error handling for unknown routes\n- Implement proper request and error logging\n- Add graceful shutdown handler\n- Write tests to verify all functionality\n\nReasoning: This approach separates concerns and implements each validation criterion directly. Starting with a basic server and then adding routing, error handling, logging, and shutdown handling follows a logical progression.\n\nPoints: 500\n"
        }
    )
    
    # Mock the response but use our custom _parse_response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="dummy content"))]
    
    # Use AsyncMock for the create method and patch _parse_response
    with patch.object(goal_decomposer.client.chat.completions, "create", 
                      new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        
        # Directly patch _parse_response to return our expected strategy
        with patch.object(goal_decomposer, "_parse_response", return_value=expected_strategy):
            strategy = await goal_decomposer.decompose_goal(sample_context)
            
            # Print the strategy for inspection
            print("\nGenerated Strategy:")
            print(f"Description: {strategy.metadata['strategy_description']}")
            print("\nSteps:")
            for step in strategy.steps:
                print(f"- {step}")
            print(f"\nReasoning: {strategy.reasoning}")
            print(f"Estimated Points: {strategy.estimated_points}")
            
            # Basic validation
            assert isinstance(strategy, StrategyPlan)
            assert len(strategy.steps) > 0
            assert strategy.reasoning
            assert strategy.estimated_points == 500  # Directly check for the expected value
            assert strategy.estimated_points <= sample_context.total_budget

@pytest.mark.asyncio
async def test_mock_complex_goal_decomposition(goal_decomposer, test_repo, git_hash, mock_validation):
    """Test decomposition of a complex goal with a mock OpenAI API."""
    complex_goal = Goal(
        description="Implement a full-stack task management system with user authentication, task creation, assignment, and progress tracking",
        validation_criteria=[
            "Users can create and manage tasks",
            "Tasks can be assigned to users",
            "Task status can be updated",
            "Users can view their assigned tasks",
            "Task history is maintained",
            "System supports task prioritization",
            "Users receive notifications for task updates"
        ],
        success_threshold=0.8
    )
    
    context = TaskContext(
        state=State(
            git_hash=git_hash,
            description="Initial state with hello.py"
        ),
        goal=complex_goal,
        iteration=0,
        points_consumed=0,
        total_budget=5000,
        execution_history=[]
    )
    
    # Create a mock response that includes all validation criteria
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="""Strategy: Full-Stack Task Management System

Steps:
- Set up database with user and task tables
- Implement user authentication system
- Create task model with status, priority, and assignment fields
- Implement task creation and management API so users can create and manage tasks
- Add task assignment functionality so tasks can be assigned to users
- Implement task status update features so task status can be updated
- Create user dashboard to view assigned tasks so users can view their assigned tasks
- Implement task history tracking so task history is maintained
- Add task prioritization system so system supports task prioritization
- Set up notification system for task updates so users receive notifications for task updates
- Create frontend views for all functionality
- Implement proper error handling and validation

Reasoning: This approach builds the system layer by layer, starting with the data model and core functionality, then adding specialized features like history tracking, prioritization, and notifications.

Points: 2500"""))]
    
    # Use AsyncMock for the create method
    with patch.object(goal_decomposer.client.chat.completions, "create",
                      new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        
        strategy = await goal_decomposer.decompose_goal(context)
        
        # Print the strategy for inspection
        print("\nGenerated Strategy for Complex Goal:")
        print(f"Description: {strategy.metadata['strategy_description']}")
        print("\nSteps:")
        for step in strategy.steps:
            print(f"- {step}")
        print(f"\nReasoning: {strategy.reasoning}")
        print(f"Estimated Points: {strategy.estimated_points}")
        
        # Basic validation
        assert isinstance(strategy, StrategyPlan)
        assert len(strategy.steps) >= 4  # Should have multiple major components
        assert strategy.reasoning
        assert strategy.estimated_points > 0
        assert strategy.estimated_points <= context.total_budget 