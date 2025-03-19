import pytest
import pytest_asyncio
from pathlib import Path
import os

from midpoint.agents.models import State, Goal, TaskContext, ExecutionTrace
from midpoint.agents.task_executor import TaskExecutor
from midpoint.agents.tools import get_current_hash

@pytest_asyncio.fixture
async def temp_repo(tmp_path):
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()
    
    # Initialize git repository
    os.system(f"cd {repo_path} && git init")
    
    # Create a test file
    test_file = repo_path / "test.txt"
    test_file.write_text("Initial content")
    
    # Add and commit the test file
    os.system(f"cd {repo_path} && git add . && git commit -m 'Initial commit'")
    
    return repo_path

@pytest_asyncio.fixture
async def sample_context(temp_repo):
    """Create a sample task context for testing."""
    git_hash = await get_current_hash(str(temp_repo))
    return TaskContext(
        state=State(
            git_hash=git_hash,
            description="Initial state with test file",
            repository_path=str(temp_repo)
        ),
        goal=Goal(
            description="Test task",
            validation_criteria=["Test passes"],
            success_threshold=0.8
        ),
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )

@pytest_asyncio.fixture
def task_executor():
    """Create a TaskExecutor instance for testing."""
    return TaskExecutor()

@pytest.mark.asyncio
async def test_execute_task_success(task_executor, sample_context):
    """Test successful task execution."""
    # Create a simple task that modifies the test file
    task = "Update test.txt with new content"
    
    # Execute the task
    trace = await task_executor.execute_task(sample_context, task)
    
    # Verify the execution trace
    assert isinstance(trace, ExecutionTrace)
    assert trace.task_description == task
    assert trace.success
    assert trace.execution_time > 0
    assert trace.points_consumed >= 0
    
    # Verify the repository state
    assert trace.resulting_state.git_hash != sample_context.state.git_hash
    assert trace.resulting_state.repository_path == sample_context.state.repository_path

@pytest.mark.asyncio
async def test_execute_task_failure(task_executor, sample_context):
    """Test task execution failure."""
    # Create a task that will fail
    task = "Execute invalid command"
    
    # Execute the task
    trace = await task_executor.execute_task(sample_context, task)
    
    # Verify the execution trace
    assert isinstance(trace, ExecutionTrace)
    assert trace.task_description == task
    assert not trace.success
    assert trace.error_message is not None
    assert trace.execution_time > 0
    
    # Verify the repository state remains unchanged
    assert trace.resulting_state.git_hash == sample_context.state.git_hash

@pytest.mark.asyncio
async def test_invalid_repository(task_executor):
    """Test execution with invalid repository."""
    # Create context with invalid repository
    invalid_context = TaskContext(
        state=State(
            git_hash="invalid",
            description="Invalid state",
            repository_path="/invalid/path"
        ),
        goal=Goal(
            description="Test task",
            validation_criteria=["Test passes"],
            success_threshold=0.8
        ),
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )
    
    # Attempt to execute task
    with pytest.raises(ValueError):
        await task_executor.execute_task(invalid_context, "Test task") 