import asyncio
import os
from pathlib import Path

from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.task_executor import TaskExecutor
from midpoint.agents.tools import get_current_hash

async def main():
    """Run a test of the TaskExecutor."""
    # Get the test repository path
    repo_path = os.getenv("MIDPOINT_TEST_REPO", os.path.expanduser("~/midpoint-test-repo"))
    repo_path = Path(repo_path)
    
    if not repo_path.exists():
        print(f"Error: Test repository not found at {repo_path}")
        print("Please run: python examples/setup_test_repo.py")
        return
    
    # Get current repository state
    current_hash = await get_current_hash(str(repo_path))
    
    # Create a test goal
    goal = Goal(
        description="Add a simple test function",
        validation_criteria=[
            "Function is properly defined",
            "Function has docstring",
            "Function has type hints"
        ],
        success_threshold=0.8
    )
    
    # Create task context
    context = TaskContext(
        state=State(
            git_hash=current_hash,
            description="Initial state before adding test function",
            repository_path=str(repo_path)
        ),
        goal=goal,
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )
    
    # Create and run the TaskExecutor
    executor = TaskExecutor()
    
    # Define a simple task
    task = "Add a test function that returns the sum of two numbers"
    
    print("\nInput:")
    print(f"Task: {task}")
    print(f"Current State: {context.state.description}")
    print(f"Git Hash: {context.state.git_hash}")
    print(f"Repository Path: {context.state.repository_path}")
    print("\nExecuting task...")
    
    try:
        # Execute the task
        trace = await executor.execute_task(context, task)
        
        # Print the results
        print("\nOutput:")
        print(f"Success: {trace.success}")
        print(f"Execution Time: {trace.execution_time:.2f} seconds")
        print(f"Points Consumed: {trace.points_consumed}")
        print(f"Resulting State: {trace.resulting_state.description}")
        print(f"New Git Hash: {trace.resulting_state.git_hash}")
        
        if not trace.success:
            print(f"\nError: {trace.error_message}")
        
        print("\nTest completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 