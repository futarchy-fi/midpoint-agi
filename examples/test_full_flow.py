import asyncio
import os
from pathlib import Path

from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.goal_decomposer import GoalDecomposer
from midpoint.agents.task_executor import TaskExecutor
from midpoint.agents.goal_validator import GoalValidator
from midpoint.agents.tools import get_current_hash

async def main():
    """Run a test of the full agent flow."""
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
        description="Create a new folder called 'test_data' in the repository root",
        validation_criteria=[
            "Folder 'test_data' exists",
            "Folder is empty",
            "Folder is in the repository root"
        ],
        success_threshold=0.8
    )
    
    # Create task context
    context = TaskContext(
        state=State(
            git_hash=current_hash,
            description="Initial state before creating test_data folder",
            repository_path=str(repo_path)
        ),
        goal=goal,
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )
    
    print("\nStarting full agent flow test...")
    print(f"Goal: {goal.description}")
    print(f"Current State: {context.state.description}")
    print(f"Git Hash: {context.state.git_hash}")
    print(f"Repository Path: {context.state.repository_path}")
    
    try:
        # Step 1: Goal Decomposition
        print("\nStep 1: Goal Decomposition")
        decomposer = GoalDecomposer()
        plan = await decomposer.decompose_goal(context)
        print(f"Decomposition Result: {plan.next_step}")
        print(f"Requires Further Decomposition: {plan.requires_further_decomposition}")
        
        # Step 2: Task Execution
        print("\nStep 2: Task Execution")
        executor = TaskExecutor()
        execution_result = await executor.execute_task(context, plan.next_step)
        print(f"Execution Success: {execution_result.success}")
        print(f"Branch Name: {execution_result.branch_name}")
        print(f"Git Hash: {execution_result.git_hash}")
        if not execution_result.success:
            print(f"Error: {execution_result.error_message}")
        
        # Step 3: Goal Validation
        print("\nStep 3: Goal Validation")
        validator = GoalValidator()
        validation_result = await validator.validate_execution(goal, execution_result)
        print(f"Validation Success: {validation_result.success}")
        print(f"Score: {validation_result.score:.2f}")
        print(f"Reasoning: {validation_result.reasoning}")
        
        print("\nTest completed!")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 