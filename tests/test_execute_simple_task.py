"""
Test script for running a real task with the task executor.
"""

import os
import sys
import asyncio
from pathlib import Path
import logging

# Add the source directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.midpoint.agents.task_executor import TaskExecutor
from src.midpoint.agents.models import TaskContext, Goal, State

async def main():
    """Run a simple task with the task executor."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get the repository path
    repo_path = os.getcwd()
    logging.info(f"Repository path: {repo_path}")
    
    # Create a task context
    state = State(
        repository_path=repo_path,
        git_hash="current-hash",  # This will be ignored since we're creating a new branch
        branch_name="main",
        description="Current repository state"
    )
    
    goal = Goal(
        description="Add a simple Hello World function to the codebase",
        validation_criteria=["Function should print 'Hello, World!'", "Function should be named 'hello_world'"]
    )
    
    task_context = TaskContext(
        goal=goal,
        state=state,
        iteration=1
    )
    
    # Create a task executor
    task_executor = TaskExecutor(model="gpt-4o")
    
    # Execute the task
    result = await task_executor.execute_task(
        task_context,
        "Create a new file called hello_world.py in the src directory with a function that prints 'Hello, World!'",
        setup_logging=True,
        debug=True
    )
    
    # Print the result
    logging.info(f"Task execution result: {result.success}")
    if result.success:
        logging.info(f"Commit hash: {result.git_hash}")
        logging.info(f"Branch name: {result.branch_name}")
    else:
        logging.error(f"Error: {result.error_message}")
    
    return result.success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 