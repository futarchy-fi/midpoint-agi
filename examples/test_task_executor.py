"""
Example script to test the TaskExecutor.

This script shows how to use the TaskExecutor to execute a simple task
in a git repository.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.midpoint.agents.models import TaskContext, RepositoryContext
from src.midpoint.agents.task_executor import TaskExecutor

def main():
    # Get the path to the test repository
    repo_path = os.getenv("MIDPOINT_TEST_REPO", os.path.expanduser("~/midpoint-test-repo"))
    
    # Create a task context
    context = TaskContext(
        task="Add a function called 'add_numbers' to hello.py that takes two numbers as input and returns their sum",
        repository=RepositoryContext(
            repository_path=repo_path,
        )
    )
    
    # Create a task executor
    executor = TaskExecutor()
    
    # Execute the task
    result = executor.execute_task(context)
    
    # Print the result
    print("\nTask execution completed.")
    print(f"Success: {result.success}")
    print(f"Final Git Hash: {result.final_git_hash}")
    print(f"Error Message: {result.error_message}")
    
    if result.output_content:
        print("\nOutput:")
        print(f"{result.output_content}")

if __name__ == "__main__":
    main() 