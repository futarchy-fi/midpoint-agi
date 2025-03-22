"""
Test the orchestrator by creating a personalized greeting function.
"""

import asyncio
import os
from pathlib import Path

from src.midpoint.orchestrator import Orchestrator
from src.midpoint.agents.models import Goal
from tests.cleanup_test_repo import cleanup_test_repo

async def main():
    # Clean up the test repository first
    if not cleanup_test_repo():
        print("Failed to clean up test repository. Aborting test.")
        return
    
    # Get the absolute path to the test repository
    repo_path = str(Path(__file__).parent.parent / "test-repo")
    
    # Create a goal to create a personalized greeting function
    goal = Goal(
        description="Create a personalized greeting function that takes a name parameter and returns a friendly greeting",
        validation_criteria=[
            "Function is defined in hello.py",
            "Function is named 'greet'",
            "Function takes a 'name' parameter",
            "Function returns a string containing the name",
            "The greeting is friendly and personalized"
        ],
        success_threshold=0.8
    )
    
    print("Starting orchestrator test...")
    print(f"Goal: {goal.description}")
    print("Repository path:", repo_path)
    
    # Run the orchestrator
    orchestrator = Orchestrator()
    result = await orchestrator.run(repo_path, goal)
    
    # Print the result
    if result.success:
        print("\nOrchestration completed successfully!")
        print(f"Final Git Hash: {result.final_state.git_hash}")
    else:
        print("\nOrchestration failed!")
        print(f"Error: {result.error_message}")
    
    if result.execution_history:
        print("\nExecution History:")
        for i, entry in enumerate(result.execution_history, 1):
            print(f"{i}. {entry['subgoal']}")
            print(f"   Git Hash: {entry['git_hash']}")
            print(f"   Validation Score: {entry['validation_score']:.2f}")

if __name__ == "__main__":
    asyncio.run(main()) 