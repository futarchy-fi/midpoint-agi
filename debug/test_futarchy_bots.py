"""
Test the orchestrator by creating a utility function in the futarchy-bots repository.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midpoint.orchestrator import Orchestrator
from midpoint.agents.models import Goal

async def main():
    # Get the absolute path to the futarchy-bots repository
    repo_path = "/Users/kas/futarchy-bots"
    
    # Create a goal to implement a utility function for timestamp conversion
    goal = Goal(
        description="Create a utility function in the futarchy/development/utils.py file that converts Unix timestamps to human-readable datetime strings. If the utils.py file doesn't exist, create it.",
        validation_criteria=[
            "Function is defined in futarchy/development/utils.py",
            "Function is named 'format_timestamp'",
            "Function takes a Unix timestamp (integer) parameter",
            "Function returns a human-readable datetime string",
            "Function includes appropriate docstring explaining its purpose",
            "Function handles both integer and float timestamp inputs",
            "The implementation uses the datetime standard library"
        ],
        success_threshold=0.8
    )
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = str(checkpoint_dir / "futarchy_test_checkpoint.json")
    
    print("Starting orchestrator test for futarchy-bots repository...")
    print(f"Goal: {goal.description}")
    print("Repository path:", repo_path)
    
    # Run the orchestrator
    orchestrator = Orchestrator()
    result = await orchestrator.run(
        repo_path=repo_path, 
        goal=goal,
        max_iterations=5,
        checkpoint_path=checkpoint_path
    )
    
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