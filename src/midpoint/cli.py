"""
Command-line interface for Midpoint.
"""

import asyncio
import argparse
from pathlib import Path
from typing import Optional, List

from .orchestrator import Orchestrator
from .agents.models import Goal
from .agents.tools import checkout_branch, get_current_hash

async def run_task(
    repo_path: str,
    goal_description: str,
    validation_criteria: List[str],
    success_threshold: float = 0.8,
    max_iterations: int = 5,
    checkpoint_path: Optional[str] = None,
    start_branch: Optional[str] = None
) -> None:
    """
    Run a Midpoint task on a repository.
    
    Args:
        repo_path: Path to the target repository
        goal_description: Description of the goal to achieve
        validation_criteria: List of criteria to validate the goal
        success_threshold: Minimum validation score for success (0.0-1.0)
        max_iterations: Maximum number of iterations to attempt
        checkpoint_path: Optional path to save checkpoints
        start_branch: Optional branch to start from (defaults to current branch)
    """
    # Checkout specified branch if provided
    if start_branch:
        print(f"Checking out branch: {start_branch}")
        await checkout_branch(repo_path, start_branch)
    
    # Create the goal
    goal = Goal(
        description=goal_description,
        validation_criteria=validation_criteria,
        success_threshold=success_threshold
    )
    
    print(f"Starting Midpoint task on repository: {repo_path}")
    print(f"Goal: {goal.description}")
    
    # Run the orchestrator
    orchestrator = Orchestrator()
    result = await orchestrator.run(
        repo_path=repo_path,
        goal=goal,
        max_iterations=max_iterations,
        checkpoint_path=checkpoint_path
    )
    
    # Print the result
    if result.success:
        print("\nTask completed successfully!")
        print(f"Final Git Hash: {result.final_state.git_hash}")
    else:
        print("\nTask failed!")
        print(f"Error: {result.error_message}")
    
    if result.execution_history:
        print("\nExecution History:")
        for i, entry in enumerate(result.execution_history, 1):
            print(f"{i}. {entry['subgoal']}")
            print(f"   Git Hash: {entry['git_hash']}")
            print(f"   Validation Score: {entry['validation_score']:.2f}")

async def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run a Midpoint task on a repository")
    parser.add_argument("repo_path", help="Path to the target repository")
    parser.add_argument("goal", help="Description of the goal to achieve")
    parser.add_argument("--criteria", nargs="+", required=True, help="Validation criteria for the goal")
    parser.add_argument("--threshold", type=float, default=0.8, help="Success threshold (0.0-1.0)")
    parser.add_argument("--iterations", type=int, default=5, help="Maximum number of iterations")
    parser.add_argument("--checkpoint", help="Path to save checkpoints")
    parser.add_argument("--branch", help="Branch to start from (defaults to current branch)")
    
    args = parser.parse_args()
    
    # Create checkpoint directory if specified
    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the task
    await run_task(
        repo_path=args.repo_path,
        goal_description=args.goal,
        validation_criteria=args.criteria,
        success_threshold=args.threshold,
        max_iterations=args.iterations,
        checkpoint_path=args.checkpoint,
        start_branch=args.branch
    )

if __name__ == "__main__":
    asyncio.run(main()) 