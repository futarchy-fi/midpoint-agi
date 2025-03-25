"""
Command-line interface for Midpoint.
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from .orchestrator import Orchestrator
from .agents.models import Goal
from .agents.tools import checkout_branch, get_current_hash

async def run_task(
    repo_path: str,
    goal: str,
    validation_criteria: List[str],
    success_threshold: float = 0.8,
    max_iterations: int = 10,
    checkpoint_path: Optional[str] = None,
    starting_branch: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a Midpoint task on a repository.
    
    Args:
        repo_path: Path to the git repository
        goal: Description of the goal to achieve
        validation_criteria: List of validation criteria
        success_threshold: Success threshold (0.0-1.0)
        max_iterations: Maximum number of iterations
        checkpoint_path: Path to save checkpoints
        starting_branch: Branch to start from
        
    Returns:
        Dictionary containing the execution result
    """
    # Create goal object
    goal_obj = Goal(
        description=goal,
        validation_criteria=validation_criteria,
        success_threshold=success_threshold
    )
    
    # Run orchestration
    result = await run_orchestration(
        repo_path=repo_path,
        goal=goal_obj,
        max_iterations=max_iterations,
        checkpoint_path=checkpoint_path
    )
    
    # Return result as dictionary
    return {
        "success": result.success,
        "final_state": {
            "git_hash": result.final_state.git_hash if result.final_state else None,
            "repository_path": result.final_state.repository_path if result.final_state else None,
            "description": result.final_state.description if result.final_state else None
        },
        "error_message": result.error_message,
        "execution_history": result.execution_history
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Midpoint task")
    parser.add_argument("repo_path", help="Path to the target repository")
    parser.add_argument("goal", help="Description of the goal to achieve")
    parser.add_argument("--criteria", nargs="+", required=True, help="Validation criteria for the goal")
    parser.add_argument("--threshold", type=float, default=0.8, help="Success threshold (0.0-1.0)")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--checkpoint", help="Path to save checkpoints")
    parser.add_argument("--branch", help="Branch to start from")
    
    args = parser.parse_args()
    
    # Run the task
    result = asyncio.run(run_task(
        repo_path=args.repo_path,
        goal=args.goal,
        validation_criteria=args.criteria,
        success_threshold=args.threshold,
        max_iterations=args.iterations,
        checkpoint_path=args.checkpoint,
        starting_branch=args.branch
    ))
    
    # Print result as JSON
    print(json.dumps(result, indent=2)) 