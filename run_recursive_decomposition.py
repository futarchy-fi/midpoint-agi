#!/usr/bin/env python
"""
Script to demonstrate recursive goal decomposition.

Usage:
    python run_recursive_decomposition.py <repository_path> <goal_description>
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.goal_decomposer import GoalDecomposer, validate_repository_state
from midpoint.agents.tools import get_current_hash

async def main():
    """Run recursive goal decomposition."""
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python run_recursive_decomposition.py <repository_path> <goal_description>")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    goal_description = sys.argv[2]
    
    # Ensure repository path exists
    if not os.path.exists(repo_path):
        print(f"Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    repo_path = str(Path(repo_path).resolve())
    
    # Get current git hash
    try:
        git_hash = await get_current_hash(repo_path)
    except Exception as e:
        print(f"Error getting git hash: {str(e)}")
        sys.exit(1)
    
    # Create goal
    goal = Goal(
        description=goal_description,
        validation_criteria=[
            "The goal is fully implemented",
            "All tests pass",
            "Code is well-documented",
            "No regressions in existing functionality"
        ],
        success_threshold=0.8
    )
    
    # Create initial context
    context = TaskContext(
        state=State(
            git_hash=git_hash,
            description="Initial state",
            repository_path=repo_path
        ),
        goal=goal,
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )
    
    # Create log file
    log_file = "goal_hierarchy.log"
    with open(log_file, "w") as f:
        f.write(f"Starting recursive goal decomposition for: {goal_description}\n")
        f.write(f"Repository: {repo_path}\n")
        f.write(f"Git Hash: {git_hash}\n\n")
        f.flush()
    
    print(f"Starting recursive goal decomposition...")
    print(f"Log file: {log_file}")
    print("You can follow the progress using: tail -f goal_hierarchy.log")
    
    # Create goal decomposer
    goal_decomposer = GoalDecomposer()
    
    # Validate repository state
    try:
        await validate_repository_state(repo_path, git_hash)
        print("Repository validation successful")
    except ValueError as e:
        print(f"Repository validation failed: {str(e)}")
        sys.exit(1)
    
    try:
        # Run recursive decomposition
        subgoals = await goal_decomposer.decompose_recursively(context, log_file)
        
        # Write summary to log
        with open(log_file, "a") as f:
            f.write("\n\nRecursive Decomposition Complete\n")
            f.write(f"Total subgoals: {len(subgoals)}\n")
            f.write(f"Executable tasks: {sum(1 for s in subgoals if not s.requires_further_decomposition)}\n\n")
            f.flush()
        
        print(f"\nRecursive decomposition complete!")
        print(f"Total subgoals: {len(subgoals)}")
        print(f"Executable tasks: {sum(1 for s in subgoals if not s.requires_further_decomposition)}")
        print(f"See {log_file} for details")
        
    except Exception as e:
        print(f"Error during recursive decomposition: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 