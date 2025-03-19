#!/usr/bin/env python
"""
Script to demonstrate deep recursive goal decomposition.

This script forces a deeper decomposition hierarchy by overriding
the requires_further_decomposition parameter based on complexity.
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from midpoint.agents.models import State, Goal, TaskContext, SubgoalPlan
from midpoint.agents.goal_decomposer import GoalDecomposer, validate_repository_state
from midpoint.agents.tools import get_current_hash
from typing import List

class DeepGoalDecomposer(GoalDecomposer):
    """Extended GoalDecomposer that forces deeper decomposition."""
    
    async def determine_next_step(self, context: TaskContext) -> SubgoalPlan:
        """Override to force deeper decomposition."""
        subgoal = await super().determine_next_step(context)
        
        # Force deeper decomposition based on complexity indicators
        complex_keywords = ["platform", "system", "framework", "architecture", "infrastructure", "database"]
        
        # Check if this is a complex task
        is_complex = any(keyword in subgoal.next_step.lower() for keyword in complex_keywords)
        
        # If it's complex and not already marked for decomposition, override it
        if is_complex and not subgoal.requires_further_decomposition:
            print(f"Forcing deeper decomposition for: {subgoal.next_step}")
            subgoal.requires_further_decomposition = True
            
        return subgoal
        
    async def decompose_recursively(self, context: TaskContext, log_file: str = "goal_hierarchy.log") -> List[SubgoalPlan]:
        """
        Modified version of decompose_recursively that handles metadata correctly.
        """
        # Validate repository state
        await validate_repository_state(
            context.state.repository_path, 
            context.state.git_hash
        )
        
        # Get the decomposition depth for logging
        depth = self._get_decomposition_depth(context)
        
        # Get the next subgoal
        subgoal = await self.determine_next_step(context)
        
        # Log this decomposition step
        self._log_decomposition_step(log_file, depth, context.goal.description, subgoal.next_step)
        
        # If subgoal doesn't need further decomposition, return it
        if not subgoal.requires_further_decomposition:
            self._log_execution_ready(log_file, depth + 1, subgoal.next_step)
            return [subgoal]
        
        # Store parent information in a way that doesn't require a metadata attribute
        parent_context = {
            "parent_goal": context.goal.description,
            "parent_context": subgoal.relevant_context
        }
        
        # Create a new context with this subgoal as the goal
        new_context = TaskContext(
            state=context.state,
            goal=Goal(
                description=subgoal.next_step,
                validation_criteria=subgoal.validation_criteria,
                success_threshold=context.goal.success_threshold
            ),
            iteration=0,  # Reset for the new subgoal
            points_consumed=context.points_consumed,
            total_budget=context.total_budget - 10,  # Subtract points for the decomposition
            execution_history=context.execution_history + [parent_context]  # Store parent info in execution history
        )
        
        # Recursively decompose and collect all resulting subgoals
        sub_subgoals = await self.decompose_recursively(new_context, log_file)
        
        # Return the current subgoal and all its sub-subgoals
        return [subgoal] + sub_subgoals
    
    def _get_decomposition_depth(self, context: TaskContext) -> int:
        """Determine the current decomposition depth from execution history."""
        # Count the number of parent_goal entries in execution history
        depth = 0
        for entry in context.execution_history:
            if isinstance(entry, dict) and 'parent_goal' in entry:
                depth += 1
        return depth

async def main():
    """Run deep recursive goal decomposition."""
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python deep_decomposition_test.py <repository_path> <goal_description>")
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
    log_file = "deep_goal_hierarchy.log"
    with open(log_file, "w") as f:
        f.write(f"Starting deep recursive goal decomposition for: {goal_description}\n")
        f.write(f"Repository: {repo_path}\n")
        f.write(f"Git Hash: {git_hash}\n\n")
        f.flush()
    
    print(f"Starting deep recursive goal decomposition...")
    print(f"Log file: {log_file}")
    print("You can follow the progress using: tail -f deep_goal_hierarchy.log")
    
    # Create deep goal decomposer
    goal_decomposer = DeepGoalDecomposer()
    
    # Validate repository state
    try:
        await validate_repository_state(repo_path, git_hash)
        print("Repository validation successful")
    except ValueError as e:
        print(f"Repository validation failed: {str(e)}")
        sys.exit(1)
    
    try:
        # Run recursive decomposition with increased max_depth
        subgoals = await goal_decomposer.decompose_recursively(context, log_file)
        
        # Write summary to log
        with open(log_file, "a") as f:
            f.write("\n\nDeep Recursive Decomposition Complete\n")
            f.write(f"Total subgoals: {len(subgoals)}\n")
            f.write(f"Executable tasks: {sum(1 for s in subgoals if not s.requires_further_decomposition)}\n\n")
            f.flush()
        
        print(f"\nDeep recursive decomposition complete!")
        print(f"Total subgoals: {len(subgoals)}")
        print(f"Executable tasks: {sum(1 for s in subgoals if not s.requires_further_decomposition)}")
        print(f"See {log_file} for details")
        
    except Exception as e:
        print(f"Error during recursive decomposition: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 