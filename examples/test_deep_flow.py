#!/usr/bin/env python
"""
Script to demonstrate the full flow with deep decomposition, execution, and validation.
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from midpoint.agents.models import State, Goal, TaskContext, SubgoalPlan
from midpoint.agents.goal_decomposer import GoalDecomposer, validate_repository_state
from midpoint.agents.task_executor import TaskExecutor
from midpoint.agents.goal_validator import GoalValidator
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

async def main():
    """Run the full flow with deep decomposition."""
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python test_deep_flow.py <repository_path> <goal_description>")
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
    
    print("\nStarting full flow test...")
    print(f"Goal: {goal.description}")
    print(f"Repository: {repo_path}")
    print(f"Git Hash: {git_hash}")
    
    try:
        # Step 1: Deep Decomposition
        print("\nStep 1: Deep Goal Decomposition")
        decomposer = DeepGoalDecomposer()
        subgoals = await decomposer.decompose_recursively(context, "deep_goal_hierarchy.log")
        
        # Find the final executable subgoal
        executable_subgoals = [s for s in subgoals if not s.requires_further_decomposition]
        if not executable_subgoals:
            print("No executable subgoals found!")
            sys.exit(1)
            
        final_subgoal = executable_subgoals[0]
        print(f"\nFound executable subgoal: {final_subgoal.next_step}")
        print(f"Validation criteria: {final_subgoal.validation_criteria}")
        
        # Step 2: Task Execution
        print("\nStep 2: Task Execution")
        executor = TaskExecutor()
        execution_result = await executor.execute_task(context, final_subgoal.next_step)
        print(f"Execution Success: {execution_result.success}")
        print(f"Branch Name: {execution_result.branch_name}")
        print(f"Git Hash: {execution_result.git_hash}")
        if not execution_result.success:
            print(f"Error: {execution_result.error_message}")
            sys.exit(1)
        
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
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 