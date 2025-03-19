#!/usr/bin/env python
"""
Production-ready script for running the GoalDecomposer with local config.

This script brings together our custom parser improvements and the local config
approach to create a robust example of using the GoalDecomposer in a real-world setting.
"""

import asyncio
import os
import argparse
from pathlib import Path
from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.tools import get_current_hash
from custom_goal_decomposer import CustomGoalDecomposer
from local_config import load_api_key_from_local_config

async def run_decomposer(goal_description, validation_criteria, quiet=False):
    """Run the goal decomposer with custom parameters."""
    if not quiet:
        print("Midpoint GoalDecomposer\n")
    
    # Load API key from local config.json
    success, api_key = load_api_key_from_local_config()
    if not success:
        print("Error: Could not find a valid API key in config.json")
        print("Please run setup_config.py to set up your API key.")
        return
    
    if not quiet:
        # Mask the API key for display
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        print(f"Found API key in local config.json: {masked_key}")
    
    # Path to test repository
    test_repo_path = os.path.join(os.getcwd(), "test-repo")
    if not quiet:
        print(f"Using test repository at: {test_repo_path}")
    
    # Get the actual git hash
    try:
        git_hash = await get_current_hash(test_repo_path)
        if not quiet:
            print(f"Retrieved actual git hash: {git_hash[:8]}...")
    except Exception as e:
        if not quiet:
            print(f"Error getting git hash: {str(e)}")
            print("Falling back to placeholder hash")
        git_hash = "abc123"  # Placeholder hash if actual hash retrieval fails
    
    # Create a goal with the provided description and criteria
    goal = Goal(
        description=goal_description,
        validation_criteria=validation_criteria,
        success_threshold=0.8
    )
    
    # Create a sample state
    state = State(
        git_hash=git_hash,
        description="Initial repository state with basic project structure"
    )
    
    # Create the task context
    context = TaskContext(
        state=state,
        goal=goal,
        iteration=0,
        execution_history=[]
    )
    
    if not quiet:
        print("\nInput:")
        print(f"Goal: {goal.description}")
        print("Validation Criteria:")
        for criterion in goal.validation_criteria:
            print(f"- {criterion}")
        print(f"Current State: {state.description}")
        print(f"Git Hash: {state.git_hash}")
        print("\nDecomposing goal (this may take a moment)...")
    
    # Initialize the custom GoalDecomposer
    decomposer = CustomGoalDecomposer()
    
    # Decompose the goal
    try:
        strategy = await decomposer.decompose_goal(context)
        
        # Print the results
        if not quiet:
            print("\nOutput:")
            print(f"Strategy: {strategy.metadata.get('strategy_description', 'Not specified')}")
            print("\nSteps:")
            for i, step in enumerate(strategy.steps, 1):
                print(f"{i}. {step}")
            
            print(f"\nReasoning: {strategy.reasoning}")
        
        return strategy
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

async def main():
    """Parse command line arguments and run the decomposer."""
    parser = argparse.ArgumentParser(description="Run the GoalDecomposer with custom parameters")
    parser.add_argument("--goal", "-g", type=str, 
                        default="Build a REST API for a blog application with user authentication and content management",
                        help="Goal description")
    parser.add_argument("--criteria", "-c", type=str, nargs="+",
                        default=["Users can register and log in securely",
                                 "Users can create, update, and delete their blog posts",
                                 "Admin users can moderate all content",
                                 "API supports pagination and filtering of posts",
                                 "All endpoints are properly documented"],
                        help="Validation criteria (multiple values)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress detailed output")
    
    args = parser.parse_args()
    
    await run_decomposer(args.goal, args.criteria, args.quiet)

if __name__ == "__main__":
    asyncio.run(main()) 