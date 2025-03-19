#!/usr/bin/env python
"""
Example script using the CustomGoalDecomposer with improved parsing.

This script creates a sample task context and uses our custom implementation
to handle the different API response format.
"""

import asyncio
import os
from pathlib import Path
from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.tools import get_current_hash
from custom_goal_decomposer import CustomGoalDecomposer
from local_config import load_api_key_from_local_config

async def main():
    """Run the example with the custom parser."""
    print("Custom GoalDecomposer Example\n")
    
    # Load API key from local config.json
    success, api_key = load_api_key_from_local_config()
    if not success:
        print("Error: Could not find a valid API key in config.json")
        print("Please run setup_config.py to set up your API key.")
        return
    
    # Mask the API key for display
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"Found API key in local config.json: {masked_key}")
    
    # Path to test repository
    test_repo_path = os.path.join(os.getcwd(), "test-repo")
    print(f"Using test repository at: {test_repo_path}")
    
    # Get the actual git hash
    try:
        git_hash = await get_current_hash(test_repo_path)
        print(f"Retrieved actual git hash: {git_hash[:8]}...")
    except Exception as e:
        print(f"Error getting git hash: {str(e)}")
        print("Falling back to placeholder hash")
        git_hash = "abc123"  # Placeholder hash if actual hash retrieval fails
    
    # Create a sample goal
    goal = Goal(
        description="Build a REST API for a blog application with user authentication and content management",
        validation_criteria=[
            "Users can register and log in securely",
            "Users can create, update, and delete their blog posts",
            "Admin users can moderate all content",
            "API supports pagination and filtering of posts",
            "All endpoints are properly documented"
        ],
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
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )
    
    # Print input information
    print("\nInput:")
    print(f"Goal: {goal.description}")
    print("Validation Criteria:")
    for criterion in goal.validation_criteria:
        print(f"- {criterion}")
    print(f"Current State: {state.description}")
    print(f"Git Hash: {state.git_hash}")
    print(f"Test Repository Path: {test_repo_path}")
    print(f"Total Budget: {context.total_budget} points")
    print("\nDecomposing goal (this may take a moment)...")
    
    # Initialize the custom GoalDecomposer
    decomposer = CustomGoalDecomposer()
    
    # Decompose the goal
    try:
        strategy = await decomposer.decompose_goal(context)
        
        # Print the results
        print("\nOutput:")
        print(f"Strategy: {strategy.metadata.get('strategy_description', 'Not specified')}")
        print("\nSteps:")
        for i, step in enumerate(strategy.steps, 1):
            print(f"{i}. {step}")
        
        print(f"\nReasoning: {strategy.reasoning}")
        print(f"\nEstimated Points: {strategy.estimated_points}")
        
        print("\nThis is an actual response from the OpenAI API using our improved parser!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 