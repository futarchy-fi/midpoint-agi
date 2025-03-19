#!/usr/bin/env python
"""
Example script demonstrating the GoalDecomposer in action.

This script creates a sample task context and uses the GoalDecomposer
to break down a complex goal into actionable steps using the actual test repository.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.goal_decomposer import GoalDecomposer
from midpoint.agents.tools import get_current_hash

# Load environment variables from .env file
load_dotenv()

async def main():
    """Run the example with the actual test repository."""
    print("GoalDecomposer Example with Actual Repository\n")
    
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
    print("\nDecomposing goal...")
    
    # Initialize the GoalDecomposer
    decomposer = GoalDecomposer()
    
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
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 