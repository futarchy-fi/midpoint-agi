#!/usr/bin/env python
"""
Debugging script for the GoalDecomposer.

This script creates a sample task context and uses a debugging version of 
the GoalDecomposer to show the raw API response.
"""

import asyncio
import os
from pathlib import Path
from midpoint.agents.models import State, Goal, TaskContext, StrategyPlan
from midpoint.agents.goal_decomposer import GoalDecomposer
from midpoint.agents.tools import get_current_hash
from openai import AsyncOpenAI

# Import our local config helper
from local_config import load_api_key_from_local_config

class DebugGoalDecomposer(GoalDecomposer):
    """A debugging version of the GoalDecomposer that shows raw API responses."""
    
    async def decompose_goal(self, context):
        """Override to add debugging output."""
        # Validate inputs
        if not context.goal:
            raise ValueError("No goal provided in context")
        if context.total_budget <= 0:
            raise ValueError("Invalid points budget")
            
        # Track points for this operation
        print("Points consumed by goal_decomposition: 10")
        
        # Prepare the user prompt
        user_prompt = self._create_user_prompt(context)
        print("\nUser Prompt Sent to API:")
        print("-" * 80)
        print(user_prompt)
        print("-" * 80)
        
        # Call OpenAI API
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Print raw response
            raw_response = response.choices[0].message.content
            print("\nRaw API Response:")
            print("-" * 80)
            print(raw_response)
            print("-" * 80)
            
            # Parse the response
            print("\nParsing response...")
            strategy = self._parse_response(raw_response)
            print(f"Parsed Strategy: {len(strategy.steps)} steps, {strategy.estimated_points} points")
            print("Steps:")
            for i, step in enumerate(strategy.steps, 1):
                print(f"{i}. {step}")
                
            # Validate the strategy
            try:
                self._validate_strategy(strategy, context)
                print("\nStrategy validation passed.")
            except ValueError as e:
                print(f"\nStrategy validation failed: {str(e)}")
                raise
            
            return strategy
            
        except Exception as e:
            print(f"\nError during API call or processing: {str(e)}")
            raise

async def main():
    """Run the debugging example."""
    print("GoalDecomposer Debugging Example\n")
    
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
    
    # Initialize the debug GoalDecomposer
    decomposer = DebugGoalDecomposer()
    
    # Decompose the goal
    try:
        strategy = await decomposer.decompose_goal(context)
        
        # Print the results
        print("\nFinal Output:")
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