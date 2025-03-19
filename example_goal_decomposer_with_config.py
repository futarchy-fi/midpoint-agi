#!/usr/bin/env python
"""
Example script demonstrating the GoalDecomposer in action using the config.json API key.

This script creates a sample task context and uses the GoalDecomposer
to break down a complex goal into actionable steps.
"""

import asyncio
import os
import json
from pathlib import Path
from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.goal_decomposer import GoalDecomposer
from openai import AsyncOpenAI

# Override the API key with the one from config.json
def load_api_key_from_config():
    """Load the OpenAI API key from the config.json file."""
    config_path = Path.home() / ".midpoint" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            api_key = config.get('openai', {}).get('api_key')
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                return True, api_key
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass
    return False, None

# Modified version of the GoalDecomposer to print debug info
class DebugGoalDecomposer(GoalDecomposer):
    """A debugging version of the GoalDecomposer that prints more information."""
    
    async def decompose_goal(self, context):
        """Override to add debugging output."""
        # Prepare the user prompt
        user_prompt = self._create_user_prompt(context)
        print("\nUser Prompt Sent to API:")
        print("-" * 50)
        print(user_prompt)
        print("-" * 50)
        
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
            print("-" * 50)
            print(raw_response)
            print("-" * 50)
            
            # Parse the response
            strategy = self._parse_response(raw_response)
            
            # Validate the strategy
            self._validate_strategy(strategy, context)
            
            return strategy
            
        except Exception as e:
            print(f"\nError during API call or processing: {str(e)}")
            raise

async def main():
    """Run the example."""
    print("GoalDecomposer Example with Config API Key\n")
    
    # Load API key from config.json
    success, api_key = load_api_key_from_config()
    if not success:
        print("Error: Could not find a valid API key in ~/.midpoint/config.json")
        print("Please make sure your API key is set up correctly.")
        return
    
    print(f"Found API key in config.json! Using it for the GoalDecomposer.")
    print(f"API Key starts with: {api_key[:7]}...\n")
    
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
        git_hash="abc123",  # Placeholder hash
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
    print("Input:")
    print(f"Goal: {goal.description}")
    print("Validation Criteria:")
    for criterion in goal.validation_criteria:
        print(f"- {criterion}")
    print(f"Current State: {state.description}")
    print(f"Total Budget: {context.total_budget} points")
    print("\nDecomposing goal (this may take a moment)...")
    
    # Initialize the GoalDecomposer with debug printing
    decomposer = DebugGoalDecomposer()
    
    # Decompose the goal
    try:
        strategy = await decomposer.decompose_goal(context)
        
        # Print the results
        print("\nProcessed Output:")
        print(f"Strategy: {strategy.metadata.get('strategy_description', 'Not specified')}")
        print("\nSteps:")
        for i, step in enumerate(strategy.steps, 1):
            print(f"{i}. {step}")
        
        print(f"\nReasoning: {strategy.reasoning}")
        print(f"\nEstimated Points: {strategy.estimated_points}")
        
        print("\nThis is an actual response from the OpenAI API!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 