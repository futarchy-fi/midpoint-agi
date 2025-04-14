#!/usr/bin/env python
"""
Real API test for the GoalDecomposer.

This script tests the GoalDecomposer with the real OpenAI API.
"""

import asyncio
import os
import json
import tempfile
from pathlib import Path
from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.goal_decomposer import GoalDecomposer
from midpoint.agents.config import get_openai_api_key
import pytest

# Get API key from environment
api_key = get_openai_api_key()
if not api_key:
    pytest.skip("OPENAI_API_KEY not set in environment")

async def test_real_api():
    """Test the GoalDecomposer with the real OpenAI API."""
    print("Running real API test for GoalDecomposer\n")
    
    # Create a temporary repository for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create some sample files in the repository
        (repo_path / "README.md").write_text("# Test Project\n\nA sample project for testing.")
        (repo_path / "requirements.txt").write_text("flask==2.0.1\nflask-sqlalchemy==2.5.1\n")
        
        # Create src directory with a sample app.py file
        src_dir = repo_path / "src"
        src_dir.mkdir(exist_ok=True)
        (src_dir / "app.py").write_text("""from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return {'message': 'Hello, world!'}

if __name__ == '__main__':
    app.run(debug=True)
""")
        
        # Create a sample goal
        goal = Goal(
            description="Add authentication to the Flask API with JWT tokens",
            validation_criteria=[
                "Users can register with email and password",
                "Users can login and receive JWT tokens",
                "Protected routes require valid tokens",
                "Token expiration and refresh functionality"
            ],
            success_threshold=0.8
        )
        
        # Create a sample state
        state = State(
            git_hash="abc123",  # Placeholder hash
            description="Basic Flask API without authentication",
            repository_path=str(repo_path)
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
        
        # Initialize the GoalDecomposer
        decomposer = GoalDecomposer()
        
        # Determine the next step
        try:
            print("Input:")
            print(f"Goal: {goal.description}")
            print("Validation Criteria:")
            for criterion in goal.validation_criteria:
                print(f"- {criterion}")
            print(f"Current State: {state.description}")
            print(f"Repository Path: {state.repository_path}")
            print("\nDetermining next step...")
            
            # Timing the API call
            import time
            start_time = time.time()
            
            subgoal = await decomposer.determine_next_state(context)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Print the results
            print("\nOutput (after {:.2f} seconds):".format(duration))
            print(f"Next Step: {subgoal.next_step}")
            print("\nValidation Criteria:")
            for criterion in subgoal.validation_criteria:
                print(f"- {criterion}")
            
            print(f"\nReasoning: {subgoal.reasoning}")
            
            # Print the tool usage metadata if available
            if "tool_usage" in subgoal.metadata:
                print("\nTool Usage:")
                for tool in subgoal.metadata["tool_usage"]:
                    print(f"- {tool}")
            
            print("\nTest completed successfully!")
            return True
        except Exception as e:
            print(f"Error: {str(e)}")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_real_api())
    exit(0 if success else 1) 