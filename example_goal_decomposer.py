#!/usr/bin/env python
"""
Example script demonstrating the GoalDecomposer in action.

This script creates a sample task context and uses the GoalDecomposer
to determine the next step toward achieving a complex goal.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.goal_decomposer import GoalDecomposer

# Load environment variables from .env file
load_dotenv()

async def main():
    """Run the example."""
    print("GoalDecomposer Example\n")
    
    # Create a temporary repository for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create some sample files in the repository
        (repo_path / "README.md").write_text("# Blog API\n\nA REST API for a blog application.")
        (repo_path / "requirements.txt").write_text("flask==2.0.1\nflask-restful==0.3.9\n")
        
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
            description="Initial repository state with basic project structure",
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
        
        # Print input information
        print("Input:")
        print(f"Goal: {goal.description}")
        print("Validation Criteria:")
        for criterion in goal.validation_criteria:
            print(f"- {criterion}")
        print(f"Current State: {state.description}")
        print(f"Repository Path: {state.repository_path}")
        print("\nDetermining next step...")
        
        # Initialize the GoalDecomposer
        decomposer = GoalDecomposer()
        
        # Determine the next step
        try:
            subgoal = await decomposer.determine_next_step(context)
            
            # Print the results
            print("\nOutput:")
            print(f"Next Step: {subgoal.next_step}")
            print("\nValidation Criteria:")
            for criterion in subgoal.validation_criteria:
                print(f"- {criterion}")
            
            print(f"\nReasoning: {subgoal.reasoning}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 