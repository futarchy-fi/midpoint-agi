#!/usr/bin/env python3
"""
Example script demonstrating the Advanced AGI System with Recursive Goal Decomposition
"""

import asyncio
import os
import tempfile
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

from agents.models import State, Goal
from agents.orchestrator import solve_problem

# Load the API key from .env file
def load_env():
    # Try to load from current directory
    if Path('.env').exists():
        load_dotenv()
    # Try to load from parent directory
    elif Path('..', '.env').exists():
        load_dotenv(Path('..', '.env'))
    # Try to load from project root
    elif Path(Path(__file__).parent.parent, '.env').exists():
        load_dotenv(Path(Path(__file__).parent.parent, '.env'))
    
    if not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in .env file or environment variables")
        return False
    return True

async def run_command(command, cwd=None):
    """Run a shell command and return its output"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.stderr}")
        return None

async def setup_test_repo():
    """Set up a test repository for the example"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Initialize git repository
    await run_command("git init", temp_dir)
    
    # Set git config for the repository
    await run_command("git config user.email 'example@example.com'", temp_dir)
    await run_command("git config user.name 'Example User'", temp_dir)
    
    # Create initial files
    readme_content = """# Test Repository

This is a test repository for demonstrating the advanced AGI system with recursive goal decomposition.
"""
    
    readme_path = Path(temp_dir) / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    # Commit initial files
    await run_command("git add README.md", temp_dir)
    await run_command("git commit -m 'Initial commit'", temp_dir)
    
    # Get the commit hash
    git_hash = await run_command("git rev-parse HEAD", temp_dir)
    
    return temp_dir, git_hash

async def main():
    """Main function for the example"""
    # Load environment variables
    if not load_env():
        return
    
    print("🚀 Starting Advanced AGI System Example\n")
    
    # Set up test repository
    print("🛠️ Setting up test repository...")
    repo_path, git_hash = await setup_test_repo()
    print(f"✅ Repository initialized at {repo_path} with hash {git_hash[:7]}\n")
    
    # Create trace file
    trace_file = Path(repo_path) / "execution_trace.md"
    
    # Define the initial state
    initial_state = State(
        git_hash=git_hash,
        description="Initial repository with README.md",
        metadata={
            "repo_path": repo_path
        }
    )
    
    # Define the goal
    goal = Goal(
        description="Create a simple Python calculator application",
        validation_criteria=[
            "Repository should contain a calculator.py file",
            "Calculator should implement addition, subtraction, multiplication, and division operations",
            "Calculator should have a simple CLI interface for performing calculations",
            "Code should be properly documented with docstrings",
            "Repository should include a updated README.md with usage instructions"
        ],
        success_threshold=0.8  # We'll accept 80% success
    )
    
    # Set up the points budget for this task
    total_budget = 5000
    
    # Save the current directory and change to the repository
    original_dir = os.getcwd()
    os.chdir(repo_path)
    
    try:
        print(f"🎯 Goal: {goal.description}")
        print(f"💰 Total budget: {total_budget} points")
        print("🔄 Starting problem solving process...\n")
        
        # Solve the problem
        result = await solve_problem(
            initial_state=initial_state,
            goal=goal,
            total_budget=total_budget,
            trace_file=str(trace_file)
        )
        
        # Print results
        print("\n🔍 Results:")
        print(f"   Success: {'✅ Yes' if result['success'] else '❌ No'}")
        print(f"   Total iterations: {result['iterations']}")
        print(f"   Points consumed: {result['points_consumed']}/{total_budget}")
        print(f"   Trace file: {trace_file}")
        
        # If successful, list the files created
        if result['success']:
            files = await run_command("find . -type f -not -path '*/\.*' | sort")
            print("\n📁 Files created:")
            for file in files.split('\n'):
                print(f"   - {file}")
    finally:
        # Change back to the original directory
        os.chdir(original_dir)
    
    print("\n✨ Example completed! ✨")

if __name__ == "__main__":
    asyncio.run(main()) 