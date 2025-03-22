#!/usr/bin/env python3
import argparse
import os
import sys
import asyncio
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from midpoint.agents.goal_decomposer import GoalDecomposer
from midpoint.agents.models import Goal, State, TaskContext
from midpoint.agents import config

async def run_decomposer(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the goal
    goal = Goal(
        description=args.objective,
        validation_criteria=[],  # Let the decomposer determine these
        success_threshold=0.8
    )
    
    # Create initial state
    state = State(
        git_hash="initial",
        description="Initial repository state",
        repository_path=args.repo
    )
    
    # Create task context
    context = TaskContext(
        state=state,
        goal=goal,
        iteration=0,
        execution_history=[]
    )
    
    # Initialize the goal decomposer
    decomposer = GoalDecomposer()
    
    try:
        # Get the next step
        result = await decomposer.determine_next_step(context)
        print(f"\nNext step: {result.next_step}")
        print("\nValidation criteria:")
        for criterion in result.validation_criteria:
            print(f"- {criterion}")
        print(f"\nReasoning: {result.reasoning}")
        print(f"\nOutput saved to {args.output_dir}")
    except Exception as e:
        print(f"Error during goal decomposition: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run the goal decomposer with specified parameters')
    parser.add_argument('--repo', required=True, help='Path to the repository to analyze')
    parser.add_argument('--branch', required=True, help='Git branch to analyze')
    parser.add_argument('--objective', required=True, help='Objective to decompose')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum depth for goal decomposition')
    parser.add_argument('--config', help='Path to custom config file')
    
    args = parser.parse_args()
    asyncio.run(run_decomposer(args))

if __name__ == "__main__":
    main() 