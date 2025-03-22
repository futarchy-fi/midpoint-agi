#!/usr/bin/env python3
import argparse
import os
import sys
import asyncio
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from midpoint.orchestrator import Orchestrator
from midpoint.agents.models import Goal, State, TaskContext

async def run_orchestration(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the goal
    goal = Goal(
        description=args.objective,
        validation_criteria=[],  # Let the decomposer determine these
        success_threshold=0.8
    )
    
    # Initialize the orchestrator
    orchestrator = Orchestrator()
    
    try:
        # Run the orchestration
        result = await orchestrator.run(
            repo_path=args.repo,
            goal=goal,
            max_iterations=args.max_iterations,
            checkpoint_path=os.path.join(args.output_dir, "checkpoint.json") if args.save_checkpoint else None
        )
        
        # Print the results
        if result.success:
            print("\nOrchestration completed successfully!")
            print(f"Final git hash: {result.final_state.git_hash}")
        else:
            print("\nOrchestration failed!")
            print(f"Error: {result.error_message}")
        
        if result.execution_history:
            print("\nExecution History:")
            for i, entry in enumerate(result.execution_history, 1):
                print(f"\nIteration {i}:")
                print(f"Subgoal: {entry['subgoal']}")
                print(f"Git Hash: {entry['git_hash']}")
                print(f"Validation Score: {entry['validation_score']:.2f}")
                print(f"Execution Time: {entry['execution_time']:.2f}s")
        
        print(f"\nOutput saved to {args.output_dir}")
    except Exception as e:
        print(f"Error during orchestration: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run the Midpoint orchestrator with specified parameters')
    parser.add_argument('--repo', required=True, help='Path to the repository to analyze')
    parser.add_argument('--branch', required=True, help='Git branch to analyze')
    parser.add_argument('--objective', required=True, help='Objective to decompose')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    parser.add_argument('--max-iterations', type=int, default=10, help='Maximum number of iterations')
    parser.add_argument('--save-checkpoint', action='store_true', help='Save checkpoint after each iteration')
    
    args = parser.parse_args()
    asyncio.run(run_orchestration(args))

if __name__ == "__main__":
    main() 