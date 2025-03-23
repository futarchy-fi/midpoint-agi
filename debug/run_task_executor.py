#!/usr/bin/env python3
"""
Command-Line Interface for the TaskExecutor agent.

This script provides a simple CLI for running tasks directly using the TaskExecutor,
similar to how the GoalDecomposer CLI works.
"""

import argparse
import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

from midpoint.agents.task_executor import TaskExecutor, configure_logging
from midpoint.agents.models import Goal, State, TaskContext, ExecutionResult
from midpoint.agents.tools.git_tools import get_current_hash, get_current_branch
from midpoint.agents import config

async def run_executor(args):
    """Run the TaskExecutor with the given arguments."""
    # Configure logging based on arguments
    log_file = configure_logging(debug=args.debug, quiet=args.quiet, log_dir_path=args.output_dir)
    logger = logging.getLogger('TaskExecutor.CLI')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load task from file if provided
    task_description = args.task
    if args.input_file:
        try:
            logger.info(f"Loading task from input file: {args.input_file}")
            with open(args.input_file, 'r') as f:
                subgoal_data = json.load(f)
                # Extract task description from subgoal file
                if 'next_step' in subgoal_data:
                    task_description = subgoal_data['next_step']
                    logger.info(f"Using task from input file: {task_description}")
                else:
                    logger.error("Input file does not contain required 'next_step' field")
                    raise ValueError("Invalid input file: missing 'next_step' field")
                    
                # Extract validation criteria if available
                validation_criteria = subgoal_data.get('validation_criteria', [])
                if validation_criteria:
                    logger.info(f"Found {len(validation_criteria)} validation criteria in input file")
                else:
                    logger.warning("No validation criteria found in input file")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON in input file: {str(e)}")
            sys.exit(1)
        except FileNotFoundError as e:
            logger.error(f"Input file not found: {str(e)}")
            sys.exit(1)
        except ValueError as e:
            # Re-raise value errors (like invalid file format) to exit with error
            logger.error(str(e))
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error loading input file: {str(e)}")
            sys.exit(1)
    elif not task_description:
        logger.error("Error: Either --task or --input-file must be provided")
        sys.exit(1)
    
    # Get current Git hash and branch
    repo_path = args.repo_path
    try:
        logger.info(f"Checking repository state at {repo_path}")
        git_hash = await get_current_hash(repo_path)
        branch = await get_current_branch(repo_path)
        logger.info(f"Current branch: {branch}, hash: {git_hash[:8]}")
    except Exception as e:
        logger.error(f"Error accessing git repository: {str(e)}")
        sys.exit(1)
    
    # Create goal
    logger.info("Creating task context")
    goal = Goal(
        description=args.goal or "Execute the specified task",
        validation_criteria=validation_criteria if 'validation_criteria' in locals() else [],
        success_threshold=0.8
    )
    
    # Create initial state
    state = State(
        git_hash=git_hash,
        description=f"Repository state on branch {branch}",
        repository_path=repo_path
    )
    
    # Create task context
    context = TaskContext(
        state=state,
        goal=goal,
        iteration=0,
        execution_history=[],
        metadata={}
    )
    
    # Initialize task executor
    logger.info("Initializing TaskExecutor")
    executor = TaskExecutor()
    
    try:
        # Execute the task
        logger.info(f"\nExecuting task: {task_description}")
        logger.info(f"Repository: {repo_path}")
        logger.info(f"Branch: {branch}")
        logger.info(f"Initial commit: {git_hash[:8]}")
        
        result = await executor.execute_task(context, task_description)
        
        # Save the result to output directory
        timestamp = int(result.execution_time) if result.execution_time else int(asyncio.get_event_loop().time())
        output_file = Path(args.output_dir) / f"task_result_{timestamp}.json"
        
        logger.info(f"Saving execution results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(
                {
                    "task": task_description,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "branch_name": result.branch_name,
                    "git_hash": result.git_hash,
                    "error_message": result.error_message,
                    "validation_results": result.validation_results if hasattr(result, 'validation_results') else []
                },
                f,
                indent=2
            )
        
        # Display results
        if result.success:
            logger.info(f"\n✅ Task completed successfully in {result.execution_time:.2f} seconds")
            logger.info(f"Branch: {result.branch_name}")
            logger.info(f"Final commit: {result.git_hash[:8]}")
            
            # Display validation results if any
            if hasattr(result, 'validation_results') and result.validation_results:
                logger.info("\nValidation Results:")
                for validation in result.validation_results:
                    logger.info(f"- {validation}")
        else:
            logger.error(f"\n❌ Task execution failed in {result.execution_time:.2f} seconds")
            logger.error(f"Error: {result.error_message}")
        
        logger.info(f"\nOutput saved to {output_file}")
        logger.info(f"Full logs saved to {log_file}")
        return result
        
    except Exception as e:
        logger.error(f"Error during task execution: {str(e)}")
        import traceback
        logger.debug(f"Exception traceback:\n{traceback.format_exc()}")
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='Run the TaskExecutor with specified parameters')
    
    # Required arguments
    parser.add_argument('--repo-path', required=True, help='Path to the repository to work with')
    
    # Task specification (one of these is required)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument('--task', help='Task description to execute')
    task_group.add_argument('--input-file', help='Path to subgoal JSON file containing the task')
    
    # Optional arguments
    parser.add_argument('--goal', help='Overall goal context for the task')
    parser.add_argument('--output-dir', default='logs', help='Directory to save output files and logs')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logging')
    parser.add_argument('--quiet', action='store_true', help='Minimize console output (show only warnings and errors)')
    parser.add_argument('--no-commit', action='store_true', help='Prevent automatic commits')
    
    args = parser.parse_args()
    
    try:
        # Run the executor with proper asyncio handling
        asyncio.run(run_executor(args))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Handle the case when we're already in an event loop
            print("Warning: Already in an event loop. Using get_event_loop() instead.")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_executor(args))
        else:
            raise

if __name__ == "__main__":
    main() 