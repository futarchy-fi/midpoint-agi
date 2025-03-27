#!/usr/bin/env python
"""
Command-line tool to show memory history for a goal/subgoal/task.
"""

import os
import json
import argparse
import logging
import subprocess
import asyncio
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Constants
GOAL_DIR = ".goal"

def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path(GOAL_DIR)
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: {GOAL_DIR}")
    return goal_path

async def show_memory_history(goal_id: str, debug: bool = False, quiet: bool = False):
    """Show memory history for a goal/subgoal/task.
    
    Args:
        goal_id: ID of the goal/subgoal/task to show memory history for
        debug: Whether to show debug output
        quiet: Whether to only show warnings and final result
    """
    # Configure logging
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Get goal file path
    goal_path = ensure_goal_dir() / f"{goal_id}.json"
    if not goal_path.exists():
        logging.error(f"Goal file not found: {goal_path}")
        return False
    
    # Load goal data
    try:
        with open(goal_path, 'r') as f:
            goal_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load goal file: {e}")
        return False
    
    # Get memory state from current_state
    if "current_state" not in goal_data:
        logging.error("No current state found in goal file")
        return False
    
    current_state = goal_data["current_state"]
    memory_hash = current_state.get("memory_hash")
    memory_repo_path = current_state.get("memory_repository_path")
    
    if not memory_hash or not memory_repo_path:
        logging.error("No memory hash or repository path found in goal state")
        return False
    
    logging.info(f"Showing memory history for {goal_id}")
    logging.info(f"Memory repository: {memory_repo_path}")
    logging.info(f"Memory hash: {memory_hash[:8]}")
    
    try:
        # Use git log to show history up to the memory hash
        process = await asyncio.create_subprocess_exec(
            "git", "log", "--oneline", memory_hash,
            cwd=memory_repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logging.error(f"Failed to get git log: {stderr.decode()}")
            return False
        
        # Print the log
        print("\nMemory Repository History:")
        print("=" * 80)
        print(stdout.decode())
        print("=" * 80)
        
        return True
        
    except Exception as e:
        logging.error(f"Error showing memory history: {e}")
        return False

def main():
    """Main entry point for the goal memory CLI."""
    parser = argparse.ArgumentParser(description="Show memory history for a goal/subgoal/task")
    parser.add_argument("goal_id", help="ID of the goal/subgoal/task to show memory history for")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and final result")
    
    args = parser.parse_args()
    
    # Run the async function
    asyncio.run(show_memory_history(args.goal_id, args.debug, args.quiet))

if __name__ == "__main__":
    main() 