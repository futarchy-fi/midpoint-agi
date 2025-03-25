"""
Command-line interface for goal branch management.
"""

import os
import json
import argparse
import logging
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .agents.models import Goal, SubgoalPlan

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


def generate_goal_id(parent_id=None):
    """Generate a goal ID in format G1 or G1-S1."""
    goal_path = ensure_goal_dir()
    
    if not parent_id:
        # Find next available top-level goal number
        existing = [f for f in goal_path.glob("G*.json")]
        next_num = len(existing) + 1
        return f"G{next_num}"
    else:
        # Find next available subgoal number for parent
        parent_base = parent_id.split('.')[0]  # Remove .json extension if present
        existing = [f for f in goal_path.glob(f"{parent_base}-S*.json")]
        next_num = len(existing) + 1
        return f"{parent_base}-S{next_num}"


def create_goal_file(goal_id, description, parent_id=None):
    """Create a goal file with the given ID and description."""
    goal_path = ensure_goal_dir()
    
    # Prepare the goal content
    goal_content = {
        "goal_id": goal_id,
        "description": description,
        "parent_goal": parent_id or "",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Write the goal file
    output_file = goal_path / f"{goal_id}.json"
    with open(output_file, 'w') as f:
        json.dump(goal_content, f, indent=2)
        
    logging.info(f"Created goal file: {output_file}")
    return str(output_file)


def list_goals():
    """List all goals and subgoals in tree format."""
    goal_path = ensure_goal_dir()
    
    # Get all goal files
    goal_files = {}
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goal_files[data["goal_id"]] = data
        except:
            logging.warning(f"Failed to read goal file: {file_path}")
    
    # Find top-level goals
    top_goals = {k: v for k, v in goal_files.items() if not v["parent_goal"]}
    
    # Build tree structure
    def print_goal_tree(goal_id, depth=0):
        if goal_id not in goal_files:
            return
            
        goal = goal_files[goal_id]
        indent = "  " * depth
        print(f"{indent}• {goal_id}: {goal['description']}")
        
        # Find and print children
        children = {k: v for k, v in goal_files.items() 
                   if v.get("parent_goal") == goal_id or 
                      v.get("parent_goal") == f"{goal_id}.json"}
        
        for child_id in sorted(children.keys()):
            print_goal_tree(child_id, depth + 1)
    
    # Print all top-level goals and their subgoals
    if top_goals:
        print("Goal Tree:")
        for goal_id in sorted(top_goals.keys()):
            print_goal_tree(goal_id)
    else:
        print("No goals found.")


def create_new_goal(description):
    """Create a new top-level goal."""
    goal_id = generate_goal_id()
    create_goal_file(goal_id, description)
    print(f"Created new goal {goal_id}: {description}")
    return goal_id


def create_new_subgoal(parent_id, description):
    """Create a new subgoal under the specified parent."""
    # Verify parent exists
    goal_path = ensure_goal_dir()
    parent_file = goal_path / f"{parent_id}.json"
    
    if not parent_file.exists():
        logging.error(f"Parent goal {parent_id} not found")
        return None
    
    # Generate subgoal ID
    subgoal_id = generate_goal_id(parent_id)
    
    # Create subgoal file
    create_goal_file(subgoal_id, description, parent_id)
    print(f"Created new subgoal {subgoal_id} under {parent_id}: {description}")
    return subgoal_id


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Goal branch management commands")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # goal new <description>
    new_parser = subparsers.add_parser("new", help="Create a new top-level goal")
    new_parser.add_argument("description", help="Description of the goal")
    
    # goal sub <parent-id> <description>
    sub_parser = subparsers.add_parser("sub", help="Create a subgoal under the specified parent")
    sub_parser.add_argument("parent_id", help="Parent goal ID")
    sub_parser.add_argument("description", help="Description of the subgoal")
    
    # goal list
    list_parser = subparsers.add_parser("list", help="List all goals and subgoals in tree format")
    
    args = parser.parse_args()
    
    if args.command == "new":
        create_new_goal(args.description)
    elif args.command == "sub":
        create_new_subgoal(args.parent_id, args.description)
    elif args.command == "list":
        list_goals()
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 