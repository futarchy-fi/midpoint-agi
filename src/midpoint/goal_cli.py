"""
Command-line interface for goal branch management.
"""

import os
import json
import argparse
import logging
import datetime
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .agents.models import Goal, SubgoalPlan

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Constants
GOAL_DIR = ".goal"
CHECKPOINT_DIR = f"{GOAL_DIR}/checkpoints"


def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path(GOAL_DIR)
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: {GOAL_DIR}")
    return goal_path


def ensure_checkpoint_dir():
    """Ensure the checkpoints directory exists."""
    checkpoint_path = Path(CHECKPOINT_DIR)
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created checkpoint directory: {CHECKPOINT_DIR}")
    return checkpoint_path


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


def get_current_hash():
    """Get the current git hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get current git hash: {e}")
        return None


def get_current_branch():
    """Get the current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get current branch: {e}")
        return None


def get_goal_id_from_branch(branch_name):
    """Extract goal ID from branch name."""
    # Branch naming convention: goal-G1-S1-abcdef (goal prefix, ID, random suffix)
    parts = branch_name.split('-')
    
    # Check if it's a goal branch
    if len(parts) < 3 or parts[0] != "goal":
        return None
    
    # Remove the 'goal' prefix and the random suffix
    if parts[1].startswith('G') and parts[1][1:].isdigit():
        if len(parts) >= 3 and parts[2].startswith('S') and parts[2][1:].isdigit():
            return f"{parts[1]}-{parts[2]}"
        return parts[1]
    
    return None


def get_parent_goal_id(goal_id):
    """Get the parent goal ID for a given goal ID."""
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal file not found: {goal_file}")
        return None
    
    try:
        with open(goal_file, 'r') as f:
            data = json.load(f)
            parent_id = data.get("parent_goal", "")
            if parent_id.endswith('.json'):
                parent_id = parent_id[:-5]  # Remove .json extension
            return parent_id
    except Exception as e:
        logging.error(f"Failed to read goal file: {e}")
        return None


def find_branch_for_goal(goal_id):
    """Find the git branch for a specific goal ID."""
    try:
        # Get all branches
        result = subprocess.run(
            ["git", "branch"],
            check=True,
            capture_output=True,
            text=True
        )
        
        branches = result.stdout.strip().split('\n')
        branches = [b.strip() for b in branches]
        branches = [b[2:] if b.startswith('* ') else b for b in branches]  # Remove the * from current branch
        
        # Look for branches with the goal ID
        for branch in branches:
            if branch.startswith(f"goal-{goal_id}-") or branch == f"goal-{goal_id}":
                return branch
        
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to list branches: {e}")
        return None


def get_recent_commits(count=10):
    """Get recent commits."""
    try:
        result = subprocess.run(
            ["git", "log", f"-{count}", "--oneline"],
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get recent commits: {e}")
        return []


def go_back_commits(steps=1):
    """Go back N commits on the current branch."""
    if steps < 1:
        logging.error("Number of steps must be at least 1")
        return False
    
    try:
        # Get current branch
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        # Get current hash before we move
        current_hash = get_current_hash()
        
        # Reset to N commits back
        reset_point = f"HEAD~{steps}"
        subprocess.run(
            ["git", "reset", "--hard", reset_point],
            check=True,
            capture_output=True,
            text=True
        )
        
        new_hash = get_current_hash()
        print(f"Moved back {steps} commit(s) on branch {current_branch}")
        print(f"From: {current_hash[:8]} -> To: {new_hash[:8]}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to go back commits: {e}")
        return False


def reset_to_commit(commit_id):
    """Reset to a specific commit on the current branch."""
    try:
        # Verify commit exists
        result = subprocess.run(
            ["git", "cat-file", "-t", commit_id],
            check=True,
            capture_output=True,
            text=True
        )
        
        if "commit" not in result.stdout:
            logging.error(f"Invalid commit ID: {commit_id}")
            return False
        
        # Get current branch
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        # Get current hash before we move
        current_hash = get_current_hash()
        
        # Reset to the specified commit
        subprocess.run(
            ["git", "reset", "--hard", commit_id],
            check=True,
            capture_output=True,
            text=True
        )
        
        new_hash = get_current_hash()
        print(f"Reset to commit {commit_id} on branch {current_branch}")
        print(f"From: {current_hash[:8]} -> To: {new_hash[:8]}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to reset to commit: {e}")
        return False


def create_checkpoint(message):
    """Create a labeled checkpoint for easy navigation."""
    try:
        # Get current hash
        current_hash = get_current_hash()
        if not current_hash:
            return False
        
        # Get current branch
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        # Create checkpoint file
        checkpoint_path = ensure_checkpoint_dir()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sanitize message for filename
        safe_message = "".join(c if c.isalnum() else "_" for c in message)
        
        checkpoint_file = checkpoint_path / f"{timestamp}_{safe_message}.json"
        
        # Prepare checkpoint data
        checkpoint_data = {
            "timestamp": timestamp,
            "message": message,
            "git_hash": current_hash,
            "branch": current_branch
        }
        
        # Save checkpoint file
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"Created checkpoint: {message}")
        print(f"Branch: {current_branch}, Commit: {current_hash[:8]}")
        return True
    except Exception as e:
        logging.error(f"Failed to create checkpoint: {e}")
        return False


def list_checkpoints():
    """List all checkpoints."""
    checkpoint_path = ensure_checkpoint_dir()
    
    checkpoint_files = list(checkpoint_path.glob("*.json"))
    if not checkpoint_files:
        print("No checkpoints found.")
        return []
    
    checkpoints = []
    for file_path in sorted(checkpoint_files, reverse=True):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                checkpoints.append(data)
        except:
            logging.warning(f"Failed to read checkpoint file: {file_path}")
    
    if checkpoints:
        print("Checkpoints:")
        for i, cp in enumerate(checkpoints):
            print(f"{i+1}. [{cp['timestamp']}] {cp['message']} (Branch: {cp['branch']}, Commit: {cp['git_hash'][:8]})")
    else:
        print("No valid checkpoints found.")
    
    return checkpoints


def go_to_parent_goal():
    """Go to the parent goal branch."""
    # Get current branch
    current_branch = get_current_branch()
    if not current_branch:
        return False
    
    # Extract goal ID from branch
    goal_id = get_goal_id_from_branch(current_branch)
    if not goal_id:
        logging.error(f"Current branch is not a goal branch: {current_branch}")
        return False
    
    # Get the parent goal ID
    parent_id = get_parent_goal_id(goal_id)
    if not parent_id:
        logging.error(f"No parent goal found for {goal_id}")
        return False
    
    # Find the branch for the parent goal
    parent_branch = find_branch_for_goal(parent_id)
    if not parent_branch:
        logging.error(f"No branch found for parent goal {parent_id}")
        return False
    
    # Switch to the parent branch
    try:
        subprocess.run(
            ["git", "checkout", parent_branch],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Switched to parent goal: {parent_id}")
        print(f"Branch: {parent_branch}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to switch to parent branch: {e}")
        return False


def go_to_subgoal(subgoal_id):
    """Go to a specific subgoal branch."""
    # Verify subgoal exists
    goal_path = ensure_goal_dir()
    subgoal_file = goal_path / f"{subgoal_id}.json"
    
    if not subgoal_file.exists():
        logging.error(f"Subgoal not found: {subgoal_id}")
        return False
    
    # Find the branch for the subgoal
    subgoal_branch = find_branch_for_goal(subgoal_id)
    if not subgoal_branch:
        logging.error(f"No branch found for subgoal {subgoal_id}")
        return False
    
    # Switch to the subgoal branch
    try:
        subprocess.run(
            ["git", "checkout", subgoal_branch],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Switched to subgoal: {subgoal_id}")
        print(f"Branch: {subgoal_branch}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to switch to subgoal branch: {e}")
        return False


def go_to_root_goal():
    """Go to the top-level goal branch."""
    # Get current branch
    current_branch = get_current_branch()
    if not current_branch:
        return False
    
    # Extract goal ID from branch
    goal_id = get_goal_id_from_branch(current_branch)
    if not goal_id:
        logging.error(f"Current branch is not a goal branch: {current_branch}")
        return False
    
    # If this is already a top-level goal (G1, G2, etc.), we're already there
    if '-' not in goal_id:
        print(f"Already at root goal: {goal_id}")
        return True
    
    # Extract the root goal ID (G part)
    root_id = goal_id.split('-')[0]
    
    # Find the branch for the root goal
    root_branch = find_branch_for_goal(root_id)
    if not root_branch:
        logging.error(f"No branch found for root goal {root_id}")
        return False
    
    # Switch to the root branch
    try:
        subprocess.run(
            ["git", "checkout", root_branch],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Switched to root goal: {root_id}")
        print(f"Branch: {root_branch}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to switch to root branch: {e}")
        return False


def list_subgoals(goal_id=None):
    """List subgoals for a specific goal ID or current branch."""
    # If no goal ID provided, use the current branch
    if not goal_id:
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        goal_id = get_goal_id_from_branch(current_branch)
        if not goal_id:
            logging.error(f"Current branch is not a goal branch: {current_branch}")
            return False
    
    # Get all goal files
    goal_path = ensure_goal_dir()
    goal_files = {}
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goal_files[data["goal_id"]] = data
        except:
            continue
    
    # Find direct subgoals
    subgoals = {k: v for k, v in goal_files.items() 
               if v.get("parent_goal") == goal_id or 
                  v.get("parent_goal") == f"{goal_id}.json"}
    
    if not subgoals:
        print(f"No subgoals found for goal {goal_id}")
        return []
    
    print(f"Subgoals for {goal_id}:")
    for subgoal_id in sorted(subgoals.keys()):
        subgoal = subgoals[subgoal_id]
        print(f"• {subgoal_id}: {subgoal['description']}")
    
    return list(subgoals.keys())


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Goal branch management commands")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Goal Management Commands
    # -----------------------
    # goal new <description>
    new_parser = subparsers.add_parser("new", help="Create a new top-level goal")
    new_parser.add_argument("description", help="Description of the goal")
    
    # goal sub <parent-id> <description>
    sub_parser = subparsers.add_parser("sub", help="Create a subgoal under the specified parent")
    sub_parser.add_argument("parent_id", help="Parent goal ID")
    sub_parser.add_argument("description", help="Description of the subgoal")
    
    # goal list
    subparsers.add_parser("list", help="List all goals and subgoals in tree format")
    
    # State Navigation Commands
    # ------------------------
    # goal back [steps]
    back_parser = subparsers.add_parser("back", help="Go back N commits on current goal branch")
    back_parser.add_argument("steps", nargs="?", type=int, default=1, help="Number of commits to go back")
    
    # goal reset <commit-id>
    reset_parser = subparsers.add_parser("reset", help="Reset to specific commit on current branch")
    reset_parser.add_argument("commit_id", help="Commit ID to reset to")
    
    # goal checkpoint "message"
    checkpoint_parser = subparsers.add_parser("checkpoint", help="Create labeled checkpoint for easy navigation")
    checkpoint_parser.add_argument("message", help="Checkpoint message")
    
    # goal checkpoints
    subparsers.add_parser("checkpoints", help="List all checkpoints")
    
    # Hierarchy Navigation Commands
    # ---------------------------
    # goal up
    subparsers.add_parser("up", help="Go to parent goal branch")
    
    # goal down <subgoal-id>
    down_parser = subparsers.add_parser("down", help="Go to specific subgoal branch")
    down_parser.add_argument("subgoal_id", help="Subgoal ID to navigate to")
    
    # goal root
    subparsers.add_parser("root", help="Go to top-level goal")
    
    # goal subs
    subparsers.add_parser("subs", help="List available subgoals for current goal")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "new":
        create_new_goal(args.description)
    elif args.command == "sub":
        create_new_subgoal(args.parent_id, args.description)
    elif args.command == "list":
        list_goals()
    elif args.command == "back":
        go_back_commits(args.steps)
    elif args.command == "reset":
        reset_to_commit(args.commit_id)
    elif args.command == "checkpoint":
        create_checkpoint(args.message)
    elif args.command == "checkpoints":
        list_checkpoints()
    elif args.command == "up":
        go_to_parent_goal()
    elif args.command == "down":
        go_to_subgoal(args.subgoal_id)
    elif args.command == "root":
        go_to_root_goal()
    elif args.command == "subs":
        list_subgoals()
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 