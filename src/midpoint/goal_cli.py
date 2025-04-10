"""
Command-line interface for goal management.
"""

import os
import json
import argparse
import logging
import datetime
import subprocess
import asyncio
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import sys

from .agents.models import Goal, SubgoalPlan, TaskContext, ExecutionResult, MemoryState, State
from .agents.goal_decomposer import decompose_goal as agent_decompose_goal
from .agents.task_executor import TaskExecutor, configure_logging as configure_executor_logging
from .agents.tools.git_tools import get_current_hash, get_current_branch

# Import validator for automated validation
from .agents.goal_validator import GoalValidator

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Constants
GOAL_DIR = ".goal"
VISUALIZATION_DIR = f"{GOAL_DIR}/visualization"


def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path(GOAL_DIR)
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: {GOAL_DIR}")
    return goal_path


def ensure_visualization_dir():
    """Ensure the visualization directory exists."""
    vis_path = Path(VISUALIZATION_DIR)
    if not vis_path.exists():
        vis_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created visualization directory: {VISUALIZATION_DIR}")
    return vis_path


def generate_goal_id(parent_id=None, is_task=False):
    """Generate a goal ID in format G1, S1, or T1."""
    goal_path = ensure_goal_dir()
    
    if is_task:
        # Find next available task number
        max_num = 0
        for file_path in goal_path.glob("T*.json"):
            # Match only files with pattern T followed by digits and .json
            match = re.match(r"T(\d+)\.json$", file_path.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        
        # Next task number is one more than the maximum found
        next_num = max_num + 1
        return f"T{next_num}"
    elif parent_id:
        # This is a subgoal - find next available subgoal number
        max_num = 0
        for file_path in goal_path.glob("S*.json"):
            # Match only files with pattern S followed by digits and .json
            match = re.match(r"S(\d+)\.json$", file_path.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        
        # Next subgoal number is one more than the maximum found
        next_num = max_num + 1
        return f"S{next_num}"
    else:
        # This is a top-level goal - find next available goal number
        max_num = 0
        for file_path in goal_path.glob("G*.json"):
            # Match only files with pattern G followed by digits and .json
            match = re.match(r"G(\d+)\.json$", file_path.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        
        # Next goal number is one more than the maximum found
        next_num = max_num + 1
        return f"G{next_num}"


def create_goal_file(goal_id, description, parent_id=None, branch_name=None):
    """Create a goal file with the given information."""
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    # Get current repository information
    current_hash = get_current_hash()
    repo_path = os.getcwd()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize memory repository if needed
    memory_repo_path = os.path.expanduser("~/.midpoint/memory")
    memory_hash = None
    
    try:
        # Import here to avoid circular imports
        from scripts.init_memory_repo import init_memory_repo
        
        # Initialize memory repository
        memory_repo_info = init_memory_repo(memory_repo_path, None, None)
        memory_repo_path = memory_repo_info["path"]
        
        # Get the current memory hash
        memory_hash = get_current_hash(memory_repo_path)
        logging.info(f"Initialized memory repository at {memory_repo_path} with hash {memory_hash[:8]}")
    except Exception as e:
        logging.warning(f"Failed to initialize memory repository: {e}")
        # Continue without memory repository - it's not critical for goal creation
    
    # If we have a parent goal, load its current state
    if parent_id:
        parent_file = goal_path / f"{parent_id}.json"
        if parent_file.exists():
            try:
                with open(parent_file, 'r') as f:
                    parent_data = json.load(f)
                    if "current_state" in parent_data:
                        # Use parent's current state values
                        current_hash = parent_data["current_state"]["git_hash"]
                        repo_path = parent_data["current_state"]["repository_path"]
                        memory_hash = parent_data["current_state"].get("memory_hash")
                        memory_repo_path = parent_data["current_state"].get("memory_repository_path")
                        logging.info(f"Using parent goal {parent_id} state")
            except Exception as e:
                logging.warning(f"Failed to load parent goal state: {e}")
    
    # Prepare goal data
    initial_state = {
        "git_hash": current_hash,
        "repository_path": repo_path,
        "description": f"Initial state before creating goal: {goal_id}",
        "timestamp": timestamp,
        "memory_hash": memory_hash,
        "memory_repository_path": memory_repo_path
    }
    
    goal_data = {
        "goal_id": goal_id,
        "description": description,
        "parent_goal": parent_id or "",
        "timestamp": timestamp,
        "is_task": False,
        "requires_further_decomposition": True,
        "branch_name": branch_name,
        "initial_state": initial_state,
        "current_state": initial_state.copy()  # Set current_state to be same as initial_state
    }
    
    # Write the goal file
    with open(goal_file, 'w') as f:
        json.dump(goal_data, f, indent=2)
    
    logging.info(f"Created goal file: {goal_file}")
    return str(goal_file)


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
    # Store original branch
    try:
        original_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get original branch: {e}")
        return None

    # Check for uncommitted changes and stash them if needed
    has_changes = False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True
        )
        has_changes = bool(result.stdout.strip())
        if has_changes:
            subprocess.run(
                ["git", "stash", "push", "-m", "Stashing changes before creating new goal"],
                check=True,
                capture_output=True,
                text=True
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to check git status or stash changes: {e}")
        return None

    try:
        # Generate goal ID
        goal_id = generate_goal_id()
        
        # Create a branch name from the description
        branch_name = f"goal-{goal_id}"
        
        # Create and checkout the branch
        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create branch: {e}")
            return None
        
        # Create goal file with branch information
        goal_data = {
            "goal_id": goal_id,
            "description": description,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "branch_name": branch_name,
            "is_task": False,
            "requires_further_decomposition": True
        }
        
        create_goal_file(goal_id, description, branch_name=branch_name)
        print(f"Created new goal {goal_id}: {description}")
        print(f"Created branch: {branch_name}")

        # Switch back to original branch
        try:
            subprocess.run(
                ["git", "checkout", original_branch],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to switch back to original branch: {e}")
            # Don't return None here as the goal was created successfully
        
        return goal_id
    finally:
        # Always restore stashed changes if we stashed them
        if has_changes:
            try:
                subprocess.run(
                    ["git", "stash", "pop"],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to restore stashed changes: {e}")
                # Don't return None here as the goal was created successfully


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


def create_new_task(parent_id, description):
    """Create a new directly executable task under the specified parent."""
    # Verify parent exists
    goal_path = ensure_goal_dir()
    parent_file = goal_path / f"{parent_id}.json"
    
    if not parent_file.exists():
        logging.error(f"Parent goal {parent_id} not found")
        return None
    
    # Load parent's current state
    try:
        with open(parent_file, 'r') as f:
            parent_data = json.load(f)
            if "current_state" not in parent_data:
                logging.error(f"Parent goal {parent_id} has no current state")
                return None
            parent_state = parent_data["current_state"]
    except Exception as e:
        logging.error(f"Failed to load parent state: {e}")
        return None
    
    # Generate task ID
    task_id = generate_goal_id(parent_id, is_task=True)
    
    # Create initial state from parent's current state
    initial_state = {
        "git_hash": parent_state["git_hash"],
        "repository_path": parent_state["repository_path"],
        "description": f"Initial state before executing task: {task_id}",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "memory_hash": parent_state.get("memory_hash"),
        "memory_repository_path": parent_state.get("memory_repository_path")
    }
    
    # Create task file with the expected structure
    task_data = {
        "goal_id": task_id,
        "description": description,
        "parent_goal": parent_id,
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "is_task": True,
        "requires_further_decomposition": False,
        "initial_state": initial_state,
        "current_state": initial_state.copy()  # Set current_state to be same as initial_state
    }
    
    # Write the task file
    task_file = goal_path / f"{task_id}.json"
    with open(task_file, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    print(f"Created new task {task_id} under {parent_id}: {description}")
    return task_id


def get_current_hash(repo_path: Optional[str] = None) -> str:
    """Get the current git hash.
    
    Args:
        repo_path: Optional path to the git repository. If not provided, uses current directory.
        
    Returns:
        The current git hash as a string.
    """
    try:
        # If no repo_path provided, use current directory
        if not repo_path:
            repo_path = os.getcwd()
            
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
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
    # Branch naming convention: goal-G1-abcdef, goal-S1-abcdef, or goal-T1-abcdef
    parts = branch_name.split('-')
    
    # Check if it's a goal branch
    if len(parts) < 2 or parts[0] != "goal":
        return None
    
    # Check for valid goal/subgoal/task ID
    if len(parts) >= 2:
        # Check if it's a valid ID
        if parts[1].startswith('G') and parts[1][1:].isdigit():
            return parts[1]
        if parts[1].startswith('S') and parts[1][1:].isdigit():
            return parts[1]
        if parts[1].startswith('T') and parts[1][1:].isdigit():
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


def go_to_child(child_id):
    """Go to a specific subgoal or task branch."""
    # Verify child exists
    goal_path = ensure_goal_dir()
    child_file = goal_path / f"{child_id}.json"
    
    if not child_file.exists():
        logging.error(f"Subgoal/task not found: {child_id}")
        return False
    
    # Find the branch for the child
    child_branch = find_branch_for_goal(child_id)
    if not child_branch:
        logging.error(f"No branch found for {child_id}")
        return False
    
    # Determine if it's a goal, subgoal, or task
    child_type = "goal"
    if child_id.startswith("S"):
        child_type = "subgoal"
    elif child_id.startswith("T"):
        child_type = "task"
    
    # Switch to the child branch
    try:
        subprocess.run(
            ["git", "checkout", child_branch],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Switched to {child_type}: {child_id}")
        print(f"Branch: {child_branch}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to switch to branch: {e}")
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
    """List subgoals and tasks for a specific goal ID or current branch."""
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
    
    # Find direct children (subgoals and tasks)
    # Use case-insensitive comparison
    children = {k: v for k, v in goal_files.items() 
               if v.get("parent_goal", "").upper() == goal_id.upper() or 
                  v.get("parent_goal", "").upper() == f"{goal_id.upper()}.json"}
    
    if not children:
        print(f"No subgoals or tasks found for goal {goal_id}")
        return []
    
    print(f"Children of {goal_id}:")
    
    # Display subgoals first
    subgoals = {k: v for k, v in children.items() if k.startswith("S")}
    if subgoals:
        print("\nSubgoals:")
        for subgoal_id in sorted(subgoals.keys()):
            subgoal = subgoals[subgoal_id]
            print(f"• {subgoal_id}: {subgoal['description']}")
    
    # Display tasks
    tasks = {k: v for k, v in children.items() if k.startswith("T") or v.get("is_task", False)}
    if tasks:
        print("\nTasks:")
        for task_id in sorted(tasks.keys()):
            task = tasks[task_id]
            print(f"• {task_id}: {task['description']}")
    
    return list(children.keys())


def mark_goal_complete(goal_id=None):
    """Mark a goal as complete."""
    # If no goal ID provided, use the current branch
    if not goal_id:
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        goal_id = get_goal_id_from_branch(current_branch)
        if not goal_id:
            logging.error(f"Current branch is not a goal branch: {current_branch}")
            return False
    
    # Verify goal exists
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal not found: {goal_id}")
        return False
    
    try:
        # Update goal file with completion status
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
        
        # Mark as complete with timestamp
        goal_data["complete"] = True
        goal_data["completion_time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save updated goal data
        with open(goal_file, 'w') as f:
            json.dump(goal_data, f, indent=2)
        
        print(f"Marked goal {goal_id} as complete")
        
        # Get current git hash for reference
        current_hash = get_current_hash()
        if current_hash:
            print(f"Current commit: {current_hash[:8]}")
        
        return True
    except Exception as e:
        logging.error(f"Failed to mark goal as complete: {e}")
        return False


def merge_subgoal(subgoal_id, testing=False):
    """Merge a specific subgoal into the current goal."""
    # Verify subgoal exists
    goal_path = ensure_goal_dir()
    subgoal_file = goal_path / f"{subgoal_id}.json"
    
    if not subgoal_file.exists():
        logging.error(f"Subgoal not found: {subgoal_id}")
        return False
    
    # Verify that the subgoal is marked as complete
    try:
        with open(subgoal_file, 'r') as f:
            subgoal_data = json.load(f)
        
        if not subgoal_data.get("complete", False):
            logging.warning(f"Subgoal {subgoal_id} is not marked as complete. Merging incomplete goals is not recommended.")
            response = input("Are you sure you want to continue? (y/N): ")
            if response.lower() != 'y':
                print("Merge cancelled.")
                return False
    except Exception as e:
        logging.error(f"Failed to read subgoal file: {e}")
        return False
    
    # Get the parent goal ID
    parent_id = subgoal_data.get("parent_goal", "")
    if not parent_id:
        logging.error(f"Subgoal {subgoal_id} doesn't have a parent goal")
        return False
    
    if parent_id.endswith('.json'):
        parent_id = parent_id[:-5]  # Remove .json extension
    
    # Get current branch and verify we're on the parent branch
    current_branch = get_current_branch()
    parent_branch = find_branch_for_goal(parent_id)
    
    if not parent_branch:
        logging.error(f"No branch found for parent goal {parent_id}")
        return False
    
    if current_branch != parent_branch and not testing:
        logging.warning(f"Current branch ({current_branch}) is not the parent goal branch ({parent_branch})")
        response = input("Would you like to switch to the parent branch first? (Y/n): ")
        if response.lower() != 'n':
            try:
                subprocess.run(
                    ["git", "checkout", parent_branch],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Switched to parent branch: {parent_branch}")
                current_branch = parent_branch
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to switch to parent branch: {e}")
                return False
    
    # Find the subgoal branch
    subgoal_branch = find_branch_for_goal(subgoal_id)
    if not subgoal_branch:
        logging.error(f"No branch found for subgoal {subgoal_id}")
        return False
    
    # Try to merge the subgoal branch into the current (parent) branch
    try:
        # First, try a merge with --no-commit to see if there are conflicts
        result = subprocess.run(
            ["git", "merge", "--no-commit", "--no-ff", subgoal_branch],
            capture_output=True,
            text=True
        )
        
        if "CONFLICT" in result.stderr or "CONFLICT" in result.stdout:
            print("Merge conflicts detected:")
            print(result.stderr)
            print(result.stdout)
            
            # Abort the merge
            subprocess.run(
                ["git", "merge", "--abort"],
                check=True,
                capture_output=True
            )
            
            print("Merge aborted due to conflicts.")
            print("Please resolve conflicts manually:")
            print(f"1. git checkout {parent_branch}")
            print(f"2. git merge {subgoal_branch}")
            print("3. Resolve conflicts")
            print("4. git add <resolved-files>")
            print("5. git commit -m 'Merge subgoal {subgoal_id}'")
            
            return False
        
        # If we got here, the merge can proceed without conflicts
        # Abort the --no-commit merge first
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            check=True,
            capture_output=True
        )
        
        # Now do the actual merge
        message = f"Merge subgoal {subgoal_id} into {parent_id}"
        subprocess.run(
            ["git", "merge", "--no-ff", "-m", message, subgoal_branch],
            check=True,
            capture_output=True
        )
        
        print(f"Successfully merged subgoal {subgoal_id} into {parent_id}")
        
        # Get the new commit hash
        new_hash = get_current_hash()
        if new_hash:
            print(f"Merge commit: {new_hash[:8]}")
        
        # Update parent goal with completion information
        parent_file = goal_path / f"{parent_id}.json"
        if parent_file.exists():
            try:
                with open(parent_file, 'r') as f:
                    parent_data = json.load(f)
                
                # Add merged subgoal to the list of merged subgoals
                if "merged_subgoals" not in parent_data:
                    parent_data["merged_subgoals"] = []
                
                merge_info = {
                    "subgoal_id": subgoal_id,
                    "merge_time": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "merge_commit": new_hash
                }
                
                parent_data["merged_subgoals"].append(merge_info)
                
                # Save updated parent data
                with open(parent_file, 'w') as f:
                    json.dump(parent_data, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to update parent goal metadata: {e}")
        
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to merge subgoal: {e}")
        return False


def show_goal_status():
    """Show completion status of all goals."""
    goal_path = ensure_goal_dir()
    if not goal_path.exists():
        print("No goals directory found")
        return
    
    # Get all goal files
    goal_files = {}
    for file in goal_path.glob("*.json"):
        try:
            with open(file, 'r') as f:
                goal_data = json.load(f)
                goal_files[goal_data["goal_id"]] = goal_data
        except Exception as e:
            logging.error(f"Failed to load goal file {file}: {e}")
    
    if not goal_files:
        print("No goals found")
        return
    
    def print_goal_status(goal_id, depth=0):
        if goal_id not in goal_files:
            return
        
        goal = goal_files[goal_id]
        indent = "  " * depth
        
        # Determine status symbol
        if goal.get("complete", False) or goal.get("completed", False):
            # Check if goal has been validated
            if "validation_status" in goal and goal["validation_status"].get("last_score", 0) >= goal.get("success_threshold", 0.8):
                status = "✅"  # Complete and validated
            else:
                status = "🔷"  # Complete but not validated
        else:
            # Special handling for tasks
            if goal.get("is_task", False):
                if "execution_result" in goal and goal["execution_result"].get("success"):
                    if "validation_status" in goal and goal["validation_status"].get("last_score", 0) >= goal.get("success_threshold", 0.8):
                        status = "✅"  # Task executed and validated
                    else:
                        status = "🔷"  # Task executed but not validated
                else:
                    status = "⏳"  # Task pending execution
            else:
                # Check for completed tasks
                completed_tasks = len(goal.get("completed_tasks", []))
                has_completed_tasks = completed_tasks > 0
                
                # Check if all subgoals are complete
                subgoals = {k: v for k, v in goal_files.items() 
                          if v.get("parent_goal", "").upper() == goal_id.upper()}
                all_subgoals_complete = all(sg.get("complete", False) or sg.get("completed", False) for sg in subgoals.values())
                
                if not subgoals:
                    status = "🔘"  # No subgoals (not yet decomposed)
                elif all_subgoals_complete:
                    if "validation_status" in goal and goal["validation_status"].get("last_score", 0) >= goal.get("success_threshold", 0.8):
                        status = "✅"  # All subgoals complete and validated
                    else:
                        status = "🔷"  # All subgoals complete but not validated
                else:
                    status = "⚪"  # Some subgoals incomplete
        
        # Get progress text
        progress_text = ""
        if "completed_task_count" in goal and "total_task_count" in goal:
            completed = goal["completed_task_count"]
            total = goal["total_task_count"]
            if total > 0:
                progress_text = f" ({completed}/{total})"
        
        # Get memory hash information from current_state
        memory_hash = goal.get("current_state", {}).get("memory_hash", "")
        memory_hash_display = f" [mem:{memory_hash[:8]}]" if memory_hash else ""
        
        # Print goal line with memory hash
        print(f"{indent}{status} {goal_id}{progress_text}{memory_hash_display}: {goal['description']}")
        
        # Print completion timestamp if available
        if (goal.get("complete", False) or goal.get("completed", False)) and \
           ("completion_time" in goal or "completion_timestamp" in goal):
            completion_time = goal.get("completion_time") or goal.get("completion_timestamp")
            print(f"{indent}   Completed: {completion_time}")
        
        # Print current state if available
        if "current_state" in goal and not goal.get("complete", False):
            current_state = goal["current_state"]
            last_updated = current_state.get("last_updated", "")
            last_task = current_state.get("last_task", "")
            if last_updated and last_task:
                print(f"{indent}   Last updated: {last_updated} by task {last_task}")
        
        # Print completed tasks if available
        if "completed_tasks" in goal and goal["completed_tasks"] and not goal.get("complete", False):
            print(f"{indent}   Completed tasks:")
            for task in goal["completed_tasks"]:
                task_id = task.get("task_id", "")
                timestamp = task.get("timestamp", "")
                task_memory_hash = task.get("final_state", {}).get("memory_hash", "")
                memory_hash_display = f" [mem:{task_memory_hash[:8]}]" if task_memory_hash else ""
                print(f"{indent}     - {task_id}{memory_hash_display} at {timestamp}")
        
        # Print merged subgoals if available
        if "merged_subgoals" in goal and goal["merged_subgoals"]:
            for merge_info in goal["merged_subgoals"]:
                subgoal_id = merge_info.get("subgoal_id", "")
                merge_time = merge_info.get("merge_time", "")
                print(f"{indent}   Merged: {subgoal_id} at {merge_time}")
        
        # Print validation status if available
        if "validation_status" in goal:
            validation = goal["validation_status"]
            last_validated = validation.get("last_validated", "")
            last_score = validation.get("last_score", 0.0)
            last_validated_by = validation.get("last_validated_by", "")
            last_git_hash = validation.get("last_git_hash", "")
            
            if last_validated:
                print(f"{indent}   Last validated: {last_validated} by {last_validated_by}")
                print(f"{indent}   Validation score: {last_score:.2%}")
                if last_git_hash:
                    print(f"{indent}   Git hash: {last_git_hash[:8]}")
        elif goal.get("complete", False) or goal.get("completed", False):
            print(f"{indent}   ⚠️  Not yet validated")
        
        # Find and print children
        children = {k: v for k, v in goal_files.items() 
                   if v.get("parent_goal", "").upper() == goal_id.upper()}
        
        for child_id in sorted(children.keys()):
            print_goal_status(child_id, depth + 1)
    
    # Find root goals (those without parents)
    root_goals = {k: v for k, v in goal_files.items() 
                  if not v.get("parent_goal")}
    
    # Print status for each root goal
    for root_id in sorted(root_goals.keys()):
        print_goal_status(root_id)


def show_goal_tree():
    """Show visual representation of goal hierarchy."""
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
    
    if not top_goals:
        print("No goals found.")
        return
    
    print("Goal Tree:")
    
    # Define a function to print goal tree recursively with better formatting
    def print_goal_tree(goal_id, prefix="", is_last=True, depth=0):
        if goal_id not in goal_files:
            return
            
        goal = goal_files[goal_id]
        
        # Check for state equality with parent
        states_equal = True
        parent_id = goal.get("parent_goal", "")
        if parent_id in goal_files:
            parent = goal_files[parent_id]
            
            # Check if this goal or task has an initial_state
            if "initial_state" in goal and "current_state" in parent:
                # Compare initial git hash with parent's current git hash
                goal_initial_hash = goal.get("initial_state", {}).get("git_hash", "")
                parent_current_hash = parent.get("current_state", {}).get("git_hash", "")
                
                # Compare initial memory hash with parent's current memory hash
                goal_initial_memory_hash = goal.get("initial_state", {}).get("memory_hash", "")
                parent_current_memory_hash = parent.get("current_state", {}).get("memory_hash", "")
                
                # Check git hash equality
                git_hash_equal = not (goal_initial_hash and parent_current_hash and goal_initial_hash != parent_current_hash)
                
                # Check memory hash equality
                memory_hash_equal = not (goal_initial_memory_hash and parent_current_memory_hash 
                                        and goal_initial_memory_hash != parent_current_memory_hash)
                
                # States are equal only if both git hash and memory hash are equal
                states_equal = git_hash_equal and memory_hash_equal
        
        # Determine status symbol
        if goal.get("complete", False) or goal.get("completed", False):
            status = "✅"
        else:
            # Special handling for tasks
            if goal.get("is_task", False):
                if "execution_result" in goal and goal["execution_result"].get("success"):
                    status = "✅"  # Completed task
                else:
                    if not states_equal:
                        status = "🔺"  # Branch task
                    else:
                        status = "🔷"  # Directly executable task
            else:
                # Check if all subgoals are complete
                # Use case-insensitive comparison
                subgoals = {k: v for k, v in goal_files.items() 
                           if v.get("parent_goal", "").upper() == goal_id.upper() or 
                              v.get("parent_goal", "").upper() == f"{goal_id.upper()}.json"}
                
                if not subgoals:
                    # Check if goal is a task or directly executable
                    if goal.get("is_task", False) or goal.get("requires_further_decomposition") is False:
                        if not states_equal:
                            status = "🔺"  # Branch task
                        else:
                            status = "🔷"  # Directly executable task
                    else:
                        status = "🔘"  # No subgoals (not yet decomposed)
                elif all(sg.get("complete", False) or sg.get("completed", False) for sg in subgoals.values()):
                    status = "⚪"  # All subgoals complete but needs explicit completion
                else:
                    if not states_equal:
                        status = "🔸"  # Branch subgoal
                    else:
                        status = "⚪"  # Some subgoals incomplete
        
        # Determine branch characters
        branch = "└── " if is_last else "├── "
        
        # Print the current goal
        print(f"{prefix}{branch}{status} {goal_id}: {goal['description']}")
        
        # Find children
        # Use case-insensitive comparison
        children = {k: v for k, v in goal_files.items() 
                   if v.get("parent_goal", "").upper() == goal_id.upper() or 
                      v.get("parent_goal", "").upper() == f"{goal_id.upper()}.json"}
        
        # Sort children by ID
        sorted_children = sorted(children.keys())
        
        # Determine new prefix for children
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        # Print children
        for i, child_id in enumerate(sorted_children):
            is_last_child = (i == len(sorted_children) - 1)
            print_goal_tree(child_id, new_prefix, is_last_child, depth + 1)
    
    # Print all top-level goals and their subgoals
    sorted_top_goals = sorted(top_goals.keys())
    for i, goal_id in enumerate(sorted_top_goals):
        is_last = (i == len(sorted_top_goals) - 1)
        print_goal_tree(goal_id, "", is_last)
    
    # Print legend
    print("\nStatus Legend:")
    print("✅ Complete")
    print("🟡 Partially completed")
    print("⚪ Incomplete")
    print("🔷 Directly executable task")
    print("🔘 Not yet decomposed")
    print("🔺 Branch task")
    print("🔸 Branch subgoal")


def show_goal_history():
    """Show timeline of goal exploration."""
    goal_path = ensure_goal_dir()
    
    # Get all goal files and sort by timestamp
    goals = []
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goals.append(data)
        except:
            logging.warning(f"Failed to read goal file: {file_path}")
    
    # Sort goals by timestamp
    goals.sort(key=lambda x: x.get("timestamp", ""))
    
    if not goals:
        print("No goals found.")
        return
    
    print("Goal History:")
    print("=============")
    
    # Print goals in chronological order
    for goal in goals:
        goal_id = goal.get("goal_id", "Unknown")
        desc = goal.get("description", "No description")
        timestamp = goal.get("timestamp", "Unknown time")
        parent = goal.get("parent_goal", "")
        
        # Format timestamp
        if len(timestamp) >= 8:
            try:
                timestamp = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
            except:
                pass  # Keep original format if parsing fails
        
        # Print basic goal info
        print(f"[{timestamp}] {goal_id}: {desc}")
        
        # Print parent info if available
        if parent:
            print(f"  └── Subgoal of: {parent}")
        
        # Print completion info if available
        if goal.get("complete", False):
            completion_time = goal.get("completion_time", "Unknown time")
            if len(completion_time) >= 8:
                try:
                    completion_time = f"{completion_time[:4]}-{completion_time[4:6]}-{completion_time[6:8]} {completion_time[9:11]}:{completion_time[11:13]}:{completion_time[13:15]}"
                except:
                    pass  # Keep original format if parsing fails
            print(f"  └── Completed: {completion_time}")
        
        # Print merged subgoals if available
        if "merged_subgoals" in goal and goal["merged_subgoals"]:
            for merge_info in goal["merged_subgoals"]:
                subgoal_id = merge_info.get("subgoal_id", "")
                merge_time = merge_info.get("merge_time", "")
                if len(merge_time) >= 8:
                    try:
                        merge_time = f"{merge_time[:4]}-{merge_time[4:6]}-{merge_time[6:8]} {merge_time[9:11]}:{merge_time[11:13]}:{merge_time[13:15]}"
                    except:
                        pass  # Keep original format if parsing fails
                print(f"  └── Merged: {subgoal_id} at {merge_time}")
        
        print("")  # Empty line between goals
    
    # Print summary
    total_goals = len(goals)
    complete_goals = sum(1 for g in goals if g.get("complete", False))
    print(f"Summary: {complete_goals}/{total_goals} goals completed")


def generate_graph():
    """Generate a graphical visualization of the goal hierarchy."""
    try:
        # Check if graphviz is installed
        subprocess.run(["dot", "-V"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Graphviz is not installed. Please install Graphviz to use this feature.")
        print("Installation instructions: https://graphviz.org/download/")
        return False
    
    goal_path = ensure_goal_dir()
    visualization_path = ensure_visualization_dir()
    
    # Get all goal files
    goal_files = {}
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goal_files[data["goal_id"]] = data
        except:
            logging.warning(f"Failed to read goal file: {file_path}")
    
    if not goal_files:
        print("No goals found.")
        return False
    
    # Create DOT file content
    dot_content = [
        "digraph Goals {",
        "  rankdir=TB;",
        "  node [shape=box, style=filled, fontname=Arial];",
        "  edge [fontname=Arial];",
        ""
    ]
    
    # Add nodes (goals)
    for goal_id, goal_data in goal_files.items():
        # Determine node color based on status
        if goal_data.get("complete", False):
            color = "green"
        else:
            # Check if all subgoals are complete
            subgoals = {k: v for k, v in goal_files.items() 
                      if v.get("parent_goal") == goal_id or 
                         v.get("parent_goal") == f"{goal_id}.json"}
            
            if not subgoals:
                if goal_data.get("is_task", False) or goal_data.get("requires_further_decomposition") is False:
                    color = "yellow"  # Task/directly executable
                else:
                    color = "gray"  # No subgoals (not yet decomposed)
            elif all(sg.get("complete", False) for sg in subgoals.values()):
                color = "orange"  # All subgoals complete but not merged
            else:
                color = "lightblue"  # Some subgoals incomplete
        
        # Escape special characters
        desc = goal_data.get("description", "").replace('"', '\\"')
        
        # Determine node shape based on type
        shape = "box"
        if goal_id.startswith("T"):
            shape = "ellipse"  # Tasks are ellipses
        elif goal_id.startswith("S"):
            shape = "box"  # Subgoals are boxes
        elif goal_id.startswith("G"):
            shape = "box"  # Goals are boxes (can make them rounded if preferred)
        
        # Add node
        dot_content.append(f'  "{goal_id}" [label="{goal_id}\\n{desc}", fillcolor={color}, shape={shape}];')
    
    # Add edges (parent-child relationships)
    for goal_id, goal_data in goal_files.items():
        parent_id = goal_data.get("parent_goal", "")
        if parent_id:
            if parent_id.endswith('.json'):
                parent_id = parent_id[:-5]  # Remove .json extension
            dot_content.append(f'  "{parent_id}" -> "{goal_id}";')
    
    # Close the graph
    dot_content.append("}")
    
    # Write DOT file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dot_file = visualization_path / f"goals_{timestamp}.dot"
    pdf_file = visualization_path / f"goals_{timestamp}.pdf"
    png_file = visualization_path / f"goals_{timestamp}.png"
    
    with open(dot_file, 'w') as f:
        f.write("\n".join(dot_content))
    
    # Generate PDF
    try:
        subprocess.run(
            ["dot", "-Tpdf", str(dot_file), "-o", str(pdf_file)],
            check=True,
            capture_output=True
        )
        print(f"PDF graph generated: {pdf_file}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate PDF: {e}")
    
    # Generate PNG
    try:
        subprocess.run(
            ["dot", "-Tpng", str(dot_file), "-o", str(png_file)],
            check=True,
            capture_output=True
        )
        print(f"PNG graph generated: {png_file}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate PNG: {e}")
    
    return True


def decompose_existing_goal(goal_id, debug=False, quiet=False, bypass_validation=False):
    """Decompose an existing goal into subgoals."""
    # Get the goal file path
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal {goal_id} not found")
        return False
    
    # Load the goal data
    try:
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read goal file: {e}")
        return False
    
    # Find the top-level goal's branch
    top_level_branch = find_top_level_branch(goal_id)
    if not top_level_branch:
        logging.error(f"Failed to find top-level goal branch for {goal_id}")
        return False
    
    # Save current branch and check for changes
    current_branch = get_current_branch()
    if not current_branch:
        logging.error("Failed to get current branch")
        return False
    
    has_changes = False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True
        )
        has_changes = bool(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to check git status: {e}")
        return False
    
    # Stash changes if needed
    if has_changes:
        try:
            subprocess.run(
                ["git", "stash", "push", "-m", f"Stashing changes before decomposing goal {goal_id}"],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to stash changes: {e}")
            return False
    
    try:
        # Switch to the top-level goal's branch
        try:
            subprocess.run(
                ["git", "checkout", top_level_branch],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to checkout branch {top_level_branch}: {e}")
            return False
        
        # Get completed tasks and add context to goal description
        completed_tasks = goal_data.get("completed_tasks", [])
        completed_ids = [task.get("task_id") for task in completed_tasks]
        
        if completed_ids:
            logging.info(f"Found {len(completed_ids)} completed tasks: {', '.join(completed_ids)}")
            # Add context about completed tasks to the goal description
            goal_description = goal_data["description"]
            if "Context:" not in goal_description:
                goal_description += f"\n\nContext: The following tasks have already been completed: {', '.join(completed_ids)}."
                goal_data["description"] = goal_description
        
        # Get task counts
        completed_count = len(completed_tasks)
        total_count = goal_data.get("total_task_count", 0)
        
        if completed_count > 0 and total_count > 0:
            logging.info(f"Current progress: {completed_count}/{total_count} tasks completed")
        
        # Get memory state from current_state
        memory_hash = None
        memory_repo_path = goal_data["current_state"].get("memory_repository_path")
        if memory_repo_path:
            memory_hash = goal_data["current_state"].get("memory_hash")
            if memory_hash:
                logging.info(f"Memory hash from context state: {memory_hash}")
            else:
                logging.warning("No memory hash found in context state")
            logging.info(f"Memory repository path from context state: {memory_repo_path}")
        else:
            logging.warning("No memory repository path found in context state")
        
        # Call the goal decomposer with the goal file as input
        result = agent_decompose_goal(
            repo_path=os.getcwd(),
            goal=goal_data["description"],
            validation_criteria=goal_data.get("validation_criteria", []),
            parent_goal_id=goal_data.get("parent_goal", None),
            goal_id=goal_id,
            memory_hash=memory_hash,
            memory_repo_path=memory_repo_path,
            debug=debug,
            quiet=quiet,
            bypass_validation=bypass_validation,
            logs_dir="logs"
        )
        
        if result["success"]:
            if result.get("goal_completed", False):
                print(f"\nGoal {goal_id} has been completed!")
            else:
                print(f"\nGoal {goal_id} successfully decomposed into a subgoal")
            
            if result.get("goal_completed", False):
                print("\n✅ Goal completed!")
                print(f"Summary: {result['completion_summary']}")
                print(f"Reasoning: {result['reasoning']}")
            else:
                print(f"\nNext step: {result['next_step']}")
                print("\nValidation criteria:")
                for criterion in result["validation_criteria"]:
                    print(f"- {criterion}")
                
                if result["can_be_decomposed"]:
                    print("\nRequires further decomposition: Yes")
                else:
                    print("\nRequires further decomposition: No")
                
                print(f"\nGoal file: {result['goal_file']}")
            
            # Get updated memory hash after decomposition
            updated_memory_hash = memory_hash
            if memory_repo_path:
                try:
                    # Import the function to get the current hash
                    from .agents.tools.git_tools import get_current_hash
                    # Get the current hash in the memory repo
                    updated_memory_hash = get_current_hash(memory_repo_path)
                    if updated_memory_hash != memory_hash:
                        logging.info(f"Memory hash updated during decomposition: {updated_memory_hash[:8]}")
                except Exception as e:
                    logging.warning(f"Failed to get updated memory hash: {e}")
            
            # Update the goal file with the decomposition result
            goal_data.update({
                "goal_completed": result.get("goal_completed", False),
                "completion_summary": result.get("completion_summary"),
                "next_step": result.get("next_step"),
                "validation_criteria": result.get("validation_criteria", []),
                "reasoning": result["reasoning"],
                "can_be_decomposed": result.get("can_be_decomposed", result.get("requires_further_decomposition", True)),
                "relevant_context": result.get("relevant_context", {}),
                "decomposed": True,  # Mark the goal as decomposed
                # Don't update current_state automatically during decomposition
                # The line that updated "current_state" has been removed
            })
            
            # If goal is completed, mark it as completed in the goal file
            if result.get("goal_completed", False):
                goal_data.update({
                    "completed": True,
                    "completion_timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "completion_summary": result["completion_summary"],
                    "completion_reasoning": result["reasoning"]
                })
            
            # Save the updated goal data
            with open(goal_file, 'w') as f:
                json.dump(goal_data, f, indent=2)
            
            # Only create subgoal if goal is not completed
            if not result.get("goal_completed", False):
                # Create the subgoal file
                subgoal_id = generate_goal_id(goal_id, is_task=not result["can_be_decomposed"])
                subgoal_file = goal_path / f"{subgoal_id}.json"
                
                # Create subgoal data with updated memory state
                subgoal_data = {
                    "goal_id": subgoal_id,
                    "description": result["next_step"],
                    "parent_goal": goal_id,
                    "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "is_task": not result["can_be_decomposed"],
                    "can_be_decomposed": result["can_be_decomposed"],
                    "validation_criteria": result["validation_criteria"],
                    "reasoning": result["reasoning"],
                    "relevant_context": result.get("relevant_context", {}),
                    "initial_state": {
                        "git_hash": result["git_hash"],
                        "repository_path": os.getcwd(),
                        "description": "Initial state before executing subgoal",
                        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "memory_hash": updated_memory_hash,
                        "memory_repository_path": memory_repo_path
                    },
                    "current_state": {
                        "git_hash": result["git_hash"],
                        "repository_path": os.getcwd(),
                        "description": "Initial state before executing subgoal",
                        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "memory_hash": updated_memory_hash,
                        "memory_repository_path": memory_repo_path
                    }
                }
                
                # Save the subgoal file
                with open(subgoal_file, 'w') as f:
                    json.dump(subgoal_data, f, indent=2)
                
                print(f"Created {'task' if not result['can_be_decomposed'] else 'subgoal'} file: {subgoal_file}")
            
            return True
        else:
            error_msg = result.get('error', 'Unknown error')
            logging.error(f"Failed to decompose goal: {error_msg}")
            print(f"Failed to decompose goal: {error_msg}", file=sys.stderr)
            return False
            
    finally:
        # Always restore the original branch and unstash changes
        try:
            # Switch back to original branch
            subprocess.run(
                ["git", "checkout", current_branch],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Unstash changes if we stashed them
            if has_changes:
                subprocess.run(
                    ["git", "stash", "pop"],
                    check=True,
                    capture_output=True,
                    text=True
                )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to restore original state: {e}")
            # Don't raise here as we're in a finally block
    
    return False


def execute_task(task_id, debug=False, quiet=False, bypass_validation=False, no_commit=False, memory_repo=None):
    """Execute a task using the TaskExecutor."""
    # Get the task file path
    goal_path = ensure_goal_dir()
    task_file = goal_path / f"{task_id}.json"
    
    if not task_file.exists():
        logging.error(f"Task {task_id} not found")
        return False
    
    # Load the task data
    try:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read task file: {e}")
        return False
    
    # Find the top-level goal's branch
    top_level_branch = find_top_level_branch(task_id)
    if not top_level_branch:
        logging.error(f"Failed to find top-level goal branch for {task_id}")
        return False
    
    # Save current branch and check for changes
    current_branch = get_current_branch()
    if not current_branch:
        logging.error("Failed to get current branch")
        return False
    
    has_changes = False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True
        )
        has_changes = bool(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to check git status: {e}")
        return False
    
    # Stash changes if needed
    if has_changes:
        try:
            subprocess.run(
                ["git", "stash", "push", "-m", f"Stashing changes before executing task {task_id}"],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to stash changes: {e}")
            return False
    
    try:
        # Switch to the top-level goal's branch
        try:
            subprocess.run(
                ["git", "checkout", top_level_branch],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to checkout branch {top_level_branch}: {e}")
            return False
        
        # Get memory state from task data
        memory_state = None
        if "initial_state" in task_data:
            initial_state = task_data["initial_state"]
            if "memory_hash" in initial_state and "memory_repository_path" in initial_state:
                memory_state = MemoryState(
                    memory_hash=initial_state["memory_hash"],
                    repository_path=initial_state["memory_repository_path"]
                )
                logging.info(f"Using memory state from task file - hash: {initial_state['memory_hash'][:8]}")
        
        # Create task context
        context = TaskContext(
            state=State(
                git_hash=task_data["initial_state"]["git_hash"],
                repository_path=task_data["initial_state"]["repository_path"],
                description=task_data["initial_state"]["description"],
                branch_name=top_level_branch,
                memory_hash=task_data["initial_state"].get("memory_hash"),
                memory_repository_path=task_data["initial_state"].get("memory_repository_path")
            ),
            goal=Goal(
                description=task_data["description"],
                validation_criteria=[]
            ),
            iteration=0,
            execution_history=[],
            memory_state=memory_state
        )
        
        # Configure logging
        configure_executor_logging(debug, quiet)
        
        # Create and run the executor
        executor = TaskExecutor()
        result = executor.execute_task(context, task_data["description"])
        
        if result.success:
            # Get current git hash
            current_hash = get_current_hash()
            if not current_hash:
                logging.error("Failed to get current git hash")
                return False
            
            # Get current memory hash if memory repository is available
            memory_hash = None
            memory_repo_path = task_data["initial_state"].get("memory_repository_path")
            if memory_repo_path:
                try:
                    memory_hash = get_current_hash(memory_repo_path)
                    logging.info(f"Updated memory hash: {memory_hash[:8]}")
                except Exception as e:
                    logging.warning(f"Failed to get updated memory hash: {e}")
            
            # Update task's current state
            task_data["current_state"] = {
                "git_hash": current_hash,
                "repository_path": result.repository_path,
                "description": f"State after executing task: {task_id}",
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "memory_hash": memory_hash or task_data["initial_state"].get("memory_hash"),
                "memory_repository_path": memory_repo_path
            }
            
            # Mark task as completed
            task_data["complete"] = True
            task_data["completion_time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save updated task data
            with open(task_file, 'w') as f:
                json.dump(task_data, f, indent=2)
            
            # Remove automatic parent state update
            # parent_goal_id = task_data.get("parent_goal")
            # if parent_goal_id:
            #     update_parent_goal_state(
            #         parent_goal_id=parent_goal_id,
            #         task_id=task_id,
            #         execution_result=result,
            #         final_state=task_data["current_state"]
            #     )
            
            print(f"\nTask {task_id} executed successfully")
            if result.validation_results:
                print(f"Result: {result.validation_results}")
            return True
        else:
            print(f"Failed to execute task: {result.error_message}", file=sys.stderr)
            return False
            
    finally:
        # Always restore the original branch and unstash changes
        try:
            # Switch back to original branch
            subprocess.run(
                ["git", "checkout", current_branch],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Unstash changes if we stashed them
            if has_changes:
                subprocess.run(
                    ["git", "stash", "pop"],
                    check=True,
                    capture_output=True,
                    text=True
                )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to restore original state: {e}")
            # Don't raise here as we're in a finally block
    
    return False


def update_parent_goal_state(parent_goal_id, task_id, execution_result, final_state):
    """
    Update the parent goal's state based on task execution results.
    
    Args:
        parent_goal_id: ID of the parent goal
        task_id: ID of the completed task
        execution_result: The execution result object
        final_state: The final state after execution
        
    Returns:
        Boolean indicating whether the parent was successfully updated
    """
    goal_path = ensure_goal_dir()
    parent_file = goal_path / f"{parent_goal_id}.json"
    
    if not parent_file.exists():
        logging.error(f"Parent goal file not found: {parent_file}")
        return False
    
    try:
        # Load the parent goal data
        with open(parent_file, 'r') as f:
            parent_data = json.load(f)
        
        # Initialize initial_state if not present
        if "initial_state" not in parent_data:
            # First task completion will set the initial state of the parent goal
            parent_data["initial_state"] = {
                "git_hash": final_state["git_hash"],
                "repository_path": final_state["repository_path"],
                "description": f"Initial state recorded when first task {task_id} was completed",
                "memory_hash": final_state.get("memory_hash"),
                "memory_repository_path": final_state.get("memory_repository_path"),
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        
        # If not tracking completed tasks yet, initialize the list
        if "completed_tasks" not in parent_data:
            parent_data["completed_tasks"] = []
        
        # Load task data to get description and validation criteria
        task_file = goal_path / f"{task_id}.json"
        task_description = f"Task {task_id}"
        task_validation_criteria = []
        
        if task_file.exists():
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                    task_description = task_data.get("description", task_description)
                    task_validation_criteria = task_data.get("validation_criteria", [])
            except Exception as e:
                logging.warning(f"Failed to read task file: {e}")
        
        # Add this task to the list of completed tasks if not already there
        task_already_recorded = False
        for completed in parent_data["completed_tasks"]:
            if completed.get("task_id") == task_id:
                task_already_recorded = True
                completed.update({
                    "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "git_hash": execution_result.git_hash,
                    "final_state": final_state,
                    "description": task_description,
                    "validation_criteria": task_validation_criteria
                })
                break
                
        if not task_already_recorded:
            parent_data["completed_tasks"].append({
                "task_id": task_id,
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "git_hash": execution_result.git_hash,
                "final_state": final_state,
                "description": task_description,
                "validation_criteria": task_validation_criteria
            })
        
        # Update the parent's current state to reflect the latest git hash
        if "current_state" not in parent_data:
            parent_data["current_state"] = {}
            
        parent_data["current_state"] = final_state
        parent_data["current_state"]["last_updated"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_data["current_state"]["last_task"] = task_id
        
        # Calculate completion progress if there are child tasks
        children = get_child_tasks(parent_goal_id)
        if children:
            completed = sum(1 for child in children if 
                           any(ct.get("task_id") == child["goal_id"] for ct in parent_data["completed_tasks"]))
            parent_data["completed_task_count"] = completed
            parent_data["total_task_count"] = len(children)
        
        # Save the updated parent data
        with open(parent_file, 'w') as f:
            json.dump(parent_data, f, indent=2)
        
        return True
    except Exception as e:
        logging.error(f"Failed to update parent goal: {str(e)}")
        return False


def get_child_tasks(parent_id):
    """
    Get all child tasks for a given parent goal.
    
    Args:
        parent_id: ID of the parent goal
        
    Returns:
        List of child task objects
    """
    goal_path = ensure_goal_dir()
    child_tasks = []
    
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Check if this is a child of the parent and is a task
            if data.get("parent_goal") == parent_id and data.get("is_task", False):
                child_tasks.append(data)
        except:
            continue
    
    return child_tasks


def main_command(args):
    """Async entry point for CLI commands."""
    # Handle async commands
    if args.command == "decompose":
        return decompose_existing_goal(args.goal_id, args.debug, args.quiet, args.bypass_validation)
    elif args.command == "execute":
        return execute_task(args.task_id, args.debug, args.quiet, args.bypass_validation, args.no_commit, args.memory_repo)
    
    # All other commands are synchronous, so just call them directly
    if args.command == "new":
        return create_new_goal(args.description)
    elif args.command == "sub":
        return create_new_subgoal(args.parent_id, args.description)
    elif args.command == "task":
        return create_new_task(args.parent_id, args.description)
    elif args.command == "list":
        return list_goals()
    elif args.command == "back":
        return go_back_commits(args.steps)
    elif args.command == "reset":
        return reset_to_commit(args.commit_id)
    elif args.command == "up":
        return go_to_parent_goal()
    elif args.command == "down":
        return go_to_child(args.subgoal_id)
    elif args.command == "root":
        return go_to_root_goal()
    elif args.command == "subs":
        return list_subgoals()
    elif args.command == "complete":
        return mark_goal_complete()
    elif args.command == "merge":
        return merge_subgoal(args.subgoal_id)
    elif args.command == "status":
        return show_goal_status()
    elif args.command == "tree":
        return show_goal_tree()
    elif args.command == "history":
        return show_goal_history()
    elif args.command == "graph":
        return generate_graph()
    elif args.command == "update-parent":
        return update_parent_from_child(args.child_id)
    elif args.command == "validate-history":
        return show_validation_history(args.goal_id, args.debug, args.quiet)
    else:
        return None


def main():
    """Main CLI entry point."""
    import asyncio
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Goal branch management commands")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Goal Management Commands
    # -----------------------
    # goal new <description>
    new_parser = subparsers.add_parser("new", help="Create a new top-level goal")
    new_parser.add_argument("description", help="Description of the goal")
    
    # goal delete <goal-id>
    delete_parser = subparsers.add_parser("delete", help="Delete a goal, subgoal, or task")
    delete_parser.add_argument("goal_id", help="ID of the goal to delete")
    
    # goal sub <parent-id> <description>
    sub_parser = subparsers.add_parser("sub", help="Create a subgoal under the specified parent")
    sub_parser.add_argument("parent_id", help="Parent goal ID")
    sub_parser.add_argument("description", help="Description of the subgoal")
    
    # goal task <parent-id> <description>
    task_parser = subparsers.add_parser("task", help="Create a new directly executable task under the specified parent")
    task_parser.add_argument("parent_id", help="Parent goal ID")
    task_parser.add_argument("description", help="Description of the task")
    
    # goal list
    subparsers.add_parser("list", help="List all goals and subgoals in tree format")
    
    # goal decompose <goal-id>
    decompose_parser = subparsers.add_parser("decompose", help="Decompose a goal into subgoals")
    decompose_parser.add_argument("goal_id", help="Goal ID to decompose")
    decompose_parser.add_argument("--debug", action="store_true", help="Show debug output")
    decompose_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    decompose_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    
    # goal execute <task-id>
    execute_parser = subparsers.add_parser("execute", help="Execute a task using the TaskExecutor")
    execute_parser.add_argument("task_id", help="Task ID to execute")
    execute_parser.add_argument("--debug", action="store_true", help="Show debug output")
    execute_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    execute_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    execute_parser.add_argument("--no-commit", action="store_true", help="Prevent automatic commits")
    execute_parser.add_argument("--memory-repo", help="Path to memory repository")
    
    # State Navigation Commands
    # ------------------------
    # goal back [steps]
    back_parser = subparsers.add_parser("back", help="Go back N commits on current goal branch")
    back_parser.add_argument("steps", nargs="?", type=int, default=1, help="Number of commits to go back")
    
    # goal reset <commit-id>
    reset_parser = subparsers.add_parser("reset", help="Reset to specific commit on current branch")
    reset_parser.add_argument("commit_id", help="Commit ID to reset to")
    
    # Hierarchy Navigation Commands
    # ---------------------------
    # goal up
    subparsers.add_parser("up", help="Go to parent goal branch")
    
    # goal down <subgoal-id>
    down_parser = subparsers.add_parser("down", help="Go to specific subgoal or task branch")
    down_parser.add_argument("subgoal_id", help="Subgoal or task ID to navigate to")
    
    # goal root
    subparsers.add_parser("root", help="Go to top-level goal")
    
    # goal subs
    subparsers.add_parser("subs", help="List available subgoals for current goal")
    
    # Result Incorporation Commands
    # ----------------------------
    # goal complete
    subparsers.add_parser("complete", help="Mark current goal as complete")
    
    # goal merge <subgoal-id>
    merge_parser = subparsers.add_parser("merge", help="Merge specific subgoal into current goal")
    merge_parser.add_argument("subgoal_id", help="Subgoal ID to merge")
    
    # goal status
    subparsers.add_parser("status", help="Show completion status of all goals")
    
    # Visualization Tools
    # ------------------
    # goal tree
    subparsers.add_parser("tree", help="Show visual representation of goal hierarchy")
    
    # goal history
    subparsers.add_parser("history", help="Show timeline of goal exploration")
    
    # goal graph
    subparsers.add_parser("graph", help="Generate graphical visualization")
    
    # goal revert <goal-id>
    revert_parser = subparsers.add_parser("revert", help="Revert a goal's current state back to its initial state")
    revert_parser.add_argument("goal_id", help="ID of the goal to revert")
    
    # goal validate <goal-id>
    validate_parser = subparsers.add_parser("validate", help="Validate a goal's completion criteria")
    validate_parser.add_argument("goal_id", help="ID of the goal to validate")
    validate_parser.add_argument("--debug", action="store_true", help="Show debug output")
    validate_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    validate_parser.add_argument("--auto", action="store_true", help="Perform automated validation using LLM")
    validate_parser.add_argument("--model", default="gpt-4o-mini", help="Model to use for validation (with --auto)")
    
    # goal validate-history <goal-id>
    validate_history_parser = subparsers.add_parser("validate-history", help="Show validation history for a goal")
    validate_history_parser.add_argument("goal_id", help="ID of the goal to show validation history for")
    validate_history_parser.add_argument("--debug", action="store_true", help="Show debug output")
    validate_history_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    
    # Add new subparser for updating parent from child
    update_parent_parser = subparsers.add_parser('update-parent', help='Update parent goal state from child goal/task')
    update_parent_parser.add_argument('child_id', help='ID of the child goal/task to use for updating parent')
    
    args = parser.parse_args()
    
    # Run the main function
    main_command(args)


def delete_goal(goal_id):
    """Delete a goal, subgoal, or task and its associated branch."""
    # Verify goal exists
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal {goal_id} not found")
        return False
    
    try:
        # Load goal data
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read goal file: {e}")
        return False
    
    # Find all child goals and tasks
    child_goals = []
    child_tasks = []
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data.get("parent_goal", "") == goal_id:
                    if data.get("is_task", False):
                        child_tasks.append(data["goal_id"])
                    else:
                        child_goals.append(data["goal_id"])
        except Exception as e:
            logging.warning(f"Failed to read goal file: {e}")
            continue
    
    # If there are children, ask for confirmation
    if child_goals or child_tasks:
        print(f"\nWarning: Goal {goal_id} has the following children:")
        if child_goals:
            print("\nSubgoals:")
            for child_id in child_goals:
                print(f"  • {child_id}")
        if child_tasks:
            print("\nTasks:")
            for child_id in child_tasks:
                print(f"  • {child_id}")
        
        response = input("\nDeleting this goal will also delete all its children. Are you sure? (y/N): ")
        if response.lower() != 'y':
            print("Deletion cancelled.")
            return False
    
    # Get the branch name
    branch_name = goal_data.get("branch_name")
    
    try:
        # Delete the goal file
        goal_file.unlink()
        print(f"Deleted goal file: {goal_file}")
        
        # If there's an associated branch, delete it
        if branch_name:
            try:
                # Switch to a different branch if we're on the one we're deleting
                current_branch = get_current_branch()
                if current_branch == branch_name:
                    # Switch to main branch or the first available branch
                    subprocess.run(
                        ["git", "checkout", "main"],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                
                # Delete the branch
                subprocess.run(
                    ["git", "branch", "-D", branch_name],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Deleted branch: {branch_name}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to delete branch {branch_name}: {e}")
                # Continue with deletion even if branch deletion fails
        
        # Recursively delete all child goals and tasks
        for child_id in child_goals + child_tasks:
            delete_goal(child_id)
        
        print(f"Successfully deleted goal {goal_id}")
        return True
    except Exception as e:
        logging.error(f"Failed to delete goal {goal_id}: {e}")
        return False


def find_top_level_branch(goal_id):
    """Find the branch of the top-level goal by traversing up the goal hierarchy."""
    goal_path = ensure_goal_dir()
    
    # Keep track of visited goals to prevent cycles
    visited = set()
    
    while goal_id and goal_id not in visited:
        visited.add(goal_id)
        
        # Check if this is a top-level goal (starts with G)
        if goal_id.startswith('G'):
            goal_file = goal_path / f"{goal_id}.json"
            if goal_file.exists():
                try:
                    with open(goal_file, 'r') as f:
                        goal_data = json.load(f)
                        branch_name = goal_data.get('branch_name')
                        if branch_name:
                            return branch_name
                        # If no branch name, generate one
                        branch_name = f"goal-{goal_id}"
                        goal_data['branch_name'] = branch_name
                        with open(goal_file, 'w') as f:
                            json.dump(goal_data, f, indent=2)
                        return branch_name
                except Exception as e:
                    logging.error(f"Failed to read goal file: {e}")
                    return None
        
        # Get parent goal ID
        goal_file = goal_path / f"{goal_id}.json"
        if not goal_file.exists():
            return None
            
        try:
            with open(goal_file, 'r') as f:
                goal_data = json.load(f)
                goal_id = goal_data.get("parent_goal", "")
                if goal_id.endswith('.json'):
                    goal_id = goal_id[:-5]  # Remove .json extension
        except Exception as e:
            logging.error(f"Failed to read goal file: {e}")
            return None
    
    return None


def revert_goal(goal_id):
    """Revert a goal's current state back to its initial state."""
    # Verify goal exists
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal {goal_id} not found")
        return False
    
    try:
        # Load goal data
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read goal file: {e}")
        return False
    
    # Check if goal has initial_state
    if "initial_state" not in goal_data:
        logging.error(f"Goal {goal_id} has no initial state")
        return False
    
    # Find all child goals and tasks
    child_goals = []
    child_tasks = []
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data.get("parent_goal", "") == goal_id:
                    if data.get("is_task", False):
                        child_tasks.append(data["goal_id"])
                    else:
                        child_goals.append(data["goal_id"])
        except:
            continue
    
    # If there are children, ask for confirmation
    if child_goals or child_tasks:
        print(f"\nWarning: Goal {goal_id} has the following children:")
        if child_goals:
            print("\nSubgoals:")
            for child_id in child_goals:
                print(f"  • {child_id}")
        if child_tasks:
            print("\nTasks:")
            for child_id in child_tasks:
                print(f"  • {child_id}")
        
        response = input("\nReverting this goal will affect all its children. Are you sure? (y/N): ")
        if response.lower() != 'y':
            print("Revert cancelled.")
            return False
    
    try:
        # Clean up the description by removing the context about completed tasks
        description = goal_data["description"]
        if "\n\nContext: The following tasks have already been completed:" in description:
            description = description.split("\n\nContext: The following tasks have already been completed:")[0]
            logging.info(f"Removed context about completed tasks from goal description")
        
        # Store essential fields that should be preserved
        preserved_fields = {
            "goal_id": goal_data["goal_id"],
            "description": description,
            "parent_goal": goal_data.get("parent_goal"),
            "timestamp": goal_data["timestamp"],
            "is_task": goal_data.get("is_task", False),
            "can_be_decomposed": goal_data.get("can_be_decomposed", goal_data.get("requires_further_decomposition", True)),
            "branch_name": goal_data.get("branch_name"),
            "initial_state": goal_data["initial_state"]
        }
        
        # Reset all state to initial values
        goal_data.clear()  # Clear all fields
        goal_data.update(preserved_fields)  # Restore preserved fields
        goal_data["current_state"] = goal_data["initial_state"].copy()
        
        # Update the goal file
        with open(goal_file, 'w') as f:
            json.dump(goal_data, f, indent=2)
        
        print(f"Successfully reverted goal {goal_id} to initial state")
        return True
    except Exception as e:
        logging.error(f"Failed to revert goal {goal_id}: {e}")
        return False


def update_parent_from_child(child_id):
    """Update a parent goal's state based on a child goal/task's current state."""
    goal_path = ensure_goal_dir()
    child_file = goal_path / f"{child_id}.json"
    
    if not child_file.exists():
        logging.error(f"Child goal/task not found: {child_id}")
        return False
    
    try:
        # Load the child data
        with open(child_file, 'r') as f:
            child_data = json.load(f)
        
        parent_id = child_data.get("parent_goal")
        if not parent_id:
            logging.error(f"Child {child_id} has no parent goal")
            return False
        
        # Get the current state from the child
        if "current_state" not in child_data:
            logging.error(f"Child {child_id} has no current state")
            return False
        
        # Create a mock execution result
        execution_result = type('ExecutionResult', (), {
            'git_hash': child_data["current_state"]["git_hash"],
            'success': True
        })
        
        # Update the parent's state
        success = update_parent_goal_state(
            parent_goal_id=parent_id,
            task_id=child_id,
            execution_result=execution_result,
            final_state=child_data["current_state"]
        )
        
        if success:
            print(f"Successfully updated parent goal {parent_id} with state from {child_id}")
        else:
            print(f"Failed to update parent goal {parent_id}")
        
        return success
    except Exception as e:
        logging.error(f"Failed to update parent from child: {str(e)}")
        return False


def show_validation_history(goal_id, debug=False, quiet=False):
    """
    Show validation history for a goal.
    
    Args:
        goal_id: ID of the goal to show validation history for
        debug: Whether to show debug output
        quiet: Whether to only show warnings and result
        
    Returns:
        bool: True if validation history was shown successfully, False otherwise
    """
    try:
        # Configure logging
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        elif quiet:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Import validation module for the function
        from midpoint.validation import ensure_validation_history_dir
        
        # Get the validation history directory
        validation_dir = ensure_validation_history_dir()
        
        # List all validation files for this goal
        validation_files = list(validation_dir.glob(f"{goal_id}_*.json"))
        
        # Filter out context files
        validation_records = [f for f in validation_files if not f.name.endswith("_context.json")]
        
        if not validation_records:
            print(f"No validation history found for goal {goal_id}")
            return True
        
        # Sort by timestamp (newest first)
        validation_records.sort(reverse=True)
        
        print(f"\nValidation History for Goal {goal_id}:")
        print("=" * 80)
        
        for i, record_file in enumerate(validation_records, 1):
            try:
                with open(record_file, 'r') as f:
                    record = json.load(f)
                
                # Extract key information
                timestamp = record.get("timestamp", "Unknown")
                score = record.get("score", 0.0)
                validated_by = record.get("validated_by", "Unknown")
                git_hash = record.get("git_hash", "Unknown")
                criteria_results = record.get("criteria_results", [])
                
                # Format timestamp for display
                if timestamp != "Unknown":
                    display_time = timestamp.replace("_", " ")
                else:
                    display_time = timestamp
                
                # Count passed criteria
                passed = sum(1 for c in criteria_results if c.get("passed", False))
                total = len(criteria_results)
                
                print(f"{i}. {display_time} - Score: {score:.2%} ({passed}/{total}) - By: {validated_by}")
                print(f"   Git Hash: {git_hash[:8]}")
                
                # Ask if user wants to see details
                if i < len(validation_records):  # Not the last record
                    choice = input("   Show details? (y/n/q): ").lower().strip()
                    if choice == 'q':
                        break
                    if choice == 'y':
                        print("\n   Criteria Results:")
                        for j, criterion in enumerate(criteria_results, 1):
                            crit_text = criterion.get("criterion", "Unknown")
                            passed = criterion.get("passed", False)
                            reasoning = criterion.get("reasoning", "")
                            
                            print(f"   {j}. {crit_text}")
                            print(f"      {'✅ Passed' if passed else '❌ Failed'}")
                            if reasoning:
                                print(f"      Reason: {reasoning}")
                        print()
                else:  # Last record, show details by default
                    print("\n   Criteria Results:")
                    for j, criterion in enumerate(criteria_results, 1):
                        crit_text = criterion.get("criterion", "Unknown")
                        passed = criterion.get("passed", False)
                        reasoning = criterion.get("reasoning", "")
                        
                        print(f"   {j}. {crit_text}")
                        print(f"      {'✅ Passed' if passed else '❌ Failed'}")
                        if reasoning:
                            print(f"      Reason: {reasoning}")
                    print()
                
                print("-" * 80)
                
            except Exception as e:
                if debug:
                    logging.error(f"Error reading validation record {record_file}: {e}")
                print(f"{i}. {record_file.name} - Error reading file")
                
        return True
    
    except Exception as e:
        logging.error(f"Failed to show validation history: {e}")
        return False


if __name__ == "__main__":
    main() 