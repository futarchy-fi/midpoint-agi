"""
Functions for managing goal state, including creation, completion, and merging.
"""

import json
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import os

# Import git-related functions
from .goal_git import (
    get_current_hash,
    get_current_branch,
    find_branch_for_goal
)

# Import file management functions
from .goal_file_management import generate_goal_id
from .constants import GOAL_DIR

def _get_git_repo_root(fallback_path: Optional[str] = None) -> str:
    """Return the git repository root (top-level) for the current working directory.

    Falls back to `fallback_path` (or `os.getcwd()`) if the current directory is not inside
    a git repository or git is unavailable.
    """
    fallback = fallback_path or os.getcwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        repo_root = result.stdout.strip()
        return repo_root or fallback
    except Exception:
        return fallback

def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path(GOAL_DIR)
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: {GOAL_DIR}")
    return goal_path

def create_goal_file(goal_id, description, parent_id=None, branch_name=None):
    """Create a goal file with initial details.
    
    Args:
        goal_id: The unique ID for the goal
        description: The goal description
        parent_id: The ID of the parent goal (if any)
        branch_name: The name of the branch associated with this goal (optional)
    
    Returns:
        Path to the created goal file, or None if failed.
    """
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if goal_file.exists():
        logging.error(f"Goal file {goal_file} already exists")
        return None
    
    # Get initial state
    try:
        current_hash = get_current_hash()
        current_branch = get_current_branch() if branch_name is None else branch_name
    except Exception as e:
        logging.error(f"Failed to get initial git state: {e}")
        return None
    
    # Initialize memory state for this goal (each goal should start with empty memory)
    memory_hash = None
    # Use ~/.midpoint/memory as default if MEMORY_REPO_PATH not set
    memory_repo_path = os.environ.get("MEMORY_REPO_PATH")
    if not memory_repo_path:
        # Expand ~ to user's home directory
        memory_repo_path = os.path.expanduser("~/.midpoint/memory")
        logging.info(f"MEMORY_REPO_PATH not set, using default: {memory_repo_path}")
        # Create directory if it doesn't exist
        os.makedirs(memory_repo_path, exist_ok=True)
    
    if memory_repo_path and os.path.exists(memory_repo_path):
        try:
            memory_hash = _ensure_memory_repo_initialized(memory_repo_path, goal_id)
        except Exception as e:
            logging.warning(f"Could not initialize memory state: {e}")
            # Still use the path even if we can't get the hash
    
    # Record the repo root rather than the current subdirectory.
    repo_root = _get_git_repo_root()

    initial_state_data = {
        "git_hash": current_hash,
        "repository_path": repo_root,
        "description": "Initial state before processing goal",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "memory_hash": memory_hash,
        "memory_repository_path": memory_repo_path
    }
    
    goal_data = {
        "goal_id": goal_id,
        "description": description,
        "parent_goal": parent_id or "",
        "branch_name": current_branch,
        "status": "pending",  # Initial status
        "is_task": False,  # All nodes start as potential goals
        "complete": False,
        "validation_criteria": [], # Add validation criteria field
        "initial_state": initial_state_data,
        "current_state": initial_state_data.copy(), # Start current state same as initial
        "created_at": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    try:
        with open(goal_file, 'w') as f:
            json.dump(goal_data, f, indent=2)
        logging.info(f"Created goal file: {goal_file}")
        return goal_file
    except Exception as e:
        logging.error(f"Failed to create goal file {goal_file}: {e}")
        return None

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

def create_new_child_goal(parent_id, description):
    """
    Create a new child goal (subgoal or task) under a parent.
    
    Note: The distinction between subgoal and task is determined later by the goal analyzer,
    not at creation time. All child goals start as regular goals with is_task=False.
    
    Args:
        parent_id: ID of the parent goal
        description: Description of the child goal
        
    Returns:
        ID of the newly created child goal, or None if creation failed
    """
    parent_file = Path(GOAL_DIR) / f"{parent_id}.json"
    if not parent_file.exists():
        logging.error(f"Parent goal {parent_id} not found")
        return None
    
    # Generate new ID
    child_id = generate_goal_id(parent_id=parent_id)
    
    # Create the goal file (it starts as a goal, analyzer determines if it's a task later)
    goal_file_path = create_goal_file(child_id, description, parent_id)
    if not goal_file_path:
        return None  # Error creating file

    logging.info(f"Created new child goal {child_id} under {parent_id}")
    return child_id

def mark_goal_complete(goal_id=None):
    """Mark a goal as complete."""
    # If no goal ID provided, use the current branch
    if not goal_id:
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        from .goal_git import get_goal_id_from_branch
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
        except Exception as e:
            logging.warning(f"Failed to read goal file: {e}")
            continue
    
    return child_tasks

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

def update_git_state(goal_id):
    """
    Update the git state in the goal file after changes have been made.
    
    Args:
        goal_id: ID of the goal to update
    """
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.warning(f"Goal file {goal_id}.json not found")
        return False
    
    try:
        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True
        )
        
        has_changes = bool(result.stdout.strip())
        if has_changes:
            # Commit the changes
            logging.info(f"Committing changes for goal {goal_id}")
            subprocess.run(
                ["git", "add", "."],
                check=True,
                capture_output=True,
                text=True
            )
            
            subprocess.run(
                ["git", "commit", "-m", f"feat({goal_id}): Progress from auto-solve process"],
                check=True,
                capture_output=True,
                text=True
            )
            
            logging.info("Changes committed successfully")
        
        # Get the current hash after potential commit
        current_hash = get_current_hash()
        
        # Load the goal file
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
        
        # Update the current state with the new hash
        goal_data["current_state"]["git_hash"] = current_hash
        goal_data["current_state"]["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Write updated goal data
        with open(goal_file, 'w') as f:
            json.dump(goal_data, f, indent=2)
            
        logging.info(f"Updated goal state for {goal_id} with new hash {current_hash[:8]}")
        return True
    except Exception as e:
        logging.error(f"Error updating git state: {e}")
        return False 