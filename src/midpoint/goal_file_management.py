"""
Goal file management module.

This module will contain functions for creating, updating, deleting, and listing goal, subgoal, and task files.
"""

# TODO: Move goal file management functions here from goal_cli.py

import os
import json
import logging
import datetime
import re
from pathlib import Path
from .goal_git import get_current_hash, get_current_branch
import subprocess
from typing import Optional, Dict

def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path(".goal")
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: .goal")
    return goal_path


def generate_goal_id(parent_id=None, is_task=False):
    """Generate a unique goal ID, ensuring it doesn't already exist.
    
    Uses 'S' prefix for subgoals (when parent_id is present) and 'G' for top-level goals.

    Args:
        parent_id: ID of the parent goal (optional)
        is_task: Deprecated, no longer used for naming.

    Returns:
        A unique goal ID string (e.g., "G1", "S1")
    """
    goal_path = ensure_goal_dir()
    
    # Determine prefix based on parent_id
    if parent_id:
        prefix = "S"  # Subgoal prefix
    else:
        prefix = "G"  # Top-level goal prefix

    max_num = 0
    
    for file_path in goal_path.glob(f"{prefix}*.json"):
        # Match only files with pattern <prefix> followed by digits and .json
        match = re.match(rf"{prefix}(\d+)\.json$", file_path.name)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    
    # Next goal number is one more than the maximum found
    next_num = max_num + 1
    new_id = f"{prefix}{next_num}"
    
    # Double check for collision (should be unlikely)
    while (goal_path / f"{new_id}.json").exists():
        logging.warning(f"Generated ID {new_id} already exists, generating next...")
        next_num += 1
        new_id = f"{prefix}{next_num}"
        
    logging.info(f"Generated new Goal ID: {new_id}")
    return new_id


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
    
    # Try to get memory state (optional)
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
            memory_hash = get_current_hash(memory_repo_path)
        except Exception as e:
            logging.warning(f"Could not get initial memory hash: {e}")
            # Still use the path even if we can't get the hash
    
    initial_state_data = {
        "git_hash": current_hash,
        "repository_path": os.getcwd(),
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
            "is_task": False
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
    """Create a new subgoal under a parent."""
    parent_file = Path(".goal") / f"{parent_id}.json"
    if not parent_file.exists():
        logging.error(f"Parent goal {parent_id} not found")
        return None
    
    # Generate new ID (no longer depends on type)
    subgoal_id = generate_goal_id(parent_id=parent_id) 
    
    # Create the goal file
    create_goal_file(subgoal_id, description, parent_id)
    logging.info(f"Created new subgoal {subgoal_id} under {parent_id}")
    return subgoal_id


def create_new_task(parent_id, description):
    """Create a new task under a parent goal."""
    parent_file = Path(".goal") / f"{parent_id}.json"
    if not parent_file.exists():
        logging.error(f"Parent goal {parent_id} not found")
        return None
    
    # Generate new ID (no longer depends on type)
    task_id = generate_goal_id(parent_id=parent_id)
    
    # Create the goal file (it starts as a goal, analyzer determines if it's a task later)
    goal_file_path = create_goal_file(task_id, description, parent_id)
    if not goal_file_path:
        return None # Error creating file
        
    # Optionally, we could mark it immediately as intended to be a task,
    # but the analyzer should handle this. For now, just create as a goal.
    # try:
    #     with open(goal_file_path, 'r+') as f:
    #         data = json.load(f)
    #         data["is_task"] = True
    #         f.seek(0)
    #         json.dump(data, f, indent=2)
    #         f.truncate()
    # except Exception as e:
    #     logging.error(f"Failed to mark {task_id} as task: {e}")
    #     # Proceed anyway, as the file was created

    logging.info(f"Created new goal node {task_id} (intended as task) under {parent_id}")
    return task_id


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
            # Safeguard for critical branches
            if branch_name in ["master", "main"]:
                logging.warning(f"Skipping deletion of critical branch '{branch_name}' potentially associated with goal {goal_id}.")
                print(f"Warning: Did not delete critical branch '{branch_name}'. Please check goal configurations.")
            else:
                # Proceed with deleting the non-critical branch
                try:
                    # Switch to a different branch if we're on the one we're deleting
                    current_branch = get_current_branch()
                    if current_branch == branch_name:
                        # Switch to master branch (assuming it exists)
                        try:
                            subprocess.run(
                                ["git", "checkout", "master"], # Changed from "main" to "master"
                                check=True,
                                capture_output=True,
                                text=True
                            )
                        except subprocess.CalledProcessError:
                            # Fallback if master doesn't exist
                            logging.warning(f"Could not checkout master while trying to delete branch {branch_name}. Attempting deletion anyway.")
                            # Proceed with deletion attempt even if checkout fails
                            pass
                    
                    # Delete the branch
                    logging.info(f"Attempting to delete branch: {branch_name}")
                    subprocess.run(
                        ["git", "branch", "-D", branch_name],
                        check=True, # Will raise CalledProcessError if deletion fails
                        capture_output=True,
                        text=True
                    )
                    print(f"Deleted branch: {branch_name}")
                except subprocess.CalledProcessError as e:
                    # Catch errors specifically from the deletion process (checkout or delete)
                    logging.error(f"Failed to delete branch {branch_name}: {e.stderr}")
                    # Continue with goal file deletion even if branch deletion fails
                    print(f"Warning: Failed to delete branch {branch_name}. It might need manual deletion.")
        
        # Recursively delete all child goals and tasks
        for child_id in child_goals + child_tasks:
            delete_goal(child_id)
        
        print(f"Successfully deleted goal {goal_id}")
        return True
    except Exception as e:
        logging.error(f"Failed to delete goal {goal_id}: {e}")
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


def load_goal_data(goal_id: str) -> Optional[Dict[str, any]]:
    """Loads the JSON data for a given goal ID."""
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    if not goal_file.exists():
        logging.error(f"Goal file not found for ID: {goal_id}")
        return None
    try:
        with open(goal_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from goal file: {goal_file}")
        return None
    except Exception as e:
        logging.error(f"Error reading goal file {goal_file}: {e}")
        return None


def _get_children_details(parent_goal_id: str) -> list[dict]:
    """Get details of direct children (subgoals and tasks) for a given parent ID."""
    goal_path = Path(".goal")
    children_details = []
    try:
        for file_path in goal_path.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check parent match (case-insensitive)
            parent_match = False
            file_parent = data.get("parent_goal", "")
            if file_parent:
                 if file_parent.upper() == parent_goal_id.upper() or \
                   file_parent.upper() == f"{parent_goal_id.upper()}.json":
                    parent_match = True

            if parent_match:
                child_info = {
                    "goal_id": data.get("goal_id", "Unknown"),
                    "description": data.get("description", "N/A"),
                    "is_task": data.get("is_task", False),
                    "complete": data.get("complete", False) or data.get("completed", False),
                    # Add other relevant fields if needed, e.g., last validation score
                    "validation_score": data.get("validation_status", {}).get("last_score")
                }
                children_details.append(child_info)
                
    except Exception as e:
        logging.error(f"Error getting children details for {parent_goal_id}: {e}")
        # Return potentially partial list or indicate error?
        # For now, just return what we have.

    # Sort by goal ID for consistent order
    children_details.sort(key=lambda x: x["goal_id"])
    return children_details


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
