import re
import json
import logging
import datetime
import subprocess
from pathlib import Path

def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path('.goal')
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
    
    # --- START EDIT: Use 'S' prefix for subgoals, 'G' for top-level ---
    # Determine prefix based on parent_id
    if parent_id:
        prefix = "S"  # Subgoal prefix
    else:
        prefix = "G"  # Top-level goal prefix

    max_num = 0
    # --- END EDIT ---
    
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

def delete_goal(goal_id):
    """Delete a goal, subgoal, or task and its associated branch."""
    from .goal_git import get_current_branch
    
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
                                ["git", "checkout", "master"],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                        except subprocess.CalledProcessError:
                            # Fallback if master doesn't exist
                            logging.warning(f"Could not checkout master while trying to delete branch {branch_name}. Attempting deletion anyway.")
                            pass
                    
                    # Delete the branch
                    logging.info(f"Attempting to delete branch: {branch_name}")
                    subprocess.run(
                        ["git", "branch", "-D", branch_name],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    print(f"Deleted branch: {branch_name}")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to delete branch {branch_name}: {e.stderr}")
                    print(f"Warning: Failed to delete branch {branch_name}. It might need manual deletion.")
        
        # Recursively delete all child goals and tasks
        for child_id in child_goals + child_tasks:
            delete_goal(child_id)
        
        print(f"Successfully deleted goal {goal_id}")
        return True
    except Exception as e:
        logging.error(f"Failed to delete goal {goal_id}: {e}")
        return False
