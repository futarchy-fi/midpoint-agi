import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# --- Constants ---
GOAL_DIR = ".goal"

# --- Helper Functions (Consider moving to a shared utils module later) ---

def _ensure_goal_dir() -> Path:
    """Ensure the goal directory exists and return its path."""
    goal_path = Path(GOAL_DIR)
    goal_path.mkdir(parents=True, exist_ok=True)
    return goal_path

def _load_goal_data(goal_id: str) -> Optional[Dict[str, Any]]:
    """Load JSON data for a specific goal ID."""
    goal_path = _ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    if not goal_file.exists():
        logging.error(f"Goal file not found: {goal_file}")
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

def _save_goal_data(goal_id: str, data: dict) -> bool:
    """Save JSON data for a specific goal ID."""
    goal_path = _ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    try:
        with open(goal_file, 'w') as f:
            json.dump(data, f, indent=2)
        logging.debug(f"Saved updated goal data to: {goal_file}")
        return True
    except Exception as e:
        logging.error(f"Error writing goal file {goal_file}: {e}")
        return False

# --- Propagation Functions ---

def propagate_success_state_to_parent(child_id: str) -> bool:
    """
    Propagates the successful state (e.g., git hash) from a completed and
    validated child goal/task to its parent goal.
    Assumes the child's state represents the desired state for the parent
    after the child's successful completion.
    
    Also updates the parent's completed_subgoals list to track which child goals
    have been completed, with validation details.
    """
    logging.info(f"Attempting to propagate success state from child {child_id} to parent.")
    child_data = _load_goal_data(child_id)
    if not child_data:
        return False

    parent_id = child_data.get("parent_goal")
    if not parent_id:
        logging.warning(f"Child {child_id} has no parent_goal field. Cannot propagate.")
        return True

    child_complete = child_data.get("complete", False) or child_data.get("completed", False)
    if not child_complete:
         logging.warning(f"Child {child_id} is not marked as complete. Cannot propagate success state.")
         return False

    child_state = child_data.get("current_state")
    if not child_state or not isinstance(child_state, dict):
        logging.error(f"Child {child_id} has no valid 'current_state'. Cannot propagate.")
        return False

    parent_data = _load_goal_data(parent_id)
    if not parent_data:
        return False

    logging.info(f"Updating parent {parent_id} state with child {child_id}'s state: {child_state.get('git_hash', 'N/A')[:8]}")
    parent_data["current_state"] = child_state
    parent_data["last_updated_by_child"] = child_id
    parent_data["last_update_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create or update the completed_subgoals list
    if "completed_subgoals" not in parent_data or not isinstance(parent_data["completed_subgoals"], list):
        parent_data["completed_subgoals"] = []
    
    # Check if this child is already in the completed_subgoals list
    child_already_tracked = False
    for i, entry in enumerate(parent_data["completed_subgoals"]):
        if isinstance(entry, dict) and entry.get("subgoal_id") == child_id:
            # Update the existing entry with latest information
            child_already_tracked = True
            parent_data["completed_subgoals"][i] = {
                "subgoal_id": child_id,
                "description": child_data.get("description", ""),
                "completion_time": child_data.get("completion_time") or child_data.get("completion_timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S"),
                "validation_status": child_data.get("validation_status", {}),
                "final_state": {
                    "git_hash": child_state.get("git_hash", ""),
                    "memory_hash": child_state.get("memory_hash", "")
                }
            }
            break
    
    # If not already tracked, add it to the list
    if not child_already_tracked:
        parent_data["completed_subgoals"].append({
            "subgoal_id": child_id,
            "description": child_data.get("description", ""),
            "completion_time": child_data.get("completion_time") or child_data.get("completion_timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "validation_status": child_data.get("validation_status", {}),
            "final_state": {
                "git_hash": child_state.get("git_hash", ""),
                "memory_hash": child_state.get("memory_hash", "")
            }
        })
    
    # For backward compatibility, update completed_tasks/merged_subgoals if they exist
    # but don't create them if they don't exist
    if "completed_tasks" in parent_data and isinstance(parent_data["completed_tasks"], list):
        # Check if this child is already in completed_tasks
        task_exists = False
        for task in parent_data["completed_tasks"]:
            if isinstance(task, dict) and task.get("task_id") == child_id:
                task_exists = True
                break
        
        # If not, add it for backward compatibility
        if not task_exists:
            parent_data["completed_tasks"].append({
                "task_id": child_id,
                "description": child_data.get("description", ""),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "final_state": {
                    "git_hash": child_state.get("git_hash", ""),
                    "memory_hash": child_state.get("memory_hash", "")
                }
            })
    
    # Update completion status counters for better tracking
    parent_data["completed_subgoal_count"] = len(parent_data["completed_subgoals"])
    total_children = len(_get_children_ids(parent_id))
    parent_data["total_subgoal_count"] = total_children
    
    return _save_goal_data(parent_id, parent_data)

# Helper function to get all children IDs for a parent
def _get_children_ids(parent_id: str) -> list:
    """Get IDs of all child goals/subgoals for a given parent."""
    children = []
    goal_dir = _ensure_goal_dir()
    
    try:
        for file_path in goal_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Check if this is a child of the parent (case-insensitive)
            if data.get("parent_goal", "").upper() == parent_id.upper():
                children.append(data.get("goal_id"))
    except Exception as e:
        logging.error(f"Error getting children for {parent_id}: {e}")
    
    return children

def propagate_failure_history_to_parent(child_id: str) -> bool:
    """
    Propagates the terminal failure (give_up) of a child goal/task to its
    parent's failure history. Reads details (reason, description, criteria)
    from the child's goal file. Overwrites previous failure entries for the same child.
    Does NOT update the parent's state.
    """
    logging.info(f"Attempting to propagate failure history from child {child_id} to parent.")
    child_data = _load_goal_data(child_id)
    if not child_data:
        return False # Child data must exist

    parent_id = child_data.get("parent_goal")
    if not parent_id:
        logging.warning(f"Child {child_id} has no parent_goal field. Cannot propagate failure history.")
        return True # Not an error if child was top-level

    # --- Extract details from child data --- 
    failure_reason = "No failure reason found in child goal file."
    # Prioritize last_analysis justification, fallback to last_execution summary
    last_analysis = child_data.get('last_analysis')
    last_execution = child_data.get('last_execution')
    if last_analysis and isinstance(last_analysis, dict) and last_analysis.get('justification'):
        failure_reason = last_analysis['justification']
    elif last_execution and isinstance(last_execution, dict) and last_execution.get('summary'):
        failure_reason = last_execution['summary']

    child_description = child_data.get("description", "No description found in child goal file.")
    child_criteria = child_data.get("validation_criteria", [])
    # ----------------------------------------

    # Load parent data
    parent_data = _load_goal_data(parent_id)
    if not parent_data:
        return False # Parent must exist

    # Ensure history list exists and is a list
    if "failed_attempts_history" not in parent_data or not isinstance(parent_data.get("failed_attempts_history"), list):
        parent_data["failed_attempts_history"] = []

    # Prepare new/updated entry details
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_failure_entry = {
        "attempt_type": "child_give_up",
        "failed_child_goal_id": child_id,
        "failed_child_description": child_description,
        "failed_child_validation_criteria": child_criteria,
        "failure_timestamp": timestamp,
        "failure_reason": failure_reason
    }

    # Check if entry for this child already exists and update it (overwrite)
    updated = False
    for i, entry in enumerate(parent_data["failed_attempts_history"]):
        if isinstance(entry, dict) and entry.get("failed_child_goal_id") == child_id:
            logging.info(f"Overwriting existing failure history entry for child {child_id} in parent {parent_id}.")
            parent_data["failed_attempts_history"][i] = new_failure_entry
            updated = True
            break

    # If no existing entry was found, append the new one
    if not updated:
        logging.info(f"Appending new failure history entry for child {child_id} to parent {parent_id}.")
        parent_data["failed_attempts_history"].append(new_failure_entry)

    return _save_goal_data(parent_id, parent_data) 