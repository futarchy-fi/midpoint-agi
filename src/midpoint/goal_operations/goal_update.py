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

    return _save_goal_data(parent_id, parent_data)

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