"""Goal revert command implementation."""
"""Goal revert logic."""

import json
import logging
from pathlib import Path

def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path('.goal')
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: .goal")
    return goal_path


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
