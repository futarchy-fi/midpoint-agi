import re
import logging
import datetime
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
