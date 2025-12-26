"""
Functions for updating validation criteria for existing goals.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from midpoint.goal_file_management import ensure_goal_dir
from midpoint.goal_criteria import generate_validation_criteria, prompt_for_validation_criteria

logger = logging.getLogger(__name__)


def update_goal_validation_criteria(goal_id: str, criteria: Optional[List[str]] = None, auto: bool = False) -> bool:
    """
    Update validation criteria for an existing goal.
    
    Args:
        goal_id: ID of the goal to update
        criteria: Optional list of criteria strings (if None and not auto, will prompt)
        auto: If True, auto-generate criteria using AI
        
    Returns:
        True if successful, False otherwise
    """
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logger.error(f"Goal {goal_id} not found")
        print(f"Error: Goal {goal_id} not found")
        return False
    
    try:
        # Load goal data
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
        
        description = goal_data.get("description", "")
        
        # Determine new criteria
        if auto:
            # Auto-generate
            print(f"\nGenerating validation criteria for goal {goal_id}...")
            print(f"Goal: {description}\n")
            try:
                new_criteria = generate_validation_criteria(description)
                print(f"\nGenerated {len(new_criteria)} criteria:")
                for i, criterion in enumerate(new_criteria, 1):
                    print(f"  {i}. {criterion}")
            except Exception as e:
                logger.error(f"Failed to generate criteria: {e}")
                print(f"Error generating criteria: {e}")
                return False
        elif criteria:
            # Use provided criteria
            new_criteria = criteria
        else:
            # Prompt user
            new_criteria = prompt_for_validation_criteria(description)
        
        # Update goal data
        goal_data["validation_criteria"] = new_criteria
        
        # Save updated goal file
        with open(goal_file, 'w') as f:
            json.dump(goal_data, f, indent=2)
        
        print(f"\nUpdated validation criteria for goal {goal_id}")
        if new_criteria:
            print(f"Criteria ({len(new_criteria)}):")
            for i, criterion in enumerate(new_criteria, 1):
                print(f"  {i}. {criterion}")
        else:
            print("(No criteria - goal will use description as implicit criteria)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update validation criteria for goal {goal_id}: {e}")
        print(f"Error: {e}")
        return False

