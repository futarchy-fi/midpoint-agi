"""
Validation module for goal validation.
This module provides functionality for validating goals, both manually and using LLM.
"""

import os
import json
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import from midpoint modules
from midpoint.goal_cli import (
    ensure_goal_dir,
    get_current_hash,
    get_current_branch
)


async def get_repository_state(goal_id: str, debug: bool = False) -> Dict[str, Any]:
    """
    Gather comprehensive repository state information for validation.
    
    Args:
        goal_id: ID of the goal being validated
        debug: Whether to show debug output
        
    Returns:
        dict: Repository state information
    """
    repo_state = {
        "current_hash": get_current_hash(),
        "current_branch": get_current_branch(),
        "timestamp": datetime.datetime.now().isoformat(),
        "goal_id": goal_id
    }
    
    try:
        # Get last commit message
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True,
            text=True,
            check=True
        )
        repo_state["last_commit_message"] = result.stdout.strip()
        
        # Get list of files changed since goal creation
        goal_path = ensure_goal_dir()
        goal_file = goal_path / f"{goal_id}.json"
        
        if goal_file.exists():
            with open(goal_file, 'r') as f:
                goal_data = json.load(f)
                
            # If goal has an initial_hash field, use it to get changed files
            if "initial_hash" in goal_data:
                initial_hash = goal_data["initial_hash"]
                result = subprocess.run(
                    ["git", "diff", "--name-only", initial_hash, "HEAD"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    repo_state["changed_files"] = [
                        file for file in result.stdout.strip().split("\n") 
                        if file.strip()
                    ]
        
        if debug:
            print(f"Debug: Gathered repository state information:")
            print(f"  - Current hash: {repo_state['current_hash']}")
            print(f"  - Current branch: {repo_state['current_branch']}")
            print(f"  - Changed files: {len(repo_state.get('changed_files', []))}")
            
    except Exception as e:
        if debug:
            print(f"Debug: Error gathering repository state: {e}")
    
    return repo_state


def create_validation_record(goal_id: str, criteria_results: List[Dict[str, Any]], 
                           repo_state: Dict[str, Any], auto: bool = False) -> Dict[str, Any]:
    """
    Create a standardized validation record.
    
    Args:
        goal_id: ID of the goal being validated
        criteria_results: List of criteria validation results
        repo_state: Repository state information
        auto: Whether this was an automated validation
        
    Returns:
        dict: Validation record
    """
    # Calculate overall score
    passed_count = sum(1 for result in criteria_results if result.get("passed", False))
    total_count = len(criteria_results)
    score = passed_count / total_count if total_count > 0 else 0.0
    
    # Create validation record
    validation_record = {
        "goal_id": goal_id,
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_hash": repo_state["current_hash"],
        "branch": repo_state["current_branch"],
        "criteria_results": criteria_results,
        "score": score,
        "validated_by": "LLM" if auto else os.getenv("USER", "unknown"),
        "automated": auto,
        "repository_state": repo_state
    }
    
    return validation_record


async def save_validation_record(goal_id: str, validation_record: Dict[str, Any]) -> str:
    """
    Save a validation record to the validation history directory.
    
    Args:
        goal_id: ID of the goal being validated
        validation_record: Validation record to save
        
    Returns:
        str: Path to the saved validation record file
    """
    # Create validation history directory if it doesn't exist
    goal_path = ensure_goal_dir()
    validation_dir = goal_path / "validation_history"
    if not validation_dir.exists():
        validation_dir.mkdir()
    
    # Save validation record
    validation_file = validation_dir / f"{goal_id}_{validation_record['timestamp']}.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_record, f, indent=2)
    
    # Update goal data with validation status
    goal_file = goal_path / f"{goal_id}.json"
    if goal_file.exists():
        try:
            with open(goal_file, 'r') as f:
                goal_data = json.load(f)
            
            goal_data["validation_status"] = {
                "last_validated": validation_record["timestamp"],
                "last_score": validation_record["score"],
                "last_validated_by": validation_record["validated_by"],
                "last_git_hash": validation_record["git_hash"],
                "last_branch": validation_record["branch"],
                "automated": validation_record["automated"]
            }
            
            with open(goal_file, 'w') as f:
                json.dump(goal_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to update goal data: {e}")
    
    return str(validation_file)


async def perform_automated_validation(goal_id: str, goal_data: Dict[str, Any], 
                                     debug: bool = False, quiet: bool = False) -> Dict[str, Any]:
    """
    Perform automated validation of a goal using LLM.
    
    Args:
        goal_id: ID of the goal to validate
        goal_data: The loaded goal data
        debug: Whether to show debug output
        quiet: Whether to only show warnings and result
        
    Returns:
        dict: Dictionary with validation results containing:
            - criteria_results: List of dictionaries with validation results for each criterion
            - score: Overall validation score (0.0 to 1.0)
    """
    criteria = goal_data.get("validation_criteria", [])
    
    if not quiet:
        print("\nPerforming automated validation using LLM...")
    
    if debug:
        print(f"Debug: Found {len(criteria)} validation criteria")
    
    # Placeholder for actual LLM validation
    # In a real implementation, this would:
    # 1. Prepare context about the goal and its state
    # 2. Create a validation prompt for the LLM
    # 3. Send the prompt to the LLM API
    # 4. Parse the LLM response into validation results
    
    # For now, we'll return mock results
    criteria_results = []
    for criterion in criteria:
        criteria_results.append({
            "criterion": criterion,
            "passed": False,  # Mock result - actual implementation would determine this
            "reasoning": "This is a placeholder for LLM reasoning",
            "evidence": []
        })
    
    # Calculate overall score
    passed_count = sum(1 for result in criteria_results if result["passed"])
    total_count = len(criteria_results)
    score = passed_count / total_count if total_count > 0 else 0.0
    
    return {
        "criteria_results": criteria_results,
        "score": score
    }


async def validate_goal(goal_id: str, debug: bool = False, quiet: bool = False, auto: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a goal's completion criteria.
    
    Args:
        goal_id: ID of the goal to validate
        debug: Whether to show debug output
        quiet: Whether to only show warnings and result
        auto: Whether to use automated LLM validation
        
    Returns:
        Tuple with:
            - bool: True if validation was successful, False otherwise
            - dict: Validation record (or empty dict if validation failed)
    """
    # Ensure goal directory exists
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal not found: {goal_id}")
        return False, {}
    
    try:
        # Load goal data
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
        
        # Get validation criteria
        criteria = goal_data.get("validation_criteria", [])
        if not criteria:
            logging.error(f"No validation criteria found for goal {goal_id}")
            return False, {}
        
        # Get repository state information
        repo_state = await get_repository_state(goal_id, debug)
        
        # Determine validation method based on auto flag
        if auto:
            # Use automated validation
            validation_results = await perform_automated_validation(goal_id, goal_data, debug, quiet)
            criteria_results = validation_results["criteria_results"]
        else:
            # For manual validation, we'll defer to CLI to handle user interaction
            # and just return the goal data and repository state
            return True, {
                "goal_data": goal_data,
                "repo_state": repo_state,
                "criteria": criteria
            }
        
        # Create and save validation record
        validation_record = create_validation_record(goal_id, criteria_results, repo_state, auto)
        await save_validation_record(goal_id, validation_record)
        
        return True, validation_record
        
    except Exception as e:
        logging.error(f"Failed to validate goal {goal_id}: {e}")
        return False, {} 