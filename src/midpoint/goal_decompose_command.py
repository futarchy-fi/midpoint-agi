"""CLI orchestration for goal decomposition."""

import os
import sys
import json
import logging
import datetime
import subprocess
from pathlib import Path

from .agents.goal_decomposer import decompose_goal as agent_decompose_goal
from .goal_file_management import generate_goal_id
from .goal_git import get_current_branch, find_top_level_branch
from .constants import GOAL_DIR

def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path(GOAL_DIR)
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: {GOAL_DIR}")
    return goal_path


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
            input_file=str(goal_file), # Pass the goal file as input_file
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
            
            # Add debug output to show all subtasks if available
            if "all_subtasks" in result and result["all_subtasks"]:
                print("\n--- DEBUG: All subtasks identified by decomposer ---")
                for i, subtask in enumerate(result["all_subtasks"], 1):
                    print(f"{i}. {subtask}")
                print("---------------------------------------------------\n")
            
            if result.get("goal_completed", False):
                print("\n✅ Goal completed!")
                print(f"Summary: {result['completion_summary']}")
                print(f"Reasoning: {result['reasoning']}")
            else:
                print(f"\nNext state: {result['next_state']}")
                print("\nValidation criteria:")
                for criterion in result["validation_criteria"]:
                    print(f"- {criterion}")
                
                # Add further steps to the output if available
                if "further_steps" in result and result["further_steps"]:
                    print("\nFurther steps to complete the goal:")
                    for step in result["further_steps"]:
                        print(f"- {step}")
                
                print(f"\nGoal file: {result['goal_file']}")
            
            # Get updated memory hash after decomposition
            updated_memory_hash = memory_hash
            if memory_repo_path:
                try:
                    from .agents.tools.git_tools import get_current_hash
                    updated_memory_hash = get_current_hash(memory_repo_path)
                    if updated_memory_hash != memory_hash:
                        logging.info(f"Memory hash updated during decomposition: {updated_memory_hash[:8]}")
                except Exception as e:
                    logging.warning(f"Failed to get updated memory hash: {e}")
            
            # Update the goal file with the decomposition result
            goal_data.update({
                "goal_completed": result.get("goal_completed", False),
                "completion_summary": result.get("completion_summary"),
                "next_state": result.get("next_state"),
                "validation_criteria": result.get("validation_criteria", []),
                "reasoning": result["reasoning"],
                "relevant_context": result.get("relevant_context", {}),
                "further_steps": result.get("further_steps", []),
                "decomposed": True,
            })
            
            if result.get("goal_completed", False):
                goal_data.update({
                    "completed": True,
                    "completion_timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "completion_summary": result["completion_summary"],
                    "completion_reasoning": result["reasoning"]
                })
            
            with open(goal_file, 'w') as f:
                json.dump(goal_data, f, indent=2)
            
            if not result.get("goal_completed", False):
                is_task = False
                subgoal_id = generate_goal_id(goal_id, is_task=is_task)
                subgoal_file = goal_path / f"{subgoal_id}.json"
                subgoal_data = {
                    "goal_id": subgoal_id,
                    "description": result["next_state"],
                    "parent_goal": goal_id,
                    "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "is_task": is_task,
                    "validation_criteria": result["validation_criteria"],
                    "reasoning": result["reasoning"],
                    "relevant_context": result.get("relevant_context", {}),
                    "further_steps": result.get("further_steps", []),
                    "initial_state": {
                        "git_hash": result["git_hash"],
                        "repository_path": goal_data.get("current_state", {}).get("repository_path", os.getcwd()),
                        "description": "Initial state before executing subgoal",
                        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "memory_hash": updated_memory_hash,
                        "memory_repository_path": memory_repo_path
                    },
                    "current_state": {
                        "git_hash": result["git_hash"],
                        "repository_path": goal_data.get("current_state", {}).get("repository_path", os.getcwd()),
                        "description": "Initial state before executing subgoal",
                        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "memory_hash": updated_memory_hash,
                        "memory_repository_path": memory_repo_path
                    }
                }
                with open(subgoal_file, 'w') as f:
                    json.dump(subgoal_data, f, indent=2)
                print(f"Created {'task' if is_task else 'subgoal'} file: {subgoal_file}")
            return True
        else:
            error_msg = result.get('error', 'Unknown error')
            logging.error(f"Failed to decompose goal: {error_msg}")
            print(f"Failed to decompose goal: {error_msg}", file=sys.stderr)
            return False
    finally:
        try:
            subprocess.run(
                ["git", "checkout", current_branch],
                check=True,
                capture_output=True,
                text=True
            )
            if has_changes:
                subprocess.run(
                    ["git", "stash", "pop"],
                    check=True,
                    capture_output=True,
                    text=True
                )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to restore original state: {e}")
    
