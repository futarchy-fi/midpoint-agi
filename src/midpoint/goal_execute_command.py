"""CLI orchestration for task execution."""

import os
import json
import sys
import logging
import datetime
import subprocess
from pathlib import Path

from .agents.task_executor import TaskExecutor, configure_logging as configure_executor_logging
from .agents.models import TaskContext, State, Goal, MemoryState
from .goal_git import get_current_branch, find_top_level_branch

def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path('.goal')
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: .goal")
    return goal_path


def execute_task(
    node_id,
    debug=False,
    quiet=False,
    bypass_validation=False,
    no_commit=False,
    memory_repo=None,
):
    """Execute a goal/task node using the TaskExecutor.

    This intentionally allows executing `G*`/`S*`/`T*` nodes directly, without
    requiring an extra "task node" wrapper.
    """
    # Get the node file path
    goal_path = ensure_goal_dir()
    task_file = goal_path / f"{node_id}.json"
    
    if not task_file.exists():
        logging.error(f"Goal/task node {node_id} not found")
        return False
    
    # Load the task data
    try:
        with open(task_file, 'r') as f:
            task_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read node file: {e}")
        return False
    
    # Find the top-level goal's branch
    top_level_branch = find_top_level_branch(node_id)
    if not top_level_branch:
        logging.error(f"Failed to find top-level goal branch for {node_id}")
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
                ["git", "stash", "push", "-m", f"Stashing changes before executing node {node_id}"],
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
        
        # Get memory state from task data
        memory_state = None
        if "initial_state" in task_data:
            initial_state = task_data["initial_state"]
            if "memory_hash" in initial_state and "memory_repository_path" in initial_state:
                memory_state = MemoryState(
                    memory_hash=initial_state["memory_hash"],
                    repository_path=initial_state["memory_repository_path"]
                )
                if initial_state["memory_hash"]:
                    logging.info(f"Using memory state from task file - hash: {initial_state['memory_hash'][:8]}")
                else:
                    logging.info("Memory hash is None in task file")
        
        # Create task context
        context = TaskContext(
            state=State(
                git_hash=task_data["initial_state"]["git_hash"],
                repository_path=task_data["initial_state"]["repository_path"],
                description=task_data["initial_state"]["description"],
                branch_name=top_level_branch,
                memory_hash=task_data["initial_state"].get("memory_hash"),
                memory_repository_path=task_data["initial_state"].get("memory_repository_path")
            ),
            goal=Goal(
                description=task_data["description"],
                validation_criteria=[]
            ),
            iteration=0,
            execution_history=[],
            memory_state=memory_state
        )
        
        # Configure logging
        configure_executor_logging(debug, quiet)
        
        # Create and run the executor
        executor = TaskExecutor()
        execution_result = executor.execute_task(context, task_data["description"])

        # --- Prepare the data for last_execution_result field ---
        last_execution_data = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "success": execution_result.success,
            "summary": execution_result.summary,
            "suggested_validation_steps": execution_result.suggested_validation_steps,
            "final_git_hash": execution_result.final_state.git_hash if execution_result.final_state else None,
            "final_memory_hash": execution_result.final_state.memory_hash if execution_result.final_state else None,
            "error_message": execution_result.error_message
        }
        # Remove null fields for cleaner output
        last_execution_data = {k: v for k, v in last_execution_data.items() if v is not None}

        # --- Update task_data regardless of success/failure ---
        # Persist both the newer `last_execution_result` and the legacy `last_execution`
        # field so other parts of the system (analyzer, status UI, update-parent) can
        # incorporate failures into subsequent decisions.
        task_data["last_execution_result"] = last_execution_data
        task_data["last_execution"] = last_execution_data

        if execution_result.success:
            logging.info(f"Node {node_id} execution reported success.")
            
            # Still mark task as completed conceptually upon successful execution report
            # (Though actual merging/acceptance is separate)
            task_data["complete"] = True
            task_data["completion_time"] = last_execution_data["timestamp"]

            # Save updated task data (now only updates complete, completion_time, and last_execution_result)
            try:
                with open(task_file, 'w') as f:
                    json.dump(task_data, f, indent=2)
                logging.info(f"Saved updated node data for {node_id} (Success)")
            except Exception as e:
                logging.error(f"Failed to save successful node data for {node_id}: {e}")
                # Even if save fails, we proceed to return True as execution succeeded

            print(f"\nNode {node_id} executed successfully.")
            print(f"Summary: {execution_result.summary}")
            if execution_result.suggested_validation_steps:
                 print("Suggested Validation Steps:")
                 for step in execution_result.suggested_validation_steps:
                     print(f"- {step}")
            return True
        else:
            logging.warning(f"Node {node_id} execution reported failure.")
            print(f"Failed to execute node {node_id}: {execution_result.summary or execution_result.error_message}", file=sys.stderr)

            # Update task data to reflect failure - keep current_state as it was before this failed attempt
            task_data["complete"] = False
            if "completion_time" in task_data:
                del task_data["completion_time"] # Remove previous completion time if it exists
            
            # Save updated task data with failure status and last_execution_result
            try:
                with open(task_file, 'w') as f:
                    json.dump(task_data, f, indent=2)
                logging.info(f"Saved updated node data for {node_id} (Failure)")
            except Exception as e:
                logging.error(f"Failed to save failed node data for {node_id}: {e}")
            return False
            
    finally:
        # Always restore the original branch and unstash changes
        try:
            # Switch back to original branch
            subprocess.run(
                ["git", "checkout", current_branch],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Unstash changes if we stashed them
            if has_changes:
                subprocess.run(
                    ["git", "stash", "pop"],
                    check=True,
                    capture_output=True,
                    text=True
                )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to restore original state: {e}")
            # Don't raise here as we're in a finally block
    
    return False
