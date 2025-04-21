"""
Command-line interface for goal management.
"""

import os
import json
import argparse
import logging
import datetime
import subprocess
import asyncio
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Set, Callable
import sys
import time
import tempfile
import shutil

# Local imports (ensure correct relative paths)
from .agents.models import Goal, SubgoalPlan, TaskContext, ExecutionResult, MemoryState, State
from .agents.goal_decomposer import decompose_goal as agent_decompose_goal
from .agents.task_executor import TaskExecutor, configure_logging as configure_executor_logging
from .agents.tools.git_tools import get_current_hash, get_current_branch, get_repository_diff
from .agents.tools.memory_tools import get_memory_diff

# Import validator for automated validation
from .agents.goal_validator import GoalValidator

# Import the new Goal Analyzer agent function
from .agents.goal_analyzer import analyze_goal as agent_analyze_goal

# Import the new parent update functions
from .goal_operations.goal_update import propagate_success_state_to_parent, propagate_failure_history_to_parent

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Constants
GOAL_DIR = ".goal"
VISUALIZATION_DIR = f"{GOAL_DIR}/visualization"

# Add import for the new goal_git module at the top (after standard imports):
from .goal_git import (
    get_current_hash,
    get_current_branch,
    get_goal_id_from_branch,
    find_branch_for_goal,
    get_recent_commits,
    go_back_commits,
    reset_to_commit,
    run_diff_command,
    find_top_level_branch
)

# Add import for the new goal_execute_command module at the top (after standard imports):
from .goal_execute_command import execute_task
from .goal_decompose_command import decompose_existing_goal
from .goal_file_management import generate_goal_id

# Import from the new goal_state and goal_visualization modules
from .goal_state import (
    ensure_goal_dir,
    create_goal_file,
    create_new_goal,
    create_new_subgoal,
    create_new_task,
    mark_goal_complete,
    merge_subgoal,
    get_child_tasks,
    update_parent_goal_state,
    update_git_state
)

from .goal_visualization import (
    ensure_visualization_dir,
    show_goal_status,
    show_goal_tree,
    show_goal_history,
    generate_graph,
    show_goal_diffs
)

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


def main_command(args):
    """Async entry point for CLI commands."""
    # Handle async commands
    if args.command == "decompose":
        return decompose_existing_goal(args.goal_id, args.debug, args.quiet, args.bypass_validation)
    elif args.command == "execute":
        return execute_task(args.task_id, args.debug, args.quiet, args.bypass_validation, args.no_commit, args.memory_repo)
    elif args.command == "solve":
        return handle_solve_command(args)
    elif args.command == "validate":
        from .validation import handle_validate_goal
        return handle_validate_goal(args.goal_id, args.debug, args.quiet, args.auto)
    
    # All other commands are synchronous, so just call them directly
    if args.command == "new":
        return create_new_goal(args.description)
    elif args.command == "sub":
        return create_new_subgoal(args.parent_id, args.description)
    elif args.command == "task":
        return create_new_task(args.parent_id, args.description)
    elif args.command == "list":
        return list_goals()
    elif args.command == "delete":
        return delete_goal(args.goal_id)
    elif args.command == "back":
        return go_back_commits(args.steps)
    elif args.command == "reset":
        return reset_to_commit(args.commit_id)
    elif args.command == "subs":
        return list_subgoals()
    elif args.command == "complete":
        return mark_goal_complete()
    elif args.command == "status":
        return show_goal_status()
    elif args.command == "tree":
        return show_goal_tree()
    elif args.command == "history":
        return show_goal_history()
    elif args.command == "graph":
        return generate_graph()
    elif args.command == "update-parent":
        return handle_update_parent_command(args)
    elif args.command == "validate-history":
        return show_validation_history(args.goal_id, args.debug, args.quiet)
    elif args.command == "analyze":
        return analyze_goal(args.goal_id, args.human)
    elif args.command == "diff":
        # Use the show_code and show_memory flags if available, otherwise use defaults
        show_code = getattr(args, 'code', True) or getattr(args, 'complete', False) or not (getattr(args, 'memory', False) or getattr(args, 'complete', False))
        show_memory = getattr(args, 'memory', False) or getattr(args, 'complete', False)
        return show_goal_diffs(args.goal_id, show_code=show_code, show_memory=show_memory)
    elif args.command == "revert":
        return revert_goal(args.goal_id)
    else:
        return None


def handle_solve_command(args):
    """
    Automate the process of analyzing, decomposing, and executing tasks defined
    in the .goal/ directory.
    
    Args:
        args: Command-line arguments including goal_id and optional flags
    """
    # Import necessary agent functions at the beginning
    from .agents.goal_analyzer import analyze_goal as agent_analyze_goal
    from .agents.goal_decomposer import decompose_goal as agent_decompose_goal
    from .agents.task_executor import TaskExecutor, configure_logging as configure_executor_logging
    
    # Configure logging based on args
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)
    
    goal_id = args.goal_id
    debug = args.debug
    quiet = args.quiet
    bypass_validation = args.bypass_validation
    
    logging.info(f"Starting automated solving for goal {goal_id}")
    
    # ===== Initial Setup =====
    # Ensure goal directory exists
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
    
    # Validate that this is a goal that can be solved (not a task)
    if goal_data.get("is_task", False):
        logging.error(f"{goal_id} is a task, not a goal. Use 'goal execute {goal_id}' instead.")
        return False
    
    # Extract initial and current state information
    initial_state = goal_data.get("initial_state", {})
    current_state = goal_data.get("current_state", {})
    
    initial_hash = initial_state.get("git_hash")
    if not initial_hash:
        logging.error(f"No initial git hash found in goal {goal_id}")
        return False
    
    # Get memory repository information
    memory_repo_path = current_state.get("memory_repository_path")
    memory_hash = current_state.get("memory_hash")
    
    # Check if the goal is already complete
    if goal_data.get("is_complete", False):
        logging.warning(f"Goal {goal_id} is already marked as complete.")
        response = input("Do you want to proceed anyway? (y/N): ")
        if response.lower() != 'y':
            logging.info("Solve operation cancelled.")
            return False
        logging.info("Proceeding with solving despite goal being marked complete.")
    
    # Record original branch and check working directory state
    try:
        original_branch = get_current_branch()
        logging.info(f"Original branch: {original_branch}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get original branch: {e}")
        return False
    
    # Check for uncommitted changes and stash them if needed
    stashed_changes = False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True
        )
        has_changes = bool(result.stdout.strip())
        if has_changes:
            logging.info("Uncommitted changes detected. Stashing before proceeding.")
            subprocess.run(
                ["git", "stash", "push", "-m", f"Stashing changes before solving goal {goal_id}"],
                check=True,
                capture_output=True,
                text=True
            )
            stashed_changes = True
            logging.info("Changes stashed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git error while checking/stashing changes: {e}")
        return False
    
    # Create logs directory to store progress information
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create a progress file to track solving steps
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_file = logs_dir / f"solve_{goal_id}_{timestamp}.log"
    
    with open(progress_file, 'w') as f:
        f.write(f"Solve Progress Log for Goal {goal_id}\n")
        f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Description: {goal_data.get('description', 'No description')}\n")
        f.write("=" * 80 + "\n\n")
    
    def log_progress(message):
        """Helper function to log progress both to console and file"""
        logging.info(message)
        with open(progress_file, 'a') as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    # ===== Git Checkout =====
    # Create a timestamp for the new branch
    solve_branch = f"solve-{goal_id}-{timestamp}"
    
    try:
        # First checkout the initial commit hash
        log_progress(f"Checking out commit {initial_hash[:8]}...")
        subprocess.run(
            ["git", "checkout", initial_hash],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Create a new solve branch from this commit
        log_progress(f"Creating new branch {solve_branch}...")
        subprocess.run(
            ["git", "checkout", "-b", solve_branch],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Update the goal's branch_name property to include this solve branch
        if "branch_name" not in goal_data:
            goal_data["branch_name"] = solve_branch
        
        # Update the solve branch in the goal file
        with open(goal_file, 'w') as f:
            json.dump(goal_data, f, indent=2)
        
        log_progress(f"Created and switched to branch {solve_branch}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git error during checkout: {e}")
        log_progress(f"ERROR: Git error during checkout: {e}")
        # Cleanup and revert to original branch
        try:
            subprocess.run(
                ["git", "checkout", original_branch],
                check=True,
                capture_output=True,
                text=True
            )
            if stashed_changes:
                subprocess.run(
                    ["git", "stash", "pop"],
                    check=True,
                    capture_output=True,
                    text=True
                )
        except subprocess.CalledProcessError as cleanup_error:
            logging.error(f"Error during cleanup after failed checkout: {cleanup_error}")
        return False
    
    # ===== Solving Loop =====
    max_iterations = 15  # Safety limit to prevent infinite loops
    current_iteration = 0
    current_goal_id = goal_id
    success = False
    
    try:
        # Create goal object with current properties
        goal = Goal(
            id=goal_id,
            description=goal_data.get("description", ""),
            metadata={
                "parent_goal": goal_data.get("parent_goal", ""),
                "is_complete": goal_data.get("is_complete", False)
            }
        )
        
        # Create state object for the analysis context
        state = State(
            git_hash=get_current_hash(),
            repository_path=os.getcwd(),
            branch_name=solve_branch,
            memory_hash=memory_hash,
            memory_repository_path=memory_repo_path,
            description=f"Current state of goal {goal_id} during auto-solve process"
        )
        
        # Create memory state object
        memory_state = MemoryState(
            memory_hash=memory_hash if memory_hash else "",
            repository_path=memory_repo_path if memory_repo_path else ""
        )
        
        # Initialize TaskContext for agent calls
        context = TaskContext(
            goal=goal,
            state=state,
            memory_state=memory_state,
            metadata={
                "goal_id": goal_id,
                "original_branch": original_branch,
                "solving_branch": solve_branch,
                "debug": debug,
                "quiet": quiet,
                "bypass_validation": bypass_validation
            }
        )
        
        log_progress(f"Starting solving loop for goal {goal_id}")
        log_progress(f"Initial state: {state.git_hash[:8]} on branch {solve_branch}")
        
        while current_iteration < max_iterations:
            current_iteration += 1
            log_progress(f"ITERATION {current_iteration}: Processing goal {current_goal_id}")
            
            # ===== START EDIT 1: Load failed attempts history =====
            # Update context with failed attempts history if it exists in the current goal file
            if goal_file.exists():
                with open(goal_file, 'r') as f:
                    loaded_goal_data = json.load(f)
                if "failed_attempts_history" in loaded_goal_data:
                    context.metadata["failed_attempts_history"] = loaded_goal_data["failed_attempts_history"]
            # ===== END EDIT 1 =====
            
            # ANALYZE: Determine what to do next with the current goal
            log_progress(f"Analyzing goal {current_goal_id}")
            try:
                analysis_result = agent_analyze_goal(
                    repo_path=os.getcwd(),
                    goal=goal.description,
                    goal_id=current_goal_id,
                    memory_hash=state.memory_hash,
                    memory_repo_path=state.memory_repository_path,
                    debug=debug,
                    quiet=quiet,
                    bypass_validation=bypass_validation
                )
                
                # Extract the recommended action from the analysis result
                if "action_type" in analysis_result:
                    recommended_action_type = analysis_result.get("action_type", "")
                elif "action" in analysis_result:
                    recommended_action_type = analysis_result.get("action", "")
                elif "suggested_action" in analysis_result:
                    recommended_action_type = analysis_result.get("suggested_action", "")
                elif goal_data.get("last_analysis", {}).get("suggested_action"):
                    # Fallback to the last analysis if available
                    recommended_action_type = goal_data.get("last_analysis", {}).get("suggested_action", "")
                else:
                    recommended_action_type = ""
                
                recommended_action = analysis_result.get("recommended_action", "")
                if not recommended_action:
                    recommended_action = analysis_result.get("justification", "")
                
                log_progress(f"Analysis recommends: {recommended_action_type} - {recommended_action}")
            except Exception as e:
                log_progress(f"ERROR: Analysis failed: {e}")
                logging.error(f"Analysis failed: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                break
            
            # Take appropriate action based on the analysis
            if recommended_action_type == "decompose":
                # DECOMPOSE: Break down the goal into smaller subgoals
                log_progress(f"Decomposing goal {current_goal_id}")
                try:
                    # Get the path to the current goal's JSON file
                    goal_file_path = goal_path / f"{current_goal_id}.json"
                    if not goal_file_path.exists():
                        log_progress(f"ERROR: Goal file {goal_file_path} not found for decomposition.")
                        break # Stop if file doesn't exist
                        
                    decompose_result = agent_decompose_goal(
                        repo_path=os.getcwd(),
                        goal=goal.description, 
                        # Pass the goal file path as input_file
                        input_file=str(goal_file_path),
                        goal_id=current_goal_id,
                        memory_hash=state.memory_hash,
                        memory_repo_path=state.memory_repository_path,
                        debug=debug,
                        quiet=quiet,
                        bypass_validation=bypass_validation
                    )
                    
                    # Handle decomposition result
                    if decompose_result.get("success", False):
                        # Get the new subgoal to work on
                        new_subgoal_id = decompose_result.get("next_goal_id", "")
                        if new_subgoal_id:
                            log_progress(f"Decomposition created subgoal {new_subgoal_id}")
                            current_goal_id = new_subgoal_id
                            # Update context for next iteration with the new subgoal
                            subgoal_file = goal_path / f"{new_subgoal_id}.json"
                            if subgoal_file.exists():
                                with open(subgoal_file, 'r') as f:
                                    subgoal_data = json.load(f)
                                
                                # Create updated goal object
                                goal = Goal(
                                    id=new_subgoal_id,
                                    description=subgoal_data.get("description", ""),
                                    metadata={
                                        "parent_goal": subgoal_data.get("parent_goal", ""),
                                        "is_complete": subgoal_data.get("is_complete", False)
                                    }
                                )
                                context.goal = goal
                                context.metadata["goal_id"] = new_subgoal_id
                        else:
                            log_progress("WARNING: Decomposition succeeded but no next subgoal was provided")
                    else:
                        log_progress(f"ERROR: Decomposition failed: {decompose_result.get('error', 'Unknown error')}")
                        break
                        
                    # Update Git state and goal files after decomposition
                    update_git_state(current_goal_id)
                except Exception as e:
                    log_progress(f"ERROR: Decomposition failed with exception: {e}")
                    if debug:
                        import traceback
                        traceback.print_exc()
                    break
                
            elif recommended_action_type == "execute":
                # EXECUTE: Directly execute a task
                if current_goal_id.startswith("T"):
                    # This is already a task, execute it
                    log_progress(f"Executing task {current_goal_id}")
                    try:
                        execution_result = execute_task(
                            current_goal_id, 
                            debug=debug, 
                            quiet=quiet, 
                            bypass_validation=bypass_validation,
                            no_commit=False,  # Allow commits
                            memory_repo=memory_repo_path
                        )
                        
                        # Handle execution result
                        if not execution_result or not execution_result.get("success", False):
                            log_progress(f"ERROR: Task execution failed: {execution_result.get('error', 'Unknown error')}")
                            break
                        else:
                            log_progress(f"Task {current_goal_id} executed successfully")
                            
                        # Update the parent goal with the task execution result
                        parent_goal_id = goal.metadata["parent_goal"]
                        if parent_goal_id:
                            # Update parent goal state
                            parent_file = goal_path / f"{parent_goal_id}.json"
                            if parent_file.exists():
                                try:
                                    with open(parent_file, 'r') as f:
                                        parent_data = json.load(f)
                                    
                                    # Update parent's current state
                                    parent_data["current_state"]["git_hash"] = get_current_hash()
                                    parent_data["current_state"]["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    
                                    with open(parent_file, 'w') as f:
                                        json.dump(parent_data, f, indent=2)
                                        
                                    log_progress(f"Updated parent goal {parent_goal_id} state")
                                    
                                    # Switch to parent goal for next iteration
                                    current_goal_id = parent_goal_id
                                    
                                    # Update context for next iteration
                                    goal = Goal(
                                        id=parent_goal_id,
                                        description=parent_data.get("description", ""),
                                        metadata={
                                            "parent_goal": parent_data.get("parent_goal", ""),
                                            "is_complete": parent_data.get("is_complete", False)
                                        }
                                    )
                                    context.goal = goal
                                    context.metadata["goal_id"] = parent_goal_id
                                    
                                except Exception as e:
                                    log_progress(f"ERROR: Failed to update parent goal state: {e}")
                        else:
                            # This was a top-level task, mark it complete
                            log_progress(f"Marking top-level task {current_goal_id} as complete")
                            mark_goal_complete(current_goal_id)
                            success = True
                            break
                    except Exception as e:
                        log_progress(f"ERROR: Task execution failed with exception: {e}")
                        if debug:
                            import traceback
                            traceback.print_exc()
                        break
                else:
                    # This is not a task yet, convert it to a task first
                    log_progress(f"Converting goal {current_goal_id} to a task")
                    try:
                        parent_id = goal.metadata["parent_goal"]
                        if not parent_id:
                            log_progress(f"ERROR: Goal {current_goal_id} has no parent to create task under")
                            break
                            
                        task_id = create_new_task(parent_id, goal.description)
                        if task_id:
                            log_progress(f"Created task {task_id}")
                            current_goal_id = task_id
                            # Update context for next iteration
                            task_file = goal_path / f"{task_id}.json"
                            if task_file.exists():
                                with open(task_file, 'r') as f:
                                    task_data = json.load(f)
                                
                                # Create updated goal object for the task
                                goal = Goal(
                                    id=task_id,
                                    description=task_data.get("description", ""),
                                    metadata={
                                        "parent_goal": task_data.get("parent_goal", ""),
                                        "is_complete": False
                                    }
                                )
                                context.goal = goal
                                context.metadata["goal_id"] = task_id
                        else:
                            log_progress("ERROR: Failed to create task")
                            break
                    except Exception as e:
                        log_progress(f"ERROR: Failed to create task: {e}")
                        if debug:
                            import traceback
                            traceback.print_exc()
                        break
                
            elif recommended_action_type == "complete":
                # COMPLETE: Goal is considered complete
                log_progress(f"Marking goal {current_goal_id} as complete")
                try:
                    mark_goal_complete(current_goal_id)
                    
                    # Check if this is a subgoal and needs to be merged to parent
                    parent_goal_id = goal.metadata["parent_goal"]
                    if parent_goal_id:
                        # Try to merge the subgoal to parent
                        log_progress(f"Merging subgoal {current_goal_id} to parent {parent_goal_id}")
                        merge_result = merge_subgoal(current_goal_id)
                        if merge_result:
                            # Move up to parent goal
                            log_progress(f"Successfully merged. Moving up to parent goal {parent_goal_id}")
                            current_goal_id = parent_goal_id
                            
                            # Update context for next iteration with parent goal
                            parent_file = goal_path / f"{parent_goal_id}.json"
                            if parent_file.exists():
                                with open(parent_file, 'r') as f:
                                    parent_data = json.load(f)
                                
                                # Create updated goal object for the parent
                                goal = Goal(
                                    id=parent_goal_id,
                                    description=parent_data.get("description", ""),
                                    metadata={
                                        "parent_goal": parent_data.get("parent_goal", ""),
                                        "is_complete": parent_data.get("is_complete", False)
                                    }
                                )
                                context.goal = goal
                                context.metadata["goal_id"] = parent_goal_id
                        else:
                            log_progress(f"WARNING: Failed to merge subgoal {current_goal_id} to parent {parent_goal_id}")
                            break
                    else:
                        # This was a top-level goal, we're done
                        log_progress(f"Top-level goal {current_goal_id} completed")
                        success = True
                        break
                except Exception as e:
                    log_progress(f"ERROR: Goal completion failed: {e}")
                    if debug:
                        import traceback
                        traceback.print_exc()
                    break
            elif recommended_action_type == "validate":
                # VALIDATE: The goal is ready for validation
                log_progress(f"Validating goal {current_goal_id}")
                try:
                    # Here we could call a validation function if needed
                    # For now, we'll just mark the goal as complete
                    mark_goal_complete(current_goal_id)
                    log_progress(f"Goal {current_goal_id} validated and marked as complete")
                    
                    # Check if this is a subgoal and needs to be merged to parent
                    parent_goal_id = goal.metadata["parent_goal"]
                    if parent_goal_id:
                        # Try to merge the subgoal to parent
                        log_progress(f"Merging subgoal {current_goal_id} to parent {parent_goal_id}")
                        merge_result = merge_subgoal(current_goal_id)
                        if merge_result:
                            # Move up to parent goal
                            log_progress(f"Successfully merged. Moving up to parent goal {parent_goal_id}")
                            current_goal_id = parent_goal_id
                            
                            # Update context for next iteration with parent goal
                            parent_file = goal_path / f"{parent_goal_id}.json"
                            if parent_file.exists():
                                with open(parent_file, 'r') as f:
                                    parent_data = json.load(f)
                                
                                # Create updated goal object for the parent
                                goal = Goal(
                                    id=parent_goal_id,
                                    description=parent_data.get("description", ""),
                                    metadata={
                                        "parent_goal": parent_data.get("parent_goal", ""),
                                        "is_complete": parent_data.get("is_complete", False)
                                    }
                                )
                                context.goal = goal
                                context.metadata["goal_id"] = parent_goal_id
                        else:
                            log_progress(f"WARNING: Failed to merge subgoal {current_goal_id} to parent {parent_goal_id}")
                            break
                    else:
                        # This was a top-level goal, we're done
                        log_progress(f"Top-level goal {current_goal_id} completed")
                        success = True
                        break
                except Exception as e:
                    log_progress(f"ERROR: Goal validation failed: {e}")
                    if debug:
                        import traceback
                        traceback.print_exc()
                    break
            # ===== START EDIT 2: Modify give_up handling =====
            elif recommended_action_type == "give_up":
                log_progress(f"Goal {current_goal_id} marked for give up by analyzer.")
                failure_justification = analysis_result.get("justification", "No justification provided.")
                log_progress(f"Reason: {failure_justification}")

                parent_goal_id = None
                # Update the current goal file first to mark it as given up
                current_goal_file_path = goal_path / f"{current_goal_id}.json"
                if current_goal_file_path.exists():
                    try:
                        with open(current_goal_file_path, 'r') as f:
                            current_goal_data = json.load(f)
                        parent_goal_id = current_goal_data.get("parent_goal")
                        
                        current_goal_data["status"] = "given_up"
                        if "last_analysis" not in current_goal_data: current_goal_data["last_analysis"] = {}
                        # Ensure the justification from the analysis result is stored
                        current_goal_data["last_analysis"]["suggested_action"] = "give_up"
                        current_goal_data["last_analysis"]["justification"] = failure_justification
                        current_goal_data["last_analysis"]["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        with open(current_goal_file_path, 'w') as f:
                            json.dump(current_goal_data, f, indent=2)
                        log_progress(f"Marked goal {current_goal_id} status as 'given_up' in its file.")
                    except Exception as e:
                        log_progress(f"ERROR: Failed to update current goal file {current_goal_id} with give_up status: {e}")
                        # Decide if we should break or try to continue propagating?
                        # Let's break for safety for now.
                        break 
                else:
                    log_progress(f"WARNING: Current goal file {current_goal_id}.json not found. Cannot mark as given_up.")
                    # Cannot proceed without the current goal file
                    break

                # Now, propagate failure to parent using the dedicated command logic
                if parent_goal_id:
                    log_progress(f"Calling update-parent command logic to propagate failure to parent {parent_goal_id}")
                    # Create a mock args object for the handler function
                    update_args = argparse.Namespace(
                        command='update-parent', 
                        child_id=current_goal_id, 
                        outcome='failed' # Hardcode outcome as failed for give_up propagation
                        # No reason needed here as the function reads it from child file
                    )
                    update_success = handle_update_parent_command(update_args)
                    
                    if update_success:
                        log_progress(f"Failure propagation to parent {parent_goal_id} initiated successfully.")
                        # Set up next iteration to analyze the parent
                        current_goal_id = parent_goal_id
                        goal_file = goal_path / f"{parent_goal_id}.json" # Update goal_file path for next loop
                        if goal_file.exists():
                            with open(goal_file, 'r') as f:
                                parent_data = json.load(f)
                            goal = Goal(
                                id=parent_goal_id,
                                description=parent_data.get("description", ""),
                                metadata=context.metadata # Keep existing metadata
                            )
                            context.goal = goal
                            context.metadata["goal_id"] = parent_goal_id
                            continue # Continue loop to analyze parent
                        else:
                             log_progress(f"ERROR: Parent goal file {parent_goal_id}.json not found after propagation attempt. Stopping.")
                             break
                    else:
                        log_progress(f"ERROR: Failure propagation call failed for parent {parent_goal_id}. Stopping.")
                        break # Exit loop if propagation command logic fails
                else:
                    log_progress(f"Top-level goal {current_goal_id} failed (given up). Stopping solve process.")
                    success = False # Mark overall process as not successful
                    break # Exit loop for top-level failure
            # ===== END EDIT 2 =====
            else:
                # Unknown action type
                log_progress(f"WARNING: Unknown action type: {recommended_action_type}")
                break
            
            # Update state for next iteration
            state.git_hash = get_current_hash()
            context.state = state
            
            # Check if the goal is now complete
            if goal.metadata.get("is_complete", False):
                log_progress(f"Goal {current_goal_id} is now complete")
                success = True
                break
                
        # Check if we hit the iteration limit
        if current_iteration >= max_iterations:
            log_progress(f"WARNING: Reached maximum iteration limit of {max_iterations}")
            
    except Exception as e:
        log_progress(f"ERROR: Unexpected error during solving loop: {e}")
        logging.error(f"Error during solving loop: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    # ===== Cleanup =====
    try:
        # Return to original branch 
        log_progress(f"Returning to original branch {original_branch}")
        subprocess.run(
            ["git", "checkout", original_branch],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Restore stashed changes if any
        if stashed_changes:
            log_progress("Restoring stashed changes")
            subprocess.run(
                ["git", "stash", "pop"],
                check=True,
                capture_output=True,
                text=True
            )
        
        # Final status
        if success:
            log_progress(f"Goal solving process completed successfully on branch {solve_branch}")
        else:
            log_progress(f"Goal solving process completed with some issues on branch {solve_branch}")
        
        log_progress(f"To continue working with this goal, use: git checkout {solve_branch}")
        log_progress(f"Progress log saved to: {progress_file}")
        
        # Print final message to user
        if success:
            logging.info(f"Goal {goal_id} solved successfully!")
        else:
            logging.warning(f"Goal solving process completed with some issues.")
        
        logging.info(f"Solution branch: {solve_branch}")
        logging.info(f"Progress log: {progress_file}")
    except Exception as e:
        log_progress(f"ERROR: Cleanup failed: {e}")
        logging.error(f"Error during cleanup: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    return success

def main():
    """Main CLI entry point."""
    import asyncio
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Goal branch management commands")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Goal Management Commands
    # -----------------------
    # goal new <description>
    new_parser = subparsers.add_parser("new", help="Create a new top-level goal")
    new_parser.add_argument("description", help="Description of the goal")
    
    # goal delete <goal-id>
    delete_parser = subparsers.add_parser("delete", help="Delete a goal, subgoal, or task")
    delete_parser.add_argument("goal_id", help="ID of the goal to delete")
    
    # goal sub <parent-id> <description>
    sub_parser = subparsers.add_parser("sub", help="Create a subgoal under the specified parent")
    sub_parser.add_argument("parent_id", help="Parent goal ID")
    sub_parser.add_argument("description", help="Description of the subgoal")
    
    # goal task <parent-id> <description>
    task_parser = subparsers.add_parser("task", help="Create a new directly executable task under the specified parent")
    task_parser.add_argument("parent_id", help="Parent goal ID")
    task_parser.add_argument("description", help="Description of the task")
    
    # goal list
    subparsers.add_parser("list", help="List all goals and subgoals in tree format")
    
    # goal decompose <goal-id>
    decompose_parser = subparsers.add_parser("decompose", help="Decompose a goal into subgoals")
    decompose_parser.add_argument("goal_id", help="Goal ID to decompose")
    decompose_parser.add_argument("--debug", action="store_true", help="Show debug output")
    decompose_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    decompose_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    
    # goal execute <task-id>
    execute_parser = subparsers.add_parser("execute", help="Execute a task using the TaskExecutor")
    execute_parser.add_argument("task_id", help="Task ID to execute")
    execute_parser.add_argument("--debug", action="store_true", help="Show debug output")
    execute_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    execute_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    execute_parser.add_argument("--no-commit", action="store_true", help="Prevent automatic commits")
    execute_parser.add_argument("--memory-repo", help="Path to memory repository")
    
    # goal solve <goal-id>
    solve_parser = subparsers.add_parser("solve", help="Automatically analyze, decompose, and execute tasks for a goal")
    solve_parser.add_argument("goal_id", help="Goal ID to solve")
    solve_parser.add_argument("--debug", action="store_true", help="Show debug output")
    solve_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    solve_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    
    # State Navigation Commands
    # ------------------------
    # goal back [steps]
    back_parser = subparsers.add_parser("back", help="Go back N commits on current goal branch")
    back_parser.add_argument("steps", nargs="?", type=int, default=1, help="Number of commits to go back")
    
    # goal reset <commit-id>
    reset_parser = subparsers.add_parser("reset", help="Reset to specific commit on current branch")
    reset_parser.add_argument("commit_id", help="Commit ID to reset to")
    
    # Removed Hierarchy Navigation Commands
    
    # goal subs
    subparsers.add_parser("subs", help="List available subgoals for current goal")
    
    # Result Incorporation Commands
    # ----------------------------
    # goal complete
    subparsers.add_parser("complete", help="Mark current goal as complete")
    
    # goal status
    subparsers.add_parser("status", help="Show completion status of all goals")
    
    # Visualization Tools
    # ------------------
    # goal tree
    subparsers.add_parser("tree", help="Show visual representation of goal hierarchy")
    
    # goal history
    subparsers.add_parser("history", help="Show timeline of goal exploration")
    
    # goal graph
    subparsers.add_parser("graph", help="Generate graphical visualization")
    
    # goal diff <goal-id>
    diff_parser = subparsers.add_parser("diff", help="Show code and memory diffs for a specific goal")
    diff_parser.add_argument("goal_id", help="ID of the goal to show diffs for")
    # Add mutually exclusive group for diff modes
    mode_group = diff_parser.add_mutually_exclusive_group()
    mode_group.add_argument("--code", action="store_true", help="Show only code diff (default)")
    mode_group.add_argument("--memory", action="store_true", help="Show only memory diff")
    mode_group.add_argument("--complete", action="store_true", help="Show both code and memory diffs")
    # Set defaults based on flags
    diff_parser.set_defaults(func=lambda args: show_goal_diffs(
                                 args.goal_id,
                                 show_code=(args.code or args.complete or not (args.memory or args.complete)), # Default to True if no flag set
                                 show_memory=(args.memory or args.complete)
                             ))
    
    # goal revert <goal-id>
    revert_parser = subparsers.add_parser("revert", help="Revert a goal's current state back to its initial state")
    revert_parser.add_argument("goal_id", help="ID of the goal to revert")
    
    # goal validate <goal-id>
    validate_parser = subparsers.add_parser("validate", help="Validate a goal's completion criteria")
    validate_parser.add_argument("goal_id", help="ID of the goal to validate")
    validate_parser.add_argument("--debug", action="store_true", help="Show debug output")
    validate_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    validate_parser.add_argument("--auto", action="store_true", help="Perform automated validation using LLM")
    validate_parser.add_argument("--model", default="gpt-4o-mini", help="Model to use for validation (with --auto)")
    
    # goal validate-history <goal-id>
    validate_history_parser = subparsers.add_parser("validate-history", help="Show validation history for a goal")
    validate_history_parser.add_argument("goal_id", help="ID of the goal to show validation history for")
    validate_history_parser.add_argument("--debug", action="store_true", help="Show debug output")
    validate_history_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    
    # Add new subparser for analyzing a goal
    analyze_parser = subparsers.add_parser("analyze", 
                                   help="Analyze a goal and suggest next steps (decompose, execute, validate, etc.)",
                                   description="""
Intelligent analysis of a goal's current state to determine the best next action.
Analysis considers child goals/tasks status, validation results, and remaining work.
Primarily recommends "decompose" for complex goals that need further breakdown,
or "validate" when enough children have been successfully completed to potentially
satisfy all requirements. Other possible recommendations include "execute" for
simple remaining work, "mark_complete", "update_parent", or "give_up" in special cases.
                                   """)
    analyze_parser.add_argument("goal_id", help="ID of the goal to analyze")
    analyze_parser.add_argument("--human", action="store_true", help="Perform interactive analysis with detailed context")
    
    # Add new subparser for updating parent from child
    update_parent_parser = subparsers.add_parser(
        'update-parent',
        help='Update parent goal state from child goal/task'
    )
    update_parent_parser.add_argument(
        'child_id',
        help='ID of the child goal/task whose outcome is being propagated'
    )
    update_parent_parser.add_argument(
        '--outcome',
        required=True,
        choices=['success', 'failed'],
        help='Outcome of the child ("success" propagates state, "failed" propagates failure history)'
    )
    
    args = parser.parse_args()
    
    # Run the main function
    main_command(args)


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


