"""
Goal solving functionality for Midpoint Goal Management System.
This module contains the functions for automatic solving of goals, broken down into
smaller components for better maintainability.
"""

import os
import json
import logging
import datetime
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Local imports
from .agents.models import Goal, SubgoalPlan, TaskContext, ExecutionResult, MemoryState, State
from .agents.goal_decomposer import decompose_goal as agent_decompose_goal
from .agents.goal_analyzer import analyze_goal as agent_analyze_goal
from .agents.task_executor import TaskExecutor
from .goal_git import get_current_hash, get_current_branch
from .goal_state import (
    ensure_goal_dir,
    create_new_task,
    mark_goal_complete,
    merge_subgoal,
    update_git_state
)
from .goal_execute_command import execute_task
from .goal_commands import handle_update_parent_command


def setup_solve_environment(goal_id: str) -> Dict[str, Any]:
    """
    Set up the environment for solving a goal.
    
    Args:
        goal_id: ID of the goal to solve
        
    Returns:
        Dictionary with setup results including:
        - success: Boolean indicating success or failure
        - goal_data: The loaded goal data
        - goal_file: Path to the goal file
        - original_branch: Original Git branch
        - stashed_changes: Boolean indicating if changes were stashed
        - progress_file: Path to progress log file
        - log_progress: Function to log progress
    """
    result = {
        "success": False,
        "goal_data": None,
        "goal_file": None,
        "original_branch": None,
        "stashed_changes": False,
        "progress_file": None,
        "log_progress": None
    }
    
    # Ensure goal directory exists
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    result["goal_file"] = goal_file
    
    if not goal_file.exists():
        logging.error(f"Goal {goal_id} not found")
        return result
    
    try:
        # Load goal data
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
        result["goal_data"] = goal_data
    except Exception as e:
        logging.error(f"Failed to read goal file: {e}")
        return result
    
    # Validate that this is a goal that can be solved (not a task)
    if goal_data.get("is_task", False):
        logging.error(f"{goal_id} is a task, not a goal. Use 'goal execute {goal_id}' instead.")
        return result
    
    # Extract initial state information
    initial_state = goal_data.get("initial_state", {})
    initial_hash = initial_state.get("git_hash")
    if not initial_hash:
        logging.error(f"No initial git hash found in goal {goal_id}")
        return result
    
    # Check if the goal is already complete
    if goal_data.get("is_complete", False):
        logging.warning(f"Goal {goal_id} is already marked as complete.")
        response = input("Do you want to proceed anyway? (y/N): ")
        if response.lower() != 'y':
            logging.info("Solve operation cancelled.")
            return result
        logging.info("Proceeding with solving despite goal being marked complete.")
    
    # Record original branch and check working directory state
    try:
        original_branch = get_current_branch()
        result["original_branch"] = original_branch
        logging.info(f"Original branch: {original_branch}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get original branch: {e}")
        return result
    
    # Check for uncommitted changes and stash them if needed
    stashed_changes = False
    try:
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True
        )
        has_changes = bool(git_status.stdout.strip())
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
        result["stashed_changes"] = stashed_changes
    except subprocess.CalledProcessError as e:
        logging.error(f"Git error while checking/stashing changes: {e}")
        return result
    
    # Create logs directory to store progress information
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create a progress file to track solving steps
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_file = logs_dir / f"solve_{goal_id}_{timestamp}.log"
    result["progress_file"] = progress_file
    
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
    
    result["log_progress"] = log_progress
    result["success"] = True
    return result


def handle_solve_command(args):
    """
    Main entry point for the solve command.
    
    Args:
        args: Command-line arguments including goal_id and flags
        
    Returns:
        Boolean indicating success or failure
    """
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
    
    # Set up the solve environment
    setup_result = setup_solve_environment(goal_id)
    if not setup_result["success"]:
        return False
    
    goal_data = setup_result["goal_data"]
    goal_file = setup_result["goal_file"]
    original_branch = setup_result["original_branch"]
    stashed_changes = setup_result["stashed_changes"]
    progress_file = setup_result["progress_file"]
    log_progress = setup_result["log_progress"]
    
    # Initialize the solving branch
    init_result = initialize_solving_branch(
        goal_id, 
        goal_data, 
        goal_file, 
        original_branch, 
        stashed_changes, 
        log_progress
    )
    
    if not init_result["success"]:
        return False
    
    solve_branch = init_result["solve_branch"]
    
    # Run the solving loop
    solve_result = run_solving_loop(
        goal_id,
        goal_data,
        goal_file,
        debug,
        quiet,
        bypass_validation,
        solve_branch,
        log_progress
    )
    
    # Cleanup process
    cleanup_solve_process(
        original_branch,
        stashed_changes,
        solve_branch,
        solve_result["success"],
        progress_file,
        log_progress,
        debug
    )
    
    return solve_result["success"]


def initialize_solving_branch(
    goal_id: str,
    goal_data: Dict[str, Any],
    goal_file: Path,
    original_branch: str,
    stashed_changes: bool,
    log_progress: callable
) -> Dict[str, Any]:
    """
    Initialize the solving branch for a goal.
    
    Args:
        goal_id: ID of the goal to solve
        goal_data: Loaded goal data
        goal_file: Path to the goal file
        original_branch: Original Git branch
        stashed_changes: Whether changes were stashed
        log_progress: Function to log progress
        
    Returns:
        Dictionary with initialization results including:
        - success: Boolean indicating success or failure
        - solve_branch: Name of the created solving branch
    """
    result = {
        "success": False,
        "solve_branch": None
    }
    
    # Extract initial state information
    initial_state = goal_data.get("initial_state", {})
    initial_hash = initial_state.get("git_hash")
    
    # Create a timestamp for the new branch
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    solve_branch = f"solve-{goal_id}-{timestamp}"
    result["solve_branch"] = solve_branch
    
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
        result["success"] = True
        return result
        
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
        return result


def handle_goal_action(
    action_type: str,
    context: TaskContext,
    current_goal_id: str,
    goal_path: Path,
    analysis_result: Dict[str, Any],
    debug: bool, 
    quiet: bool, 
    bypass_validation: bool,
    memory_repo_path: Optional[str],
    log_progress: callable
) -> Dict[str, Any]:
    """
    Handle a specific goal action based on the analysis result.
    
    Args:
        action_type: Type of action to perform ("decompose", "execute", etc.)
        context: TaskContext object with current execution context
        current_goal_id: Current goal ID being processed
        goal_path: Path to the goal directory
        analysis_result: Analysis result from the goal analyzer
        debug: Enable debug logging
        quiet: Minimize logging output
        bypass_validation: Skip repository validation checks
        memory_repo_path: Path to memory repository
        log_progress: Function to log progress
        
    Returns:
        Dictionary with action results including:
        - success: Boolean indicating success or failure
        - status: Status string ('continue', 'break', 'complete')
        - current_goal_id: New current goal ID (might be changed by action)
        - context: Updated context object
    """
    result = {
        "success": False,
        "status": "continue",
        "current_goal_id": current_goal_id,
        "context": context
    }
    
    if action_type == "decompose":
        # DECOMPOSE: Break down the goal into smaller subgoals
        return handle_decompose_action(
            context, current_goal_id, goal_path, debug, quiet, bypass_validation, 
            memory_repo_path, log_progress
        )
    
    elif action_type == "execute":
        # EXECUTE: Directly execute a task
        return handle_execute_action(
            context, current_goal_id, goal_path, debug, quiet, bypass_validation, 
            memory_repo_path, log_progress
        )
    
    elif action_type == "complete" or action_type == "validate":
        # COMPLETE/VALIDATE: Goal is considered complete
        return handle_complete_action(
            context, current_goal_id, goal_path, debug, log_progress
        )
    
    elif action_type == "give_up":
        # GIVE_UP: Goal has been determined to be unachievable
        return handle_give_up_action(
            context, current_goal_id, goal_path, analysis_result, log_progress
        )
        
    else:
        # Unknown action type
        log_progress(f"WARNING: Unknown action type: {action_type}")
        result["status"] = "break"
        return result


def cleanup_solve_process(
    original_branch: str,
    stashed_changes: bool,
    solve_branch: str,
    success: bool,
    progress_file: Path,
    log_progress: callable,
    debug: bool
) -> None:
    """
    Clean up after the solve process completes.
    
    Args:
        original_branch: Original git branch to return to
        stashed_changes: Whether there were stashed changes to restore
        solve_branch: Name of the solving branch
        success: Whether the solving process was successful
        progress_file: Path to the progress log file
        log_progress: Function to log progress
        debug: Whether debug mode is enabled
    """
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
            logging.info(f"Goal solved successfully!")
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


def run_solving_loop(
    goal_id: str,
    goal_data: Dict[str, Any],
    goal_file: Path,
    debug: bool,
    quiet: bool,
    bypass_validation: bool,
    solve_branch: str,
    log_progress: callable
) -> Dict[str, Any]:
    """
    Run the main solving loop for a goal.
    
    Args:
        goal_id: ID of the goal to solve
        goal_data: Loaded goal data
        goal_file: Path to the goal file
        debug: Enable debug logging
        quiet: Minimize logging output
        bypass_validation: Skip repository validation checks
        solve_branch: Name of the solving branch
        log_progress: Function to log progress
        
    Returns:
        Dictionary with solving results including:
        - success: Boolean indicating success or failure
    """
    result = {
        "success": False
    }
    
    # Extract current state information
    current_state = goal_data.get("current_state", {})
    memory_repo_path = current_state.get("memory_repository_path")
    memory_hash = current_state.get("memory_hash")
    
    # Get the goal directory
    goal_path = goal_file.parent
    
    # Set up solving loop variables
    max_iterations = 15  # Safety limit to prevent infinite loops
    current_iteration = 0
    current_goal_id = goal_id
    
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
            
            # Load failed attempts history if it exists
            current_goal_file = goal_path / f"{current_goal_id}.json"
            if current_goal_file.exists():
                with open(current_goal_file, 'r') as f:
                    loaded_goal_data = json.load(f)
                if "failed_attempts_history" in loaded_goal_data:
                    context.metadata["failed_attempts_history"] = loaded_goal_data["failed_attempts_history"]
            
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
                recommended_action_type = ""
                if "action_type" in analysis_result:
                    recommended_action_type = analysis_result.get("action_type", "")
                elif "action" in analysis_result:
                    recommended_action_type = analysis_result.get("action", "")
                elif "suggested_action" in analysis_result:
                    recommended_action_type = analysis_result.get("suggested_action", "")
                elif loaded_goal_data.get("last_analysis", {}).get("suggested_action"):
                    # Fallback to the last analysis if available
                    recommended_action_type = loaded_goal_data.get("last_analysis", {}).get("suggested_action", "")
                
                recommended_action = analysis_result.get("recommended_action", "")
                if not recommended_action:
                    recommended_action = analysis_result.get("justification", "")
                
                log_progress(f"Analysis recommends: {recommended_action_type} - {recommended_action}")
                
                # Take appropriate action based on the analysis
                action_result = handle_goal_action(
                    recommended_action_type,
                    context,
                    current_goal_id,
                    goal_path,
                    analysis_result,
                    debug,
                    quiet,
                    bypass_validation,
                    memory_repo_path,
                    log_progress
                )
                
                if action_result["status"] == "break":
                    break
                elif action_result["status"] == "complete":
                    result["success"] = True
                    break
                
                # Update context and current goal ID from action result
                context = action_result["context"]
                current_goal_id = action_result["current_goal_id"]
                
                # Update state for next iteration
                state.git_hash = get_current_hash()
                context.state = state
                
            except Exception as e:
                log_progress(f"ERROR: Analysis failed: {e}")
                logging.error(f"Analysis failed: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                break
            
            # Check if the goal is now complete
            if context.goal.metadata.get("is_complete", False):
                log_progress(f"Goal {current_goal_id} is now complete")
                result["success"] = True
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
    
    return result


def handle_decompose_action(
    context: TaskContext,
    current_goal_id: str,
    goal_path: Path,
    debug: bool,
    quiet: bool,
    bypass_validation: bool,
    memory_repo_path: Optional[str],
    log_progress: callable
) -> Dict[str, Any]:
    """
    Handle the decompose action for a goal.
    
    Args:
        context: TaskContext object with current execution context
        current_goal_id: Current goal ID being processed
        goal_path: Path to the goal directory
        debug: Enable debug logging
        quiet: Minimize logging output
        bypass_validation: Skip repository validation checks
        memory_repo_path: Path to memory repository
        log_progress: Function to log progress
        
    Returns:
        Dictionary with action results
    """
    result = {
        "success": False,
        "status": "continue",
        "current_goal_id": current_goal_id,
        "context": context
    }
    
    log_progress(f"Decomposing goal {current_goal_id}")
    try:
        # Get the path to the current goal's JSON file
        goal_file_path = goal_path / f"{current_goal_id}.json"
        if not goal_file_path.exists():
            log_progress(f"ERROR: Goal file {goal_file_path} not found for decomposition.")
            result["status"] = "break"
            return result
            
        decompose_result = agent_decompose_goal(
            repo_path=os.getcwd(),
            goal=context.goal.description, 
            # Pass the goal file path as input_file
            input_file=str(goal_file_path),
            goal_id=current_goal_id,
            memory_hash=context.state.memory_hash,
            memory_repo_path=context.state.memory_repository_path,
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
                result["current_goal_id"] = new_subgoal_id
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
                    result["context"] = context
                    result["success"] = True
            else:
                log_progress("WARNING: Decomposition succeeded but no next subgoal was provided")
        else:
            log_progress(f"ERROR: Decomposition failed: {decompose_result.get('error', 'Unknown error')}")
            result["status"] = "break"
            return result
            
        # Update Git state and goal files after decomposition
        update_git_state(current_goal_id)
        return result
        
    except Exception as e:
        log_progress(f"ERROR: Decomposition failed with exception: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        result["status"] = "break"
        return result


def handle_execute_action(
    context: TaskContext,
    current_goal_id: str,
    goal_path: Path,
    debug: bool,
    quiet: bool,
    bypass_validation: bool,
    memory_repo_path: Optional[str],
    log_progress: callable
) -> Dict[str, Any]:
    """
    Handle the execute action for a goal or task.
    
    Args:
        context: TaskContext object with current execution context
        current_goal_id: Current goal ID being processed
        goal_path: Path to the goal directory
        debug: Enable debug logging
        quiet: Minimize logging output
        bypass_validation: Skip repository validation checks
        memory_repo_path: Path to memory repository
        log_progress: Function to log progress
        
    Returns:
        Dictionary with action results
    """
    result = {
        "success": False,
        "status": "continue",
        "current_goal_id": current_goal_id,
        "context": context
    }
    
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
                result["status"] = "break"
                return result
            else:
                log_progress(f"Task {current_goal_id} executed successfully")
                
            # Update the parent goal with the task execution result
            parent_goal_id = context.goal.metadata["parent_goal"]
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
                        result["current_goal_id"] = parent_goal_id
                        
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
                        result["context"] = context
                        result["success"] = True
                        
                    except Exception as e:
                        log_progress(f"ERROR: Failed to update parent goal state: {e}")
                        result["status"] = "break"
                        return result
            else:
                # This was a top-level task, mark it complete
                log_progress(f"Marking top-level task {current_goal_id} as complete")
                mark_goal_complete(current_goal_id)
                result["success"] = True
                result["status"] = "complete"
                return result
        except Exception as e:
            log_progress(f"ERROR: Task execution failed with exception: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            result["status"] = "break"
            return result
    else:
        # This is not a task yet, convert it to a task first
        log_progress(f"Converting goal {current_goal_id} to a task")
        try:
            parent_id = context.goal.metadata["parent_goal"]
            if not parent_id:
                log_progress(f"ERROR: Goal {current_goal_id} has no parent to create task under")
                result["status"] = "break"
                return result
                
            task_id = create_new_task(parent_id, context.goal.description)
            if task_id:
                log_progress(f"Created task {task_id}")
                result["current_goal_id"] = task_id
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
                    result["context"] = context
                    result["success"] = True
                    return result
            else:
                log_progress("ERROR: Failed to create task")
                result["status"] = "break"
                return result
        except Exception as e:
            log_progress(f"ERROR: Failed to create task: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            result["status"] = "break"
            return result
    
    return result


def handle_complete_action(
    context: TaskContext,
    current_goal_id: str,
    goal_path: Path,
    debug: bool,
    log_progress: callable
) -> Dict[str, Any]:
    """
    Handle the complete/validate action for a goal.
    
    Args:
        context: TaskContext object with current execution context
        current_goal_id: Current goal ID being processed
        goal_path: Path to the goal directory
        debug: Enable debug logging
        log_progress: Function to log progress
        
    Returns:
        Dictionary with action results
    """
    result = {
        "success": False,
        "status": "continue",
        "current_goal_id": current_goal_id,
        "context": context
    }
    
    log_progress(f"Marking goal {current_goal_id} as complete")
    try:
        mark_goal_complete(current_goal_id)
        
        # Check if this is a subgoal and needs to be merged to parent
        parent_goal_id = context.goal.metadata["parent_goal"]
        if parent_goal_id:
            # Try to merge the subgoal to parent
            log_progress(f"Merging subgoal {current_goal_id} to parent {parent_goal_id}")
            merge_result = merge_subgoal(current_goal_id)
            if merge_result:
                # Move up to parent goal
                log_progress(f"Successfully merged. Moving up to parent goal {parent_goal_id}")
                result["current_goal_id"] = parent_goal_id
                
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
                    result["context"] = context
                    result["success"] = True
                    return result
            else:
                log_progress(f"WARNING: Failed to merge subgoal {current_goal_id} to parent {parent_goal_id}")
                result["status"] = "break"
                return result
        else:
            # This was a top-level goal, we're done
            log_progress(f"Top-level goal {current_goal_id} completed")
            result["success"] = True
            result["status"] = "complete"
            return result
    except Exception as e:
        log_progress(f"ERROR: Goal completion failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        result["status"] = "break"
        return result


def handle_give_up_action(
    context: TaskContext,
    current_goal_id: str,
    goal_path: Path,
    analysis_result: Dict[str, Any],
    log_progress: callable
) -> Dict[str, Any]:
    """
    Handle the give_up action for a goal.
    
    Args:
        context: TaskContext object with current execution context
        current_goal_id: Current goal ID being processed
        goal_path: Path to the goal directory
        analysis_result: Analysis result from the goal analyzer
        log_progress: Function to log progress
        
    Returns:
        Dictionary with action results
    """
    result = {
        "success": False,
        "status": "continue",
        "current_goal_id": current_goal_id,
        "context": context
    }
    
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
            if "last_analysis" not in current_goal_data: 
                current_goal_data["last_analysis"] = {}
            # Ensure the justification from the analysis result is stored
            current_goal_data["last_analysis"]["suggested_action"] = "give_up"
            current_goal_data["last_analysis"]["justification"] = failure_justification
            current_goal_data["last_analysis"]["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with open(current_goal_file_path, 'w') as f:
                json.dump(current_goal_data, f, indent=2)
            log_progress(f"Marked goal {current_goal_id} status as 'given_up' in its file.")
        except Exception as e:
            log_progress(f"ERROR: Failed to update current goal file {current_goal_id} with give_up status: {e}")
            result["status"] = "break"
            return result
    else:
        log_progress(f"WARNING: Current goal file {current_goal_id}.json not found. Cannot mark as given_up.")
        result["status"] = "break"
        return result

    # Now, propagate failure to parent using the dedicated command logic
    if parent_goal_id:
        log_progress(f"Calling update-parent command logic to propagate failure to parent {parent_goal_id}")
        # Create a mock args object for the handler function
        update_args = argparse.Namespace(
            command='update-parent', 
            child_id=current_goal_id, 
            outcome='failed' # Hardcode outcome as failed for give_up propagation
        )
        update_success = handle_update_parent_command(update_args)
        
        if update_success:
            log_progress(f"Failure propagation to parent {parent_goal_id} initiated successfully.")
            # Set up next iteration to analyze the parent
            result["current_goal_id"] = parent_goal_id
            goal_file = goal_path / f"{parent_goal_id}.json"
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
                result["context"] = context
                result["success"] = True
                return result
            else:
                log_progress(f"ERROR: Parent goal file {parent_goal_id}.json not found after propagation attempt. Stopping.")
                result["status"] = "break"
                return result
        else:
            log_progress(f"ERROR: Failure propagation call failed for parent {parent_goal_id}. Stopping.")
            result["status"] = "break"
            return result
    else:
        log_progress(f"Top-level goal {current_goal_id} failed (given up). Stopping solve process.")
        result["status"] = "break"
        return result
