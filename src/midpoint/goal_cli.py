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
        from .goal_solver import handle_solve_command
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


def analyze_goal(goal_id, human_mode=False):
    """Analyze a goal to determine next actions using the goal analyzer agent."""
    # Ensure goal directory exists
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal {goal_id} not found")
        return False
    
    # ... existing code ...


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


