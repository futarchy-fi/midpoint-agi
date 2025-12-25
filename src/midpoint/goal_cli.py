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

# ---------------------------------------------------------------------------
# Navigation helpers (kept for backward compatibility + tests)
# ---------------------------------------------------------------------------

def get_parent_goal_id(goal_id: str) -> str | None:
    """
    Return the parent goal id for `goal_id`, or:
    - "" if the goal exists and has no parent
    - None if the goal file does not exist / can't be read

    Note: tests patch `GOAL_DIR`, so we must use it (not hard-code ".goal").
    """
    try:
        goal_file = Path(GOAL_DIR) / f"{goal_id}.json"
        if not goal_file.exists():
            return None
        with open(goal_file, "r") as f:
            data = json.load(f)
        return data.get("parent_goal", "")
    except Exception:
        return None


def go_back_commits(steps: int = 1) -> bool:
    """Go back N commits on current branch (simple wrapper used by tests)."""
    if steps < 1:
        logging.error("Number of steps must be at least 1")
        return False
    try:
        subprocess.run(
            ["git", "reset", "--hard", f"HEAD~{steps}"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to go back commits: {e}")
        return False


def reset_to_commit(commit_id: str) -> bool:
    """Reset to a specific commit (simple wrapper used by tests)."""
    try:
        subprocess.run(
            ["git", "reset", "--hard", commit_id],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to reset to commit: {e}")
        return False


def go_to_parent_goal() -> bool:
    """Checkout the parent goal branch of the current goal branch."""
    try:
        current_branch = get_current_branch()
        goal_id = get_goal_id_from_branch(current_branch) if current_branch else None
        if not goal_id:
            logging.error("Not on a goal branch")
            return False

        parent_id = get_parent_goal_id(goal_id)
        if parent_id is None:
            logging.error(f"Could not determine parent for goal {goal_id}")
            return False
        if parent_id == "":
            logging.info("Already at a root goal (no parent)")
            return False

        parent_branch = find_branch_for_goal(parent_id)
        if not parent_branch:
            logging.error(f"No branch found for parent goal {parent_id}")
            return False

        subprocess.run(
            ["git", "checkout", parent_branch],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to checkout parent goal: {e}")
        return False


def go_to_child(child_goal_id: str) -> bool:
    """Checkout the branch for a child goal/task."""
    try:
        child_branch = find_branch_for_goal(child_goal_id)
        if not child_branch:
            logging.error(f"No branch found for child {child_goal_id}")
            return False
        subprocess.run(
            ["git", "checkout", child_branch],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to checkout child goal: {e}")
        return False


def go_to_root_goal() -> bool:
    """Walk up parents until reaching the root goal, then checkout its branch."""
    current_branch = get_current_branch()
    goal_id = get_goal_id_from_branch(current_branch) if current_branch else None
    if not goal_id:
        logging.error("Not on a goal branch")
        return False

    # Walk up until parent is empty.
    while True:
        parent_id = get_parent_goal_id(goal_id)
        if parent_id is None:
            logging.error(f"Could not determine parent for goal {goal_id}")
            return False
        if parent_id == "":
            break
        goal_id = parent_id

    root_branch = find_branch_for_goal(goal_id)
    if not root_branch:
        logging.error(f"No branch found for root goal {goal_id}")
        return False

    try:
        subprocess.run(
            ["git", "checkout", root_branch],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to checkout root goal: {e}")
        return False


def list_subgoals() -> list[str]:
    """
    List direct child goals/tasks of the current goal.

    Returns list of child goal IDs. Prints a short human-readable list (used by tests).
    """
    current_branch = get_current_branch()
    goal_id = get_goal_id_from_branch(current_branch) if current_branch else None
    if not goal_id:
        print("Not on a goal branch.")
        return []

    goal_path = Path(GOAL_DIR)
    children: list[str] = []
    if not goal_path.exists():
        print(f"No goal directory found at {goal_path}")
        return []

    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            if (data.get("parent_goal") or "") == goal_id:
                child_id = data.get("goal_id")
                if child_id:
                    children.append(child_id)
        except Exception:
            continue

    children.sort()
    print(f"Children of {goal_id}:")
    for cid in children:
        print(f"- {cid}")
    return children

# Import command implementations from their modules
from .goal_file_management import (
    generate_goal_id,
    ensure_goal_dir
)

from .goal_git import (
    get_current_hash,
    get_current_branch,
    get_goal_id_from_branch,
    find_branch_for_goal
)

from .goal_state import (
    ensure_goal_dir,
    create_goal_file,
    create_new_goal,
    mark_goal_complete,
    merge_subgoal
)

from .goal_visualization import (
    show_goal_status,
    show_goal_tree,
    show_goal_diffs
)

# Import commands for goal management
from .goal_decompose_command import decompose_existing_goal
from .goal_execute_command import execute_task
from .goal_revert import revert_goal
from .goal_file_management import delete_goal

def analyze_goal(goal_id, human_mode=False):
    """Analyze a goal to determine next actions using the goal analyzer agent."""
    # Ensure goal directory exists
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal {goal_id} not found")
        return False
    
    # Delegate to a command module (parallel to decompose/execute/validate).
    from .goal_analyze_command import analyze_existing_goal
    return analyze_existing_goal(goal_id, debug=bool(human_mode))

def show_validation_history(goal_id, debug=False, quiet=False):
    """Show validation history for a specific goal."""
    validation_dir = Path("logs/validation_history")
    if not validation_dir.exists():
        if not quiet:
            print(f"No validation history found (directory {validation_dir} does not exist)")
        return False
    
    # Find all validation files for this goal
    history_files = sorted(list(validation_dir.glob(f"{goal_id}_*.json")), 
                          key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not history_files:
        if not quiet:
            print(f"No validation history found for goal {goal_id}")
        return False
    
    print(f"Validation History for Goal {goal_id}:")
    print("═" * 50)
    
    for i, file_path in enumerate(history_files, 1):
        if file_path.name.endswith("_context.json"):
            continue  # Skip context files
            
        try:
            with open(file_path, 'r') as f:
                record = json.load(f)
                
            timestamp = record.get("timestamp", "Unknown")
            score = record.get("score", 0.0)
            validated_by = record.get("validated_by", "Unknown")
            git_hash = record.get("git_hash", "")
            criteria_results = record.get("criteria_results", [])
            passed_count = sum(1 for cr in criteria_results if cr.get("passed", False))
            total_count = len(criteria_results)
            
            print(f"{i}. {timestamp} - Score: {score:.2%} ({passed_count}/{total_count})")
            print(f"   Validated by: {validated_by}")
            if git_hash:
                print(f"   Git hash: {git_hash[:8]}")
            
            if debug:
                print("\n   Criteria Results:")
                for j, cr in enumerate(criteria_results, 1):
                    status = "✅ Passed" if cr.get("passed", False) else "❌ Failed"
                    print(f"   {j}. {status}: {cr.get('criterion', 'Unknown criterion')}")
                    if not cr.get("passed", False) and "reasoning" in cr:
                        print(f"      Reason: {cr['reasoning']}")
            
            print("─" * 50)
        except Exception as e:
            if debug:
                logging.error(f"Error reading validation file {file_path}: {e}")
            continue
    
    return True

def handle_update_parent_command(args):
    """Handle the update-parent command to propagate success or failure to parent goal."""
    from .goal_operations.goal_update import (
        propagate_success_state_to_parent, 
        propagate_failure_history_to_parent
    )
    
    if args.outcome == 'success':
        return propagate_success_state_to_parent(args.child_id)
    elif args.outcome == 'failed':
        return propagate_failure_history_to_parent(args.child_id)
    else:
        logging.error(f"Unknown outcome type: {args.outcome}")
        return False

def main_command(args):
    """Entry point for CLI commands."""
    # Handle async commands
    if args.command == "decompose":
        return decompose_existing_goal(args.goal_id, args.debug, args.quiet, args.bypass_validation)
    elif args.command == "execute":
        return execute_task(args.node_id, args.debug, args.quiet, args.bypass_validation, args.no_commit, args.memory_repo)
    elif args.command == "validate":
        from .validation import handle_validate_goal
        return handle_validate_goal(args.goal_id, args.debug, args.quiet, args.auto)
    
    # All other commands are synchronous, so just call them directly
    if args.command == "new":
        return create_new_goal(args.description)
    elif args.command == "delete":
        return delete_goal(args.goal_id)
    elif args.command == "complete":
        return mark_goal_complete()
    elif args.command == "status":
        return show_goal_status()
    elif args.command == "tree":
        return show_goal_tree()
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

def main():
    """Main CLI entry point."""
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
    
    # goal decompose <goal-id>
    decompose_parser = subparsers.add_parser("decompose", help="Decompose a goal into subgoals")
    decompose_parser.add_argument("goal_id", help="Goal ID to decompose")
    decompose_parser.add_argument("--debug", action="store_true", help="Show debug output")
    decompose_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    decompose_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    
    # goal execute <node-id>
    execute_parser = subparsers.add_parser("execute", help="Execute a goal/task node using the TaskExecutor")
    execute_parser.add_argument("node_id", help="Node ID to execute (e.g., G1, S1)")
    execute_parser.add_argument("--debug", action="store_true", help="Show debug output")
    execute_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    execute_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    execute_parser.add_argument("--no-commit", action="store_true", help="Prevent automatic commits")
    execute_parser.add_argument("--memory-repo", help="Path to memory repository")
    
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
    
    # goal diff <goal-id>
    diff_parser = subparsers.add_parser("diff", help="Show code and memory diffs for a specific goal")
    diff_parser.add_argument("goal_id", help="ID of the goal to show diffs for")
    # Add mutually exclusive group for diff modes
    mode_group = diff_parser.add_mutually_exclusive_group()
    mode_group.add_argument("--code", action="store_true", help="Show only code diff (default)")
    mode_group.add_argument("--memory", action="store_true", help="Show only memory diff")
    mode_group.add_argument("--complete", action="store_true", help="Show both code and memory diffs")
    
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
    
    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return
    
    # Run the main command
    main_command(args)

if __name__ == "__main__":
    main()


