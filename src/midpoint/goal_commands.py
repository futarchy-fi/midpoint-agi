"""CLI command parsing and dispatching for goal management."""

import argparse
import logging
import json
import subprocess
import datetime
from pathlib import Path

# Import command implementations from their new modules
from .goal_file_management import (
    list_goals, list_subgoals, delete_goal
)
from .goal_git import (
    go_back_commits, reset_to_commit
)
from .goal_visualization import (
    show_goal_status, show_goal_tree, show_goal_history, generate_graph, show_goal_diffs
)
from .goal_analysis import (
    analyze_goal, show_validation_history
)
from .goal_state import (
    create_new_goal, create_new_subgoal, create_new_task, mark_goal_complete, 
    merge_subgoal, update_parent_goal_state, update_git_state
)
# Import from goal_cli.py only if not yet refactored
from .goal_decompose_command import decompose_existing_goal
from .goal_revert import revert_goal
from .goal_execute_command import execute_task

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
    elif args.command == "merge":
        return merge_subgoal(args.subgoal_id)
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
        show_code = getattr(args, 'code', True) or getattr(args, 'complete', False) or not (getattr(args, 'memory', False) or getattr(args, 'complete', False))
        show_memory = getattr(args, 'memory', False) or getattr(args, 'complete', False)
        return show_goal_diffs(args.goal_id, show_code=show_code, show_memory=show_memory)
    elif args.command == "revert":
        return revert_goal(args.goal_id)
    else:
        return None

def handle_update_parent_command(args):
    """Handle the update-parent command to propagate success or failure to parent goal."""
    from .goal_operations.goal_update import (
        propagate_success_state_to_parent, propagate_failure_history_to_parent
    )
    
    if args.outcome == 'success':
        return propagate_success_state_to_parent(args.child_id)
    elif args.outcome == 'failed':
        return propagate_failure_history_to_parent(args.child_id)
    else:
        logging.error(f"Unknown outcome type: {args.outcome}")
        return False

def handle_solve_command(args):
    """
    Automate the process of analyzing, decomposing, and executing tasks defined
    in the .goal/ directory.
    (Implementation moved from goal_cli.py)
    """
    # Import and delegate to the implementation in goal_solver.py
    from .goal_solver import handle_solve_command as solver_handle_solve_command
    return solver_handle_solve_command(args)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Goal branch management commands")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    # Goal Management Commands
    new_parser = subparsers.add_parser("new", help="Create a new top-level goal")
    new_parser.add_argument("description", help="Description of the goal")
    delete_parser = subparsers.add_parser("delete", help="Delete a goal, subgoal, or task")
    delete_parser.add_argument("goal_id", help="ID of the goal to delete")
    sub_parser = subparsers.add_parser("sub", help="Create a subgoal under the specified parent")
    sub_parser.add_argument("parent_id", help="Parent goal ID")
    sub_parser.add_argument("description", help="Description of the subgoal")
    task_parser = subparsers.add_parser("task", help="Create a new directly executable task under the specified parent")
    task_parser.add_argument("parent_id", help="Parent goal ID")
    task_parser.add_argument("description", help="Description of the task")
    subparsers.add_parser("list", help="List all goals and subgoals in tree format")
    decompose_parser = subparsers.add_parser("decompose", help="Decompose a goal into subgoals")
    decompose_parser.add_argument("goal_id", help="Goal ID to decompose")
    decompose_parser.add_argument("--debug", action="store_true", help="Show debug output")
    decompose_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    decompose_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    execute_parser = subparsers.add_parser("execute", help="Execute a task using the TaskExecutor")
    execute_parser.add_argument("task_id", help="Task ID to execute")
    execute_parser.add_argument("--debug", action="store_true", help="Show debug output")
    execute_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    execute_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    execute_parser.add_argument("--no-commit", action="store_true", help="Prevent automatic commits")
    execute_parser.add_argument("--memory-repo", help="Path to memory repository")
    solve_parser = subparsers.add_parser("solve", help="Automatically analyze, decompose, and execute tasks for a goal")
    solve_parser.add_argument("goal_id", help="Goal ID to solve")
    solve_parser.add_argument("--debug", action="store_true", help="Show debug output")
    solve_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    solve_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    back_parser = subparsers.add_parser("back", help="Go back N commits on current goal branch")
    back_parser.add_argument("steps", nargs="?", type=int, default=1, help="Number of commits to go back")
    reset_parser = subparsers.add_parser("reset", help="Reset to specific commit on current branch")
    reset_parser.add_argument("commit_id", help="Commit ID to reset to")
    subparsers.add_parser("subs", help="List available subgoals for current goal")
    subparsers.add_parser("complete", help="Mark current goal as complete")
    merge_parser = subparsers.add_parser("merge", help="Merge specific subgoal into current goal")
    merge_parser.add_argument("subgoal_id", help="Subgoal ID to merge")
    subparsers.add_parser("status", help="Show completion status of all goals")
    subparsers.add_parser("tree", help="Show visual representation of goal hierarchy")
    subparsers.add_parser("history", help="Show timeline of goal exploration")
    subparsers.add_parser("graph", help="Generate graphical visualization")
    diff_parser = subparsers.add_parser("diff", help="Show code and memory diffs for a specific goal")
    diff_parser.add_argument("goal_id", help="ID of the goal to show diffs for")
    mode_group = diff_parser.add_mutually_exclusive_group()
    mode_group.add_argument("--code", action="store_true", help="Show only code diff (default)")
    mode_group.add_argument("--memory", action="store_true", help="Show only memory diff")
    mode_group.add_argument("--complete", action="store_true", help="Show both code and memory diffs")
    diff_parser.set_defaults(func=lambda args: show_goal_diffs(
                                 args.goal_id,
                                 show_code=(args.code or args.complete or not (args.memory or args.complete)),
                                 show_memory=(args.memory or args.complete)
                             ))
    revert_parser = subparsers.add_parser("revert", help="Revert a goal's current state back to its initial state")
    revert_parser.add_argument("goal_id", help="ID of the goal to revert")
    validate_parser = subparsers.add_parser("validate", help="Validate a goal's completion criteria")
    validate_parser.add_argument("goal_id", help="ID of the goal to validate")
    validate_parser.add_argument("--debug", action="store_true", help="Show debug output")
    validate_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    validate_parser.add_argument("--auto", action="store_true", help="Perform automated validation using LLM")
    validate_parser.add_argument("--model", default="gpt-4o-mini", help="Model to use for validation (with --auto)")
    validate_history_parser = subparsers.add_parser("validate-history", help="Show validation history for a goal")
    validate_history_parser.add_argument("goal_id", help="ID of the goal to show validation history for")
    validate_history_parser.add_argument("--debug", action="store_true", help="Show debug output")
    validate_history_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
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
    main_command(args)

if __name__ == "__main__":
    main()
