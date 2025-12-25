"""CLI orchestration for goal analysis.

This module mirrors the structure of:
- `goal_decompose_command.py` (decomposer)
- `goal_execute_command.py` (executor)
- `validation.py` (validator)

It is responsible for:
1) Loading `.goal/<goal_id>.json`
2) Ensuring we are on the correct goal branch
3) Calling the GoalAnalyzer agent with keyword arguments (to avoid arg-order bugs)
4) Persisting `last_analysis` back onto the goal file for other commands to use
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .agents.goal_analyzer import analyze_goal as agent_analyze_goal
from .goal_git import find_top_level_branch, get_current_branch


def ensure_goal_dir() -> Path:
    """Ensure the .goal directory exists."""
    goal_path = Path(".goal")
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info("Created goal directory: .goal")
    return goal_path


def _suggest_command_for_action(action: str, goal_id: str) -> str:
    """Suggest a follow-up CLI command based on analyzer action."""
    action = (action or "").strip().lower()
    if action in {"decompose"}:
        return f"goal decompose {goal_id}"
    if action in {"execute"}:
        # Could be a goal that needs converting to a task; execute expects a task id.
        return f"goal execute <task-id>"
    if action in {"validate"}:
        return f"goal validate {goal_id}"
    if action in {"complete", "mark_complete"}:
        return "goal complete"
    if action in {"give_up"}:
        return f"goal update-parent {goal_id} --outcome failed"
    return ""


def analyze_existing_goal(
    goal_id: str,
    debug: bool = False,
    quiet: bool = False,
    bypass_validation: bool = False,
    human: bool = False,
) -> bool:
    """Analyze an existing goal using the GoalAnalyzer.

    Returns:
        True on success, False on failure.
    """
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"

    if not goal_file.exists():
        logging.error(f"Goal {goal_id} not found")
        return False

    try:
        with open(goal_file, "r") as f:
            goal_data: Dict[str, Any] = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read goal file: {e}")
        return False

    # Determine the "correct" branch to analyze on (top-level goal branch).
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
            text=True,
        )
        has_changes = bool(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to check git status: {e}")
        return False

    # Stash changes if needed
    if has_changes:
        try:
            subprocess.run(
                ["git", "stash", "push", "-m", f"Stashing changes before analyzing goal {goal_id}"],
                check=True,
                capture_output=True,
                text=True,
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
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to checkout branch {top_level_branch}: {e}")
            return False

        description = goal_data.get("description") or ""
        validation_criteria = goal_data.get("validation_criteria") or []

        current_state = goal_data.get("current_state") or {}
        memory_repo_path = current_state.get("memory_repository_path") or os.getenv("MEMORY_REPO_PATH")
        memory_hash = current_state.get("memory_hash")

        # Call agent analyzer using keyword args (prevents arg-order bugs)
        result = agent_analyze_goal(
            repo_path=os.getcwd(),
            goal=description,
            validation_criteria=validation_criteria,
            parent_goal_id=goal_data.get("parent_goal") or None,
            goal_id=goal_id,
            memory_hash=memory_hash,
            memory_repo_path=memory_repo_path,
            debug=debug or human,
            quiet=quiet,
            bypass_validation=bypass_validation,
            logs_dir="logs",
            input_file=None,  # keep simple; goal context is already loaded above
        )

        if not isinstance(result, dict) or not result.get("success", False):
            logging.error(f"Goal analysis failed: {result}")
            return False

        action = result.get("action", "")
        justification = result.get("justification", "")
        strategic_guidance = result.get("strategic_guidance", "")
        suggested_command = _suggest_command_for_action(action, goal_id)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        goal_data["last_analysis"] = {
            "timestamp": timestamp,
            "suggested_action": action,
            "justification": justification,
            "strategic_guidance": strategic_guidance,
            "suggested_command": suggested_command,
            "mode": "human" if human else "auto",
        }
        # Used by GoalDecomposer as an optional hint.
        goal_data["analysis_justification"] = justification

        try:
            with open(goal_file, "w") as f:
                json.dump(goal_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to persist analysis results to goal file: {e}")
            return False

        # User-facing output
        print(f"Analysis for {goal_id}: {action}")
        if justification:
            print(f"Justification: {justification}")
        if strategic_guidance:
            print(f"Strategic guidance: {strategic_guidance}")
        if suggested_command:
            print(f"Suggested command: {suggested_command}")

        return True
    finally:
        # Always restore the original branch and unstash changes
        try:
            subprocess.run(
                ["git", "checkout", current_branch],
                check=True,
                capture_output=True,
                text=True,
            )

            if has_changes:
                subprocess.run(
                    ["git", "stash", "pop"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to restore original state: {e}")

