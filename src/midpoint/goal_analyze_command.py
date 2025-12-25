"""CLI orchestration for goal analysis.

This is intentionally parallel to `goal_decompose_command.py`:
- Load `.goal/<goal_id>.json`
- Checkout the goal's top-level branch
- Call the GoalAnalyzer agent using **keyword arguments** (avoid arg-order bugs)
- Persist `last_analysis` back into the goal file
"""

import os
import json
import logging
import datetime
import subprocess
from pathlib import Path

from .agents.goal_analyzer import analyze_goal as agent_analyze_goal
from .goal_git import get_current_branch, find_top_level_branch
from .constants import GOAL_DIR


def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path(GOAL_DIR)
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: {GOAL_DIR}")
    return goal_path


def analyze_existing_goal(goal_id, debug=False, quiet=False, bypass_validation=False):
    """Analyze an existing goal using the GoalAnalyzer agent."""
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"

    if not goal_file.exists():
        logging.error(f"Goal {goal_id} not found")
        return False

    # Load the goal data
    try:
        with open(goal_file, "r") as f:
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

        description = goal_data.get("description", "")
        validation_criteria = goal_data.get("validation_criteria", [])
        current_state = goal_data.get("current_state", {}) or {}

        memory_repo_path = current_state.get("memory_repository_path") or os.getenv("MEMORY_REPO_PATH")
        memory_hash = current_state.get("memory_hash")

        # Prefer the repo path stored in the goal file (repo root), fall back to cwd.
        repo_path = current_state.get("repository_path") or os.getcwd()

        # Call analyzer with keyword args so `goal_id` can't be mistaken for `repo_path`
        analysis = agent_analyze_goal(
            repo_path=repo_path,
            goal=description,
            validation_criteria=validation_criteria,
            parent_goal_id=goal_data.get("parent_goal") or None,
            goal_id=goal_id,
            memory_hash=memory_hash,
            memory_repo_path=memory_repo_path,
            debug=debug,
            quiet=quiet,
            bypass_validation=bypass_validation,
            logs_dir="logs",
            input_file=None,
        )

        if not isinstance(analysis, dict) or not analysis.get("success", False):
            logging.error(f"Analysis failed: {analysis}")
            return False

        # Persist a small, human-readable summary back into the goal file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        goal_data["last_analysis"] = {
            "timestamp": timestamp,
            "suggested_action": analysis.get("action", ""),
            "justification": analysis.get("justification", ""),
            "strategic_guidance": analysis.get("strategic_guidance", ""),
        }

        with open(goal_file, "w") as f:
            json.dump(goal_data, f, indent=2)

        # Print result
        print(f"Analysis for {goal_id}: {goal_data['last_analysis']['suggested_action']}")
        if goal_data["last_analysis"]["justification"]:
            print(f"Justification: {goal_data['last_analysis']['justification']}")
        if goal_data["last_analysis"]["strategic_guidance"]:
            print(f"Strategic guidance: {goal_data['last_analysis']['strategic_guidance']}")

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

