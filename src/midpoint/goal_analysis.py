"""Goal analysis helpers used by the CLI.

Historically `goal_cli.py` contained an `analyze_goal()` wrapper that called the GoalAnalyzer
agent directly. That was error-prone because the agent API expects `repo_path` and `goal`
as leading arguments.

This module keeps the CLI surface small by delegating to `goal_analyze_command.py`,
which mirrors the other agent command modules (`goal_decompose_command.py`, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .goal_analyze_command import analyze_existing_goal


def analyze_goal(goal_id: str, human_mode: bool = False) -> bool:
    """Analyze a goal and persist `last_analysis` into `.goal/<goal_id>.json`."""
    return analyze_existing_goal(goal_id=goal_id, human=human_mode, debug=human_mode)


def show_validation_history(goal_id: str, debug: bool = False, quiet: bool = False) -> bool:
    """Show validation history for a specific goal.

    This is used by `goal validate-history` and is kept here so `goal_commands.py` can
    import it without depending on the legacy `goal_cli.py`.
    """
    validation_dir = Path("logs/validation_history")
    if not validation_dir.exists():
        if not quiet:
            print(f"No validation history found (directory {validation_dir} does not exist)")
        return False

    history_files = sorted(
        list(validation_dir.glob(f"{goal_id}_*.json")),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not history_files:
        if not quiet:
            print(f"No validation history found for goal {goal_id}")
        return False

    print(f"Validation History for Goal {goal_id}:")
    print("═" * 50)

    for i, file_path in enumerate(history_files, 1):
        if file_path.name.endswith("_context.json"):
            continue

        try:
            with open(file_path, "r") as f:
                record: Dict[str, Any] = json.load(f)

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
        except Exception:
            # Best-effort history display; skip malformed records.
            continue

    return True

