"""CLI orchestration for task execution."""

import os
import json
import sys
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

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


def _ensure_attempt_dir(goal_path: Path, node_id: str) -> Path:
    """
    Create `.goal/<node_id>/attempts/` and return that path.
    This directory is intended for append-only attempt journaling.
    """
    attempt_dir = goal_path / node_id / "attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    return attempt_dir


def _write_attempt_record(goal_path: Path, node_id: str, record: Dict[str, Any]) -> Optional[str]:
    """
    Write a single attempt record JSON file under `.goal/<node_id>/attempts/`.
    Returns the relative path (from `.goal/`) if successful.
    """
    attempt_dir = _ensure_attempt_dir(goal_path, node_id)
    attempt_id = str(record.get("attempt_id") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Ensure unique filename if multiple attempts happen in the same second.
    candidate = attempt_dir / f"{attempt_id}.json"
    suffix = 1
    while candidate.exists():
        candidate = attempt_dir / f"{attempt_id}_{suffix}.json"
        suffix += 1

    try:
        with open(candidate, "w") as f:
            json.dump(record, f, indent=2)
        return str(candidate.relative_to(goal_path))
    except Exception as e:
        logging.warning(f"Failed to write attempt record for {node_id}: {e}")
        return None


def _read_memory_document_at_commit(
    memory_repo_path: Optional[str],
    memory_commit: Optional[str],
    document_path: Optional[str],
    *,
    char_limit: int = 1_000_000,
) -> Optional[Dict[str, Any]]:
    """
    Read a memory document at a specific commit using `git show <commit>:<path>`.

    Returns a dict containing the embedded transcript content and truncation metadata.
    If any required inputs are missing or the read fails, returns None.
    """
    if not memory_repo_path or not memory_commit or not document_path:
        return None

    try:
        result = subprocess.run(
            ["git", "show", f"{memory_commit}:{document_path}"],
            cwd=memory_repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        content = result.stdout or ""
        truncated = False
        if char_limit is not None and len(content) > char_limit:
            content = content[:char_limit]
            truncated = True
        return {
            "content": content,
            "truncated": truncated,
            "included_chars": len(content),
        }
    except Exception as e:
        logging.warning(f"Failed to read memory transcript at {memory_commit}:{document_path}: {e}")
        return None


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

        attempt_started_at = datetime.datetime.now()
        attempt_started_ts = attempt_started_at.strftime("%Y%m%d_%H%M%S")
        
        # Create and run the executor
        executor = TaskExecutor()
        execution_result = executor.execute_task(context, task_data["description"])
        attempt_ended_at = datetime.datetime.now()
        attempt_ended_ts = attempt_ended_at.strftime("%Y%m%d_%H%M%S")

        # --- Append attempt record under `.goal/<node_id>/attempts/` (append-only journal) ---
        try:
            meta = execution_result.metadata or {}
            tool_usage = meta.get("tool_usage", []) if isinstance(meta, dict) else []

            tool_call_counts: Dict[str, int] = {}
            last_tool_calls = []
            for entry in tool_usage or []:
                if isinstance(entry, dict):
                    tool_name = entry.get("tool")
                    if tool_name:
                        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                        last_tool_calls.append({"tool": tool_name, "args": entry.get("args", {})})
            last_tool_calls = last_tool_calls[-10:]

            mem_store = meta.get("memory_store_result") if isinstance(meta, dict) else None
            mem_store_compact = None
            if isinstance(mem_store, dict):
                mem_store_compact = {
                    "success": mem_store.get("success"),
                    "document_path": mem_store.get("document_path"),
                    "error": mem_store.get("error"),
                }
            # Normalize memory document path + commit for embedding transcript.
            mem_doc_path = None
            mem_doc_commit = None
            if isinstance(mem_store_compact, dict):
                doc = mem_store_compact.get("document_path")
                if isinstance(doc, dict):
                    mem_doc_path = doc.get("path")
                    mem_doc_commit = doc.get("memory_hash")
                elif isinstance(doc, str):
                    mem_doc_path = doc
                    mem_doc_commit = (
                        (execution_result.final_state.memory_hash if execution_result.final_state else None)
                        or task_data.get("initial_state", {}).get("memory_hash")
                    )

            embedded_transcript = _read_memory_document_at_commit(
                task_data.get("initial_state", {}).get("memory_repository_path"),
                mem_doc_commit,
                mem_doc_path,
            )

            attempt_record = {
                "attempt_id": attempt_started_ts,
                "node_id": node_id,
                "node_description": task_data.get("description"),
                "parent_goal": task_data.get("parent_goal") or "",
                "top_level_branch": top_level_branch,
                "started_at": attempt_started_ts,
                "ended_at": attempt_ended_ts,
                "duration_seconds": (attempt_ended_at - attempt_started_at).total_seconds(),
                "baseline": {
                    "git_hash": task_data.get("initial_state", {}).get("git_hash"),
                    "memory_hash": task_data.get("initial_state", {}).get("memory_hash"),
                    "repository_path": task_data.get("initial_state", {}).get("repository_path"),
                    "memory_repository_path": task_data.get("initial_state", {}).get("memory_repository_path"),
                },
                "outcome": {
                    "success": execution_result.success,
                    "summary": execution_result.summary,
                    "error_message": execution_result.error_message,
                    "final_git_hash": execution_result.final_state.git_hash if execution_result.final_state else None,
                    "final_memory_hash": execution_result.final_state.memory_hash if execution_result.final_state else None,
                },
                "diagnostics": {
                    "tool_call_counts": tool_call_counts,
                    "last_tool_calls": last_tool_calls,
                },
                "artifacts": {
                    # Embed the attempt transcript directly so we don't depend on pointers
                    # into the memory repo (which may be GC'd/rewritten later).
                    "memory_transcript": embedded_transcript,
                },
            }

            attempt_rel_path = _write_attempt_record(goal_path, node_id, attempt_record)
            if attempt_rel_path:
                logging.info(f"Wrote attempt record: .goal/{attempt_rel_path}")
        except Exception as e:
            logging.warning(f"Failed to append attempt record for {node_id}: {e}")

        # --- Prepare the data for last_execution field ---
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
        # Canonical field used across the codebase.
        task_data["last_execution"] = last_execution_data

        if execution_result.success:
            logging.info(f"Node {node_id} execution reported success.")
            
            # Still mark task as completed conceptually upon successful execution report
            # (Though actual merging/acceptance is separate)
            task_data["complete"] = True
            task_data["completion_time"] = last_execution_data["timestamp"]

            # Save updated task data (now only updates complete, completion_time, and last_execution)
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
            
            # Save updated task data with failure status and last_execution
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
