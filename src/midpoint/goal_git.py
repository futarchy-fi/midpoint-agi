import os
import subprocess
import logging
import json
from typing import Optional
from pathlib import Path


def get_current_hash(repo_path: Optional[str] = None) -> str:
    """Get the current git hash.
    
    Args:
        repo_path: Optional path to the git repository. If not provided, uses current directory.
        
    Returns:
        The current git hash as a string.
    """
    try:
        # If no repo_path provided, use current directory
        if not repo_path:
            repo_path = os.getcwd()
            
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get current git hash: {e}")
        return None


def get_current_branch():
    """Get the current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get current branch: {e}")
        return None


def get_goal_id_from_branch(branch_name):
    """Extract goal ID from branch name."""
    # Branch naming convention: goal-G1-abcdef or goal-S1-abcdef
    parts = branch_name.split('-')
    
    # Check if it's a goal branch
    if len(parts) < 2 or parts[0] != "goal":
        return None
    
    # Check for valid goal/subgoal ID
    if len(parts) >= 2:
        # Check if it's a valid ID (G or S prefix only)
        if parts[1].startswith('G') and parts[1][1:].isdigit():
            return parts[1]
        if parts[1].startswith('S') and parts[1][1:].isdigit():
            return parts[1]
    
    return None


def find_branch_for_goal(goal_id):
    """Find the git branch for a specific goal ID."""
    try:
        # Get all branches
        result = subprocess.run(
            ["git", "branch"],
            check=True,
            capture_output=True,
            text=True
        )
        
        branches = result.stdout.strip().split('\n')
        branches = [b.strip() for b in branches]
        branches = [b[2:] if b.startswith('* ') else b for b in branches]  # Remove the * from current branch
        
        # Look for branches with the goal ID
        for branch in branches:
            if branch.startswith(f"goal-{goal_id}-") or branch == f"goal-{goal_id}":
                return branch
        
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to list branches: {e}")
        return None


def get_recent_commits(count=10):
    """Get recent commits."""
    try:
        result = subprocess.run(
            ["git", "log", f"-{count}", "--oneline"],
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get recent commits: {e}")
        return []


def go_back_commits(steps=1):
    """Go back N commits on the current branch."""
    if steps < 1:
        logging.error("Number of steps must be at least 1")
        return False
    
    try:
        # Get current branch
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        # Get current hash before we move
        current_hash = get_current_hash()
        
        # Reset to N commits back
        reset_point = f"HEAD~{steps}"
        subprocess.run(
            ["git", "reset", "--hard", reset_point],
            check=True,
            capture_output=True,
            text=True
        )
        
        new_hash = get_current_hash()
        print(f"Moved back {steps} commit(s) on branch {current_branch}")
        print(f"From: {current_hash[:8]} -> To: {new_hash[:8]}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to go back commits: {e}")
        return False


def reset_to_commit(commit_id):
    """Reset to a specific commit on the current branch."""
    try:
        # Verify commit exists
        result = subprocess.run(
            ["git", "cat-file", "-t", commit_id],
            check=True,
            capture_output=True,
            text=True
        )
        
        if "commit" not in result.stdout:
            logging.error(f"Invalid commit ID: {commit_id}")
            return False
        
        # Get current branch
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        # Get current hash before we move
        current_hash = get_current_hash()
        
        # Reset to the specified commit
        subprocess.run(
            ["git", "reset", "--hard", commit_id],
            check=True,
            capture_output=True,
            text=True
        )
        
        new_hash = get_current_hash()
        print(f"Reset to commit {commit_id} on branch {current_branch}")
        print(f"From: {current_hash[:8]} -> To: {new_hash[:8]}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to reset to commit: {e}")
        return False


def run_diff_command(repo_path: str, initial_hash: str, final_hash: str) -> Optional[str]:
    """Runs 'git diff' in the specified repository and returns the output."""
    if not repo_path or not os.path.exists(repo_path):
        logging.warning(f"Repository path not found or not specified: {repo_path}")
        return None
    if not initial_hash or not final_hash:
        logging.warning("Initial or final hash missing, cannot perform diff.")
        return None
    if initial_hash == final_hash:
        logging.info(f"Initial and final hashes are the same ({initial_hash[:8]}) in {repo_path}. No changes.")
        return "(No changes)"

    # Use '..' notation which is standard for git diff range
    # Add '--' to prevent ambiguity if hashes resemble filenames
    command = ["git", "diff", f"{initial_hash}..{final_hash}", "--"]
    logging.debug(f"Running diff in {repo_path}: {' '.join(command)}")
    try:
        # Using subprocess.run to capture output directly
        result = subprocess.run(
            command,
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            errors='ignore' # Ignore potential decoding errors in diff output
        )
        # Return stdout if it exists, otherwise indicate no textual changes
        return result.stdout if result.stdout else "(No textual changes detected)"
    except FileNotFoundError:
         logging.error(f"Git command not found. Ensure Git is installed and in PATH.")
         return f"Error: Git command not found in {repo_path}"
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running git diff in {repo_path}: {e}")
        logging.error(f"Stderr: {e.stderr}")
        # Provide more context on error
        error_message = f"Error running git diff in {repo_path}."
        # Check stderr for common git diff errors
        if "ambiguous argument" in e.stderr or "unknown revision" in e.stderr:
             error_message += f" One of the hashes ({initial_hash[:8]}, {final_hash[:8]}) might be invalid or not found in this repository."
        elif "fatal: bad object" in e.stderr:
            error_message += f" One of the hashes ({initial_hash[:8]}, {final_hash[:8]}) is not a valid git object."
        else:
            # Include generic stderr if specific patterns don't match
            error_message += f" Stderr: {e.stderr.strip()}"
        return error_message
    except Exception as e:
        logging.error(f"Unexpected error running git diff in {repo_path}: {e}")
        return f"Unexpected error during diff in {repo_path}: {e}"


def find_top_level_branch(goal_id):
    """Find the branch of the top-level goal by traversing up the goal hierarchy."""
    goal_path = ensure_goal_dir()  # Use the local ensure_goal_dir function
    
    # Keep track of visited goals to prevent cycles
    visited = set()
    
    while goal_id and goal_id not in visited:
        visited.add(goal_id)
        
        # Check if this is a top-level goal (starts with G)
        if goal_id.startswith('G'):
            goal_file = goal_path / f"{goal_id}.json"
            if goal_file.exists():
                try:
                    with open(goal_file, 'r') as f:
                        goal_data = json.load(f)
                        branch_name = goal_data.get('branch_name')
                        if branch_name:
                            return branch_name
                        # If no branch name, generate one
                        branch_name = f"goal-{goal_id}"
                        goal_data['branch_name'] = branch_name
                        with open(goal_file, 'w') as f:
                            json.dump(goal_data, f, indent=2)
                        return branch_name
                except Exception as e:
                    logging.error(f"Failed to read goal file: {e}")
                    return None
        
        # Get parent goal ID
        goal_file = goal_path / f"{goal_id}.json"
        if not goal_file.exists():
            return None
            
        try:
            with open(goal_file, 'r') as f:
                goal_data = json.load(f)
                goal_id = goal_data.get("parent_goal", "")
                if goal_id.endswith('.json'):
                    goal_id = goal_id[:-5]  # Remove .json extension
        except Exception as e:
            logging.error(f"Failed to read goal file: {e}")
            return None
    
    return None


def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path('.goal')
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: .goal")
    return goal_path
