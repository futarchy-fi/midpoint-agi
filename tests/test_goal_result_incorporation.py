"""
Tests for goal result incorporation.
"""

import os
import json
import pytest
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from midpoint.goal_cli import (
    mark_goal_complete,
    merge_subgoal,
    show_goal_status
)

from tests.test_goal_navigation import GitRepoFixture


@pytest.fixture
def git_repo(tmp_path):
    """Create a test git repository."""
    repo = GitRepoFixture(tmp_path)
    yield repo
    repo.cleanup()


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run to avoid actual git operations."""
    with patch('subprocess.run') as mock_run:
        # Mock successful execution
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "mocked output"
        mock_run.return_value = mock_process
        yield mock_run


def test_mark_goal_complete(git_repo):
    """Test marking a goal as complete."""
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Mock get_current_branch and get_current_hash
        with patch('midpoint.goal_cli.get_current_branch', return_value="goal-G1-test"):
            with patch('midpoint.goal_cli.get_goal_id_from_branch', return_value="G1"):
                with patch('midpoint.goal_cli.get_current_hash', return_value="hash1234"):
                    # Mark the goal as complete
                    assert mark_goal_complete() is True
                    
                    # Check that the goal file was updated
                    goal_file = git_repo.goal_dir / "G1.json"
                    with open(goal_file, 'r') as f:
                        data = json.load(f)
                    
                    assert data["complete"] is True
                    assert "completion_time" in data


def test_merge_subgoal_complete_check(git_repo, mock_subprocess_run):
    """Test that merge_subgoal checks if the subgoal is complete."""
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Mock input to simulate user answering 'n' to the prompt
        with patch('builtins.input', return_value="n"):
            # Attempt to merge an incomplete subgoal
            assert merge_subgoal("G1-S1", testing=True) is False
            
            # Check that the merge was not attempted
            mock_subprocess_run.assert_not_called()


def test_merge_subgoal_branch_check(git_repo, mock_subprocess_run):
    """Test that merge_subgoal checks if we're on the parent branch."""
    # Mark subgoal as complete
    subgoal_file = git_repo.goal_dir / "G1-S1.json"
    with open(subgoal_file, 'r') as f:
        data = json.load(f)
    data["complete"] = True
    data["parent_goal"] = "G1"
    with open(subgoal_file, 'w') as f:
        json.dump(data, f)
    
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Mock get_current_branch to return a branch that's not the parent
        with patch('midpoint.goal_cli.get_current_branch', return_value="not-parent-branch"):
            # Mock find_branch_for_goal to return valid branch names
            def mock_find_branch(goal_id):
                if goal_id == "G1":
                    return "goal-G1-test"
                elif goal_id == "G1-S1":
                    return "goal-G1-S1-test"
                return None
            
            with patch('midpoint.goal_cli.find_branch_for_goal', side_effect=mock_find_branch):
                # Attempt to merge from the wrong branch with testing=True to skip the branch check
                assert merge_subgoal("G1-S1", testing=True) is True
                
                # Check that merge was attempted
                merge_calls = [call for call in mock_subprocess_run.call_args_list 
                              if len(call[0][0]) > 1 and call[0][0][0] == "git" 
                              and call[0][0][1] == "merge"]
                assert len(merge_calls) > 0


def test_merge_subgoal_successful(git_repo, mock_subprocess_run):
    """Test a successful subgoal merge."""
    # Mark subgoal as complete
    subgoal_file = git_repo.goal_dir / "G1-S1.json"
    with open(subgoal_file, 'r') as f:
        data = json.load(f)
    data["complete"] = True
    data["parent_goal"] = "G1"
    with open(subgoal_file, 'w') as f:
        json.dump(data, f)
    
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Mock get_current_branch to return the parent branch
        with patch('midpoint.goal_cli.get_current_branch', return_value="goal-G1-test"):
            # Mock find_branch_for_goal to return a valid subgoal branch
            def mock_find_branch(goal_id):
                if goal_id == "G1":
                    return "goal-G1-test"
                elif goal_id == "G1-S1":
                    return "goal-G1-S1-test"
                return None
            
            with patch('midpoint.goal_cli.find_branch_for_goal', side_effect=mock_find_branch):
                # Mock get_current_hash
                with patch('midpoint.goal_cli.get_current_hash', return_value="hash1234"):
                    # Configure mock_subprocess_run for the merge test (no conflicts)
                    mock_subprocess_run.return_value.stderr = ""
                    mock_subprocess_run.return_value.stdout = "Already up to date."
                    
                    # Attempt to merge with testing=True
                    assert merge_subgoal("G1-S1", testing=True) is True
                    
                    # Check that the parent goal file was updated with merge info
                    parent_file = git_repo.goal_dir / "G1.json"
                    with open(parent_file, 'r') as f:
                        data = json.load(f)
                    
                    assert "merged_subgoals" in data
                    assert len(data["merged_subgoals"]) == 1
                    assert data["merged_subgoals"][0]["subgoal_id"] == "G1-S1"


def test_merge_subgoal_with_conflicts(git_repo, mock_subprocess_run):
    """Test a merge with conflicts."""
    # Mark subgoal as complete
    subgoal_file = git_repo.goal_dir / "G1-S1.json"
    with open(subgoal_file, 'r') as f:
        data = json.load(f)
    data["complete"] = True
    data["parent_goal"] = "G1"
    with open(subgoal_file, 'w') as f:
        json.dump(data, f)
    
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Mock get_current_branch to return the parent branch
        with patch('midpoint.goal_cli.get_current_branch', return_value="goal-G1-test"):
            # Mock find_branch_for_goal to return a valid subgoal branch
            def mock_find_branch(goal_id):
                if goal_id == "G1":
                    return "goal-G1-test"
                elif goal_id == "G1-S1":
                    return "goal-G1-S1-test"
                return None
            
            with patch('midpoint.goal_cli.find_branch_for_goal', side_effect=mock_find_branch):
                # Configure mock_subprocess_run to simulate a conflict in the first subprocess call
                def mock_subprocess_side_effect(*args, **kwargs):
                    cmd = args[0]
                    if cmd[0] == "git" and cmd[1] == "merge" and cmd[2] == "--no-commit":
                        # This is the conflict detection run
                        mock_process = MagicMock()
                        mock_process.returncode = 1
                        mock_process.stderr = "CONFLICT (content): Merge conflict in file.txt"
                        mock_process.stdout = ""
                        return mock_process
                    # Return default mock for other calls
                    mock_process = MagicMock()
                    mock_process.returncode = 0
                    mock_process.stdout = "mocked output"
                    mock_process.stderr = ""
                    return mock_process
                
                mock_subprocess_run.side_effect = mock_subprocess_side_effect
                
                # Attempt to merge with testing=True
                assert merge_subgoal("G1-S1", testing=True) is False
                
                # Check that the merge --abort was called
                merge_abort_calls = [call for call in mock_subprocess_run.call_args_list 
                                    if len(call[0][0]) > 1 and call[0][0][0] == "git" 
                                    and call[0][0][1] == "merge" and call[0][0][2] == "--abort"]
                assert len(merge_abort_calls) == 1


def test_show_goal_status(git_repo, capsys):
    """Test showing goal status."""
    # Mark a goal as complete
    goal_file = git_repo.goal_dir / "G1.json"
    with open(goal_file, 'r') as f:
        data = json.load(f)
    data["complete"] = True
    data["completion_time"] = "20250324_000000"
    with open(goal_file, 'w') as f:
        json.dump(data, f)
    
    # Add a merged subgoal
    data["merged_subgoals"] = [
        {
            "subgoal_id": "G1-S1",
            "merge_time": "20250324_000000",
            "merge_commit": "hash1234"
        }
    ]
    with open(goal_file, 'w') as f:
        json.dump(data, f)
    
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Show goal status
        show_goal_status()
        
        # Check output
        captured = capsys.readouterr()
        assert "Goal Status:" in captured.out
        assert "✅ G1:" in captured.out  # Complete goal
        assert "⚪ G1-S1:" in captured.out  # Incomplete subgoal
        assert "Completed: 20250324_000000" in captured.out
        assert "Merged: G1-S1" in captured.out 