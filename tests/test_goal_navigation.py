"""
Tests for goal navigation commands.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from midpoint.goal_cli import (
    get_goal_id_from_branch,
    get_parent_goal_id,
    find_branch_for_goal,
    go_back_commits,
    reset_to_commit,
    go_to_parent_goal,
    go_to_child,
    go_to_root_goal,
    list_subgoals
)

# Test fixture for a mock Git repository
class GitRepoFixture:
    """Test fixture for a mock Git repository."""
    
    def __init__(self, tmp_path):
        """Initialize the fixture."""
        self.repo_path = tmp_path / "test_repo"
        self.repo_path.mkdir()
        
        # Set up goal directory
        self.goal_dir = self.repo_path / ".goal"
        self.goal_dir.mkdir()
        
        # Create some test goals
        self._create_goal_branch("G1", "Test goal 1")
        self._create_goal_branch("S1", "Test subgoal 1", parent="G1")
        
        # Change to repo directory
        self.original_dir = os.getcwd()
        os.chdir(self.repo_path)
    
    def _create_goal_branch(self, goal_id, description, parent=None):
        """Create a goal branch and JSON file."""
        # Create goal file
        goal_file = self.goal_dir / f"{goal_id}.json"
        goal_data = {
            "goal_id": goal_id,
            "description": description,
            "parent_goal": parent or "",
            "timestamp": "20250324_000000",
            "is_task": False,
            "branch_name": f"goal-{goal_id}-test",
            "initial_state": {
                "git_hash": "hash1234",
                "repository_path": str(self.repo_path),
                "description": f"Initial state for {goal_id}",
                "timestamp": "20250324_000000"
            },
            "current_state": {
                "git_hash": "hash5678",
                "repository_path": str(self.repo_path),
                "description": f"Current state for {goal_id}",
                "timestamp": "20250324_010000"
            }
        }
        
        with open(goal_file, 'w') as f:
            json.dump(goal_data, f)
    
    def cleanup(self):
        """Clean up the fixture."""
        os.chdir(self.original_dir)


@pytest.fixture
def git_repo(tmp_path):
    """Fixture for a mock Git repository."""
    repo = GitRepoFixture(tmp_path)
    yield repo
    repo.cleanup()


@pytest.fixture
def mock_subprocess_run(monkeypatch):
    """Mock subprocess.run for Git commands."""
    mock_run = MagicMock()
    mock_run.return_value.stdout = "goal-G1-test"
    monkeypatch.setattr("subprocess.run", mock_run)
    return mock_run


def test_get_goal_id_from_branch():
    """Test extracting goal ID from branch name."""
    assert get_goal_id_from_branch("goal-G1-test") == "G1"
    assert get_goal_id_from_branch("goal-S1-test") == "S1"
    assert get_goal_id_from_branch("main") is None


def test_get_parent_goal_id(git_repo):
    """Test getting parent goal ID."""
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        assert get_parent_goal_id("S1") == "G1"
        assert get_parent_goal_id("G1") == ""
        
        # Test with non-existent goal
        assert get_parent_goal_id("X1") is None


def test_find_branch_for_goal(git_repo, mock_subprocess_run):
    """Test finding branch for a goal."""
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Set up mock return for git branch list
        mock_subprocess_run.return_value.stdout = "* goal-G1-test\n  goal-S1-test\n  main"
        
        branch = find_branch_for_goal("G1")
        assert branch == "goal-G1-test"


def test_go_back_commits(git_repo, mock_subprocess_run):
    """Test going back N commits."""
    assert go_back_commits(2) is True
    
    # Check that git reset was called correctly
    mock_subprocess_run.assert_called_with(
        ["git", "reset", "--hard", "HEAD~2"],
        check=True,
        capture_output=True,
        text=True
    )


def test_reset_to_commit(git_repo, mock_subprocess_run):
    """Test resetting to a specific commit."""
    assert reset_to_commit("abcd1234") is True
    
    # Check that git reset was called correctly
    mock_subprocess_run.assert_called_with(
        ["git", "reset", "--hard", "abcd1234"],
        check=True,
        capture_output=True,
        text=True
    )


def test_go_to_parent_goal(git_repo, mock_subprocess_run):
    """Test going to parent goal."""
    # Mock get_current_branch
    with patch('midpoint.goal_cli.get_current_branch', return_value="goal-S1-test"):
        # Mock get_goal_id_from_branch
        with patch('midpoint.goal_cli.get_goal_id_from_branch', return_value="S1"):
            # Mock get_parent_goal_id
            with patch('midpoint.goal_cli.get_parent_goal_id', return_value="G1"):
                # Mock find_branch_for_goal
                with patch('midpoint.goal_cli.find_branch_for_goal', return_value="goal-G1-test"):
                    assert go_to_parent_goal() is True
                    
                    # Check that git checkout was called correctly
                    mock_subprocess_run.assert_called_with(
                        ["git", "checkout", "goal-G1-test"],
                        check=True,
                        capture_output=True,
                        text=True
                    )


def test_go_to_child(git_repo, mock_subprocess_run):
    """Test going to child goal."""
    # Mock find_branch_for_goal
    with patch('midpoint.goal_cli.find_branch_for_goal', return_value="goal-S1-test"):
        assert go_to_child("S1") is True
        
        # Check that git checkout was called correctly
        mock_subprocess_run.assert_called_with(
            ["git", "checkout", "goal-S1-test"],
            check=True,
            capture_output=True,
            text=True
        )


def test_go_to_root_goal(git_repo, mock_subprocess_run):
    """Test going to root goal."""
    # Mock get_current_branch
    with patch('midpoint.goal_cli.get_current_branch', return_value="goal-S2-test"):
        # Mock get_goal_id_from_branch
        with patch('midpoint.goal_cli.get_goal_id_from_branch', return_value="S2"):
            # Mock get_parent_goal_id
            with patch('midpoint.goal_cli.get_parent_goal_id', side_effect=["S1", "G1", ""]):
                # Mock find_branch_for_goal
                with patch('midpoint.goal_cli.find_branch_for_goal', return_value="goal-G1-test"):
                    assert go_to_root_goal() is True
                    
                    # Check that git checkout was called correctly
                    mock_subprocess_run.assert_called_with(
                        ["git", "checkout", "goal-G1-test"],
                        check=True,
                        capture_output=True,
                        text=True
                    )


def test_list_subgoals(git_repo, capsys):
    """Test listing subgoals."""
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Mock get_current_branch
        with patch('midpoint.goal_cli.get_current_branch', return_value="goal-G1-test"):
            # Mock get_goal_id_from_branch
            with patch('midpoint.goal_cli.get_goal_id_from_branch', return_value="G1"):
                subgoals = list_subgoals()
                
                # Check output
                captured = capsys.readouterr()
                assert "Children of G1:" in captured.out
                assert "S1" in captured.out
                
                # Check returned data
                assert len(subgoals) == 1
                assert "S1" in subgoals 