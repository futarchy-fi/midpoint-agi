"""
Tests for goal navigation commands.
"""

import os
import json
import pytest
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from midpoint.goal_cli import (
    go_back_commits,
    reset_to_commit,
    create_checkpoint,
    list_checkpoints,
    go_to_parent_goal,
    go_to_subgoal,
    go_to_root_goal,
    list_subgoals,
    get_goal_id_from_branch,
    get_parent_goal_id,
    find_branch_for_goal
)

class GitRepoFixture:
    """Class to manage a test git repository."""
    
    def __init__(self, tmp_path):
        self.repo_path = tmp_path / "test_repo"
        self.repo_path.mkdir()
        self.goal_dir = self.repo_path / ".goal"
        self.goal_dir.mkdir()
        self.checkpoints_dir = self.repo_path / ".goal" / "checkpoints"
        self.checkpoints_dir.mkdir()
        
        # Initialize git repository
        os.chdir(self.repo_path)
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
        
        # Create initial commit
        (self.repo_path / "README.md").write_text("# Test Repository")
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], check=True, capture_output=True)
        
        # Save initial hash
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        self.initial_hash = result.stdout.strip()
        
        # Create goal branches
        self._create_goal_branch("G1", "First goal")
        self._create_goal_branch("G1-S1", "First subgoal", parent="G1")
        self._create_goal_branch("G1-S1-S1", "Nested subgoal", parent="G1-S1")
        self._create_goal_branch("G2", "Second goal")
    
    def _create_goal_branch(self, goal_id, description, parent=None):
        """Create a goal branch for testing."""
        # Create goal file
        goal_file = self.goal_dir / f"{goal_id}.json"
        goal_content = {
            "goal_id": goal_id,
            "description": description,
            "parent_goal": parent or "",
            "timestamp": "20250324_000000"
        }
        goal_file.write_text(json.dumps(goal_content))
        
        # Create branch
        branch_name = f"goal-{goal_id}-test"
        subprocess.run(["git", "checkout", "-b", branch_name], check=True, capture_output=True)
        
        # Create a commit on this branch
        (self.repo_path / f"{goal_id}.txt").write_text(f"Content for {goal_id}")
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"Add {goal_id}"], check=True, capture_output=True)
        
        # Save hash
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        setattr(self, f"{goal_id.replace('-', '_')}_hash", result.stdout.strip())
        setattr(self, f"{goal_id.replace('-', '_')}_branch", branch_name)
    
    def cleanup(self):
        """Clean up the test repository."""
        os.chdir(Path.cwd().parent)  # Move out of the repo directory
        shutil.rmtree(self.repo_path)


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


def test_get_goal_id_from_branch():
    """Test extracting goal ID from branch name."""
    assert get_goal_id_from_branch("goal-G1-test") == "G1"
    assert get_goal_id_from_branch("goal-G1-S1-test") == "G1-S1"
    assert get_goal_id_from_branch("not-a-goal-branch") is None
    assert get_goal_id_from_branch("goal-invalid") is None


def test_get_parent_goal_id(git_repo):
    """Test getting parent goal ID."""
    with patch('midpoint.goal_cli.GOAL_DIR', git_repo.goal_dir):
        # Test with a subgoal
        assert get_parent_goal_id("G1-S1") == "G1"
        
        # Test with a nested subgoal
        assert get_parent_goal_id("G1-S1-S1") == "G1-S1"
        
        # Test with a top-level goal
        assert get_parent_goal_id("G1") == ""
        
        # Test with non-existent goal
        assert get_parent_goal_id("non-existent") is None


def test_find_branch_for_goal(git_repo, mock_subprocess_run):
    """Test finding branch for a goal ID."""
    mock_subprocess_run.return_value.stdout = """
  goal-G1-test
* goal-G1-S1-test
  goal-G1-S1-S1-test
  goal-G2-test
"""
    assert find_branch_for_goal("G1") == "goal-G1-test"
    assert find_branch_for_goal("G1-S1") == "goal-G1-S1-test"
    assert find_branch_for_goal("non-existent") is None


def test_go_back_commits(git_repo, mock_subprocess_run):
    """Test going back commits."""
    # Mock get_current_branch
    with patch('midpoint.goal_cli.get_current_branch', return_value="goal-G1-test"):
        # Mock get_current_hash
        with patch('midpoint.goal_cli.get_current_hash', side_effect=["hash1", "hash2"]):
            assert go_back_commits(1) is True
            # Check that git reset was called correctly
            mock_subprocess_run.assert_called_with(
                ["git", "reset", "--hard", "HEAD~1"],
                check=True,
                capture_output=True,
                text=True
            )


def test_reset_to_commit(git_repo, mock_subprocess_run):
    """Test resetting to a specific commit."""
    # Mock get_current_branch
    with patch('midpoint.goal_cli.get_current_branch', return_value="goal-G1-test"):
        # Mock get_current_hash
        with patch('midpoint.goal_cli.get_current_hash', side_effect=["hash1", "hash2"]):
            # Mock git cat-file
            mock_subprocess_run.return_value.stdout = "commit"
            
            assert reset_to_commit("abcd1234") is True
            
            # Check that git reset was called correctly
            mock_subprocess_run.assert_called_with(
                ["git", "reset", "--hard", "abcd1234"],
                check=True,
                capture_output=True,
                text=True
            )


def test_create_checkpoint(git_repo):
    """Test creating a checkpoint."""
    # Patch CHECKPOINT_DIR
    with patch('midpoint.goal_cli.CHECKPOINT_DIR', str(git_repo.checkpoints_dir)):
        # Mock get_current_branch and get_current_hash
        with patch('midpoint.goal_cli.get_current_branch', return_value="goal-G1-test"):
            with patch('midpoint.goal_cli.get_current_hash', return_value="hash1234"):
                assert create_checkpoint("Test checkpoint") is True
                
                # Check that checkpoint file was created
                checkpoint_files = list(git_repo.checkpoints_dir.glob("*.json"))
                assert len(checkpoint_files) == 1
                
                # Check checkpoint content
                with open(checkpoint_files[0], 'r') as f:
                    data = json.load(f)
                    assert data["message"] == "Test checkpoint"
                    assert data["git_hash"] == "hash1234"
                    assert data["branch"] == "goal-G1-test"


def test_list_checkpoints(git_repo, capsys):
    """Test listing checkpoints."""
    # Create test checkpoints
    checkpoint_data = {
        "timestamp": "20250324_000000",
        "message": "Test checkpoint",
        "git_hash": "hash1234",
        "branch": "goal-G1-test"
    }
    checkpoint_file = git_repo.checkpoints_dir / "20250324_000000_Test_checkpoint.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    
    # Patch CHECKPOINT_DIR
    with patch('midpoint.goal_cli.CHECKPOINT_DIR', str(git_repo.checkpoints_dir)):
        checkpoints = list_checkpoints()
        
        # Check output
        captured = capsys.readouterr()
        assert "Checkpoints:" in captured.out
        assert "Test checkpoint" in captured.out
        assert "goal-G1-test" in captured.out
        assert "hash1234" in captured.out
        
        # Check returned data
        assert len(checkpoints) == 1
        assert checkpoints[0]["message"] == "Test checkpoint"


def test_go_to_parent_goal(git_repo, mock_subprocess_run):
    """Test going to parent goal."""
    # Mock get_current_branch
    with patch('midpoint.goal_cli.get_current_branch', return_value="goal-G1-S1-test"):
        # Mock get_goal_id_from_branch
        with patch('midpoint.goal_cli.get_goal_id_from_branch', return_value="G1-S1"):
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


def test_go_to_subgoal(git_repo, mock_subprocess_run):
    """Test going to a specific subgoal."""
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Mock find_branch_for_goal
        with patch('midpoint.goal_cli.find_branch_for_goal', return_value="goal-G1-S1-test"):
            assert go_to_subgoal("G1-S1") is True
            
            # Check that git checkout was called correctly
            mock_subprocess_run.assert_called_with(
                ["git", "checkout", "goal-G1-S1-test"],
                check=True,
                capture_output=True,
                text=True
            )


def test_go_to_root_goal(git_repo, mock_subprocess_run):
    """Test going to the root goal."""
    # Mock get_current_branch
    with patch('midpoint.goal_cli.get_current_branch', return_value="goal-G1-S1-S1-test"):
        # Mock get_goal_id_from_branch
        with patch('midpoint.goal_cli.get_goal_id_from_branch', return_value="G1-S1-S1"):
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
                assert "Subgoals for G1:" in captured.out
                assert "G1-S1" in captured.out
                
                # Check returned data
                assert len(subgoals) == 1
                assert "G1-S1" in subgoals 