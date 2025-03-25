"""
Tests for goal visualization tools.
"""

import os
import json
import pytest
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from midpoint.goal_cli import (
    show_goal_tree,
    show_goal_history,
    generate_graph,
    ensure_visualization_dir
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


def test_ensure_visualization_dir():
    """Test ensuring the visualization directory exists."""
    with patch('midpoint.goal_cli.VISUALIZATION_DIR', '.test_vis'):
        try:
            path = ensure_visualization_dir()
            assert path.exists()
            assert path.name == '.test_vis'
        finally:
            # Clean up
            if path.exists():
                path.rmdir()


def test_show_goal_tree(git_repo, capsys):
    """Test showing the goal tree."""
    # Add completion status to a goal
    goal_file = git_repo.goal_dir / "G1.json"
    with open(goal_file, 'r') as f:
        data = json.load(f)
    data["complete"] = True
    with open(goal_file, 'w') as f:
        json.dump(data, f)
    
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Show goal tree
        show_goal_tree()
        
        # Check output
        captured = capsys.readouterr()
        assert "Goal Tree:" in captured.out
        assert "✅ G1:" in captured.out  # Complete goal
        assert "G1-S1:" in captured.out  # Subgoal
        assert "G1-S1-S1:" in captured.out  # Nested subgoal
        assert "G2:" in captured.out  # Another top-level goal
        assert "Status Legend:" in captured.out


def test_show_goal_history(git_repo, capsys):
    """Test showing goal history."""
    # Add completion and timestamp info
    goal_file = git_repo.goal_dir / "G1.json"
    with open(goal_file, 'r') as f:
        data = json.load(f)
    data["complete"] = True
    data["completion_time"] = "20250324_000000"
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
        # Show goal history
        show_goal_history()
        
        # Check output
        captured = capsys.readouterr()
        assert "Goal History:" in captured.out
        assert "G1: First goal" in captured.out
        assert "G1-S1: First subgoal" in captured.out
        assert "Completed:" in captured.out
        assert "Merged: G1-S1" in captured.out
        assert "Summary:" in captured.out


def test_generate_graph_graphviz_not_installed():
    """Test generating a graph when Graphviz is not installed."""
    # Mock subprocess.run to simulate Graphviz not being installed
    with patch('subprocess.run', side_effect=FileNotFoundError):
        assert generate_graph() is False


def test_generate_graph(git_repo, mock_subprocess_run):
    """Test generating a graph."""
    # Create visualization directory
    vis_dir = git_repo.repo_path / ".goal" / "visualization"
    vis_dir.mkdir(exist_ok=True)
    
    # Patch directories and functions
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        with patch('midpoint.goal_cli.VISUALIZATION_DIR', str(vis_dir)):
            # Generate graph
            assert generate_graph() is True
            
            # Check that the correct commands were called
            # First check for Graphviz installation
            mock_subprocess_run.assert_any_call(
                ["dot", "-V"],
                check=True,
                capture_output=True
            )
            
            # Check that PDF generation was attempted
            pdf_calls = [call for call in mock_subprocess_run.call_args_list 
                        if len(call[0][0]) > 1 and call[0][0][0] == "dot" and call[0][0][1] == "-Tpdf"]
            assert len(pdf_calls) == 1
            
            # Check that PNG generation was attempted
            png_calls = [call for call in mock_subprocess_run.call_args_list 
                        if len(call[0][0]) > 1 and call[0][0][0] == "dot" and call[0][0][1] == "-Tpng"]
            assert len(png_calls) == 1
            
            # Check that DOT file was created
            dot_files = list(vis_dir.glob("*.dot"))
            assert len(dot_files) == 1 