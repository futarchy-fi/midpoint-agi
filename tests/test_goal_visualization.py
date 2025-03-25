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
    # Create files with the new flat ID format
    test_goals = [
        {"goal_id": "G1", "description": "First goal", "parent_goal": "", "complete": True},
        {"goal_id": "S1", "description": "First subgoal", "parent_goal": "G1"},
        {"goal_id": "S2", "description": "Second subgoal", "parent_goal": "S1"},
        {"goal_id": "G2", "description": "Second goal", "parent_goal": ""}
    ]
    
    # Recreate goal files with new format
    for goal_data in test_goals:
        goal_file = git_repo.goal_dir / f"{goal_data['goal_id']}.json"
        with open(goal_file, 'w') as f:
            json.dump(goal_data, f)
    
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Show goal tree
        show_goal_tree()
        
        # Check output
        captured = capsys.readouterr()
        assert "Goal Tree:" in captured.out
        assert "✅ G1:" in captured.out  # Complete goal
        assert "S1:" in captured.out  # Subgoal
        assert "S2:" in captured.out  # Nested subgoal
        assert "G2:" in captured.out  # Another top-level goal
        assert "Status Legend:" in captured.out


def test_show_goal_history(git_repo, capsys):
    """Test showing goal history."""
    # Create files with the new flat ID format
    test_goals = [
        {
            "goal_id": "G1", 
            "description": "First goal", 
            "parent_goal": "", 
            "complete": True,
            "completion_time": "20250324_000000",
            "timestamp": "20250324_000000",
            "merged_subgoals": [
                {
                    "subgoal_id": "S1",
                    "merge_time": "20250324_000000",
                    "merge_commit": "hash1234"
                }
            ]
        },
        {
            "goal_id": "S1", 
            "description": "First subgoal", 
            "parent_goal": "G1",
            "timestamp": "20250324_000000"
        },
        {
            "goal_id": "G2", 
            "description": "Second goal", 
            "parent_goal": "",
            "timestamp": "20250324_000000"
        }
    ]
    
    # Recreate goal files with new format
    for goal_data in test_goals:
        goal_file = git_repo.goal_dir / f"{goal_data['goal_id']}.json"
        with open(goal_file, 'w') as f:
            json.dump(goal_data, f)
    
    # Patch GOAL_DIR
    with patch('midpoint.goal_cli.GOAL_DIR', str(git_repo.goal_dir)):
        # Show goal history
        show_goal_history()
        
        # Check output
        captured = capsys.readouterr()
        assert "Goal History:" in captured.out
        assert "G1: First goal" in captured.out
        assert "S1: First subgoal" in captured.out
        assert "Completed:" in captured.out
        assert "Merged: S1" in captured.out
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