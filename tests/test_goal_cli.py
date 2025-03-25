"""
Tests for the goal CLI commands.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from midpoint.goal_cli import (
    ensure_goal_dir,
    generate_goal_id,
    create_goal_file,
    create_new_goal,
    create_new_subgoal,
    list_goals
)

@pytest.fixture
def goal_dir(tmp_path):
    """Create a temporary goal directory for testing."""
    test_goal_dir = tmp_path / ".goal"
    test_goal_dir.mkdir()
    
    # Save and restore the original GOAL_DIR
    import midpoint.goal_cli
    original_goal_dir = midpoint.goal_cli.GOAL_DIR
    midpoint.goal_cli.GOAL_DIR = str(test_goal_dir)
    
    yield test_goal_dir
    
    # Restore the original GOAL_DIR
    midpoint.goal_cli.GOAL_DIR = original_goal_dir


def test_ensure_goal_dir(goal_dir):
    """Test ensuring the goal directory exists."""
    path = ensure_goal_dir()
    assert path.exists()
    assert path == goal_dir


def test_generate_goal_id(goal_dir):
    """Test generating goal IDs."""
    # Test top-level goal ID
    goal_id = generate_goal_id()
    assert goal_id == "G1"
    
    # Create a goal file
    (goal_dir / f"{goal_id}.json").write_text("{}")
    
    # Test another top-level goal ID
    goal_id2 = generate_goal_id()
    assert goal_id2 == "G2"
    
    # Test subgoal ID (now returns S1 instead of G1-S1)
    subgoal_id = generate_goal_id(goal_id)
    assert subgoal_id == "S1"
    
    # Create a subgoal file
    (goal_dir / f"{subgoal_id}.json").write_text("{}")
    
    # Test another subgoal ID
    subgoal_id2 = generate_goal_id(goal_id)
    assert subgoal_id2 == "S2"
    
    # Test task ID
    task_id = generate_goal_id(goal_id, is_task=True)
    assert task_id == "T1"


def test_create_goal_file(goal_dir):
    """Test creating a goal file."""
    goal_id = "G1"
    description = "Test goal"
    
    filename = create_goal_file(goal_id, description)
    
    # Check file exists
    goal_file = goal_dir / f"{goal_id}.json"
    assert goal_file.exists()
    
    # Check file contents
    data = json.loads(goal_file.read_text())
    assert data["goal_id"] == goal_id
    assert data["description"] == description
    assert data["parent_goal"] == ""


def test_create_new_goal(goal_dir, capsys):
    """Test creating a new goal."""
    description = "Test new goal"
    
    with patch('midpoint.goal_cli.generate_goal_id', return_value="G1"):
        goal_id = create_new_goal(description)
    
    # Check output
    captured = capsys.readouterr()
    assert f"Created new goal {goal_id}" in captured.out
    
    # Check file exists
    goal_file = goal_dir / f"{goal_id}.json"
    assert goal_file.exists()


def test_create_new_subgoal(goal_dir, capsys):
    """Test creating a new subgoal."""
    parent_id = "G1"
    description = "Test subgoal"
    
    # Create parent goal file
    parent_file = goal_dir / f"{parent_id}.json"
    parent_file.write_text(json.dumps({"goal_id": parent_id, "description": "Parent goal"}))
    
    with patch('midpoint.goal_cli.generate_goal_id', return_value=f"{parent_id}-S1"):
        subgoal_id = create_new_subgoal(parent_id, description)
    
    # Check output
    captured = capsys.readouterr()
    assert f"Created new subgoal {subgoal_id} under {parent_id}" in captured.out
    
    # Check file exists
    subgoal_file = goal_dir / f"{subgoal_id}.json"
    assert subgoal_file.exists()
    
    # Check file contents
    data = json.loads(subgoal_file.read_text())
    assert data["goal_id"] == subgoal_id
    assert data["description"] == description
    assert data["parent_goal"] == parent_id


def test_list_goals(goal_dir, capsys):
    """Test listing goals."""
    # Create test goal files
    goals = [
        {"goal_id": "G1", "description": "First goal", "parent_goal": ""},
        {"goal_id": "G1-S1", "description": "First subgoal", "parent_goal": "G1"},
        {"goal_id": "G2", "description": "Second goal", "parent_goal": ""}
    ]
    
    for goal in goals:
        goal_file = goal_dir / f"{goal['goal_id']}.json"
        goal_file.write_text(json.dumps(goal))
    
    # List the goals
    list_goals()
    
    # Check output
    captured = capsys.readouterr()
    assert "Goal Tree:" in captured.out
    assert "• G1: First goal" in captured.out
    assert "  • G1-S1: First subgoal" in captured.out
    assert "• G2: Second goal" in captured.out 