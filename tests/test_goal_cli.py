"""
Tests for the goal CLI commands.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from midpoint.goal_cli import (
    ensure_goal_dir,
    generate_goal_id,
    create_goal_file,
    create_new_goal,
    create_new_subgoal,
    list_goals,
    execute_task
)
from midpoint.agents.models import ExecutionResult

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
    
    # Test with uncommitted changes
    with patch('subprocess.run') as mock_run:
        # Mock git status to show changes
        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
                return type('Result', (), {'stdout': 'main\n', 'returncode': 0})
            elif cmd == ["git", "status", "--porcelain"]:
                return type('Result', (), {'stdout': 'M test.txt', 'returncode': 0})
            elif cmd == ["git", "stash", "push", "-m", "Stashing changes before creating new goal"]:
                return type('Result', (), {'stdout': '', 'returncode': 0})
            elif cmd == ["git", "checkout", "-b", "goal-G1"]:
                return type('Result', (), {'stdout': 'Switched to a new branch', 'returncode': 0})
            elif cmd == ["git", "rev-parse", "HEAD"]:
                return type('Result', (), {'stdout': 'abc123\n', 'returncode': 0})
            elif cmd == ["git", "checkout", "main"]:
                return type('Result', (), {'stdout': '', 'returncode': 0})
            elif cmd == ["git", "stash", "pop"]:
                return type('Result', (), {'stdout': '', 'returncode': 0})
            return type('Result', (), {'stdout': '', 'returncode': 0})
        
        mock_run.side_effect = mock_run_side_effect
        
        with patch('midpoint.goal_cli.generate_goal_id', return_value="G1"):
            goal_id = create_new_goal(description)
        
        # Check output
        captured = capsys.readouterr()
        assert f"Created new goal {goal_id}" in captured.out
        assert "Created branch: goal-G1" in captured.out
        
        # Check file exists
        goal_file = goal_dir / f"{goal_id}.json"
        assert goal_file.exists()
        
        # Check file contents
        data = json.loads(goal_file.read_text())
        assert data["goal_id"] == goal_id
        assert data["description"] == description
        assert data["branch_name"] == "goal-G1"
        assert data["is_task"] is False
        assert data["requires_further_decomposition"] is True
        
        # Verify git commands were called in the correct order
        assert len(mock_run.call_args_list) == 7
        assert mock_run.call_args_list[0][0][0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        assert mock_run.call_args_list[1][0][0] == ["git", "status", "--porcelain"]
        assert mock_run.call_args_list[2][0][0] == ["git", "stash", "push", "-m", "Stashing changes before creating new goal"]
        assert mock_run.call_args_list[3][0][0] == ["git", "checkout", "-b", "goal-G1"]
        assert mock_run.call_args_list[4][0][0] == ["git", "rev-parse", "HEAD"]
        assert mock_run.call_args_list[5][0][0] == ["git", "checkout", "main"]
        assert mock_run.call_args_list[6][0][0] == ["git", "stash", "pop"]
    
    # Test successful creation without changes
    with patch('subprocess.run') as mock_run:
        # Mock git status to show no changes
        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
                return type('Result', (), {'stdout': 'main\n', 'returncode': 0})
            elif cmd == ["git", "status", "--porcelain"]:
                return type('Result', (), {'stdout': '', 'returncode': 0})
            elif cmd == ["git", "checkout", "-b", "goal-G2"]:
                return type('Result', (), {'stdout': 'Switched to a new branch', 'returncode': 0})
            elif cmd == ["git", "rev-parse", "HEAD"]:
                return type('Result', (), {'stdout': 'def456\n', 'returncode': 0})
            elif cmd == ["git", "checkout", "main"]:
                return type('Result', (), {'stdout': '', 'returncode': 0})
            return type('Result', (), {'stdout': '', 'returncode': 0})
        
        mock_run.side_effect = mock_run_side_effect
        
        with patch('midpoint.goal_cli.generate_goal_id', return_value="G2"):
            goal_id = create_new_goal(description)
        
        # Check output
        captured = capsys.readouterr()
        assert f"Created new goal {goal_id}" in captured.out
        assert "Created branch: goal-G2" in captured.out
        
        # Check file exists
        goal_file = goal_dir / f"{goal_id}.json"
        assert goal_file.exists()
        
        # Check file contents
        data = json.loads(goal_file.read_text())
        assert data["goal_id"] == goal_id
        assert data["description"] == description
        assert data["branch_name"] == "goal-G2"
        assert data["is_task"] is False
        assert data["requires_further_decomposition"] is True
        
        # Verify git commands were called in the correct order
        assert len(mock_run.call_args_list) == 5
        assert mock_run.call_args_list[0][0][0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        assert mock_run.call_args_list[1][0][0] == ["git", "status", "--porcelain"]
        assert mock_run.call_args_list[2][0][0] == ["git", "checkout", "-b", "goal-G2"]
        assert mock_run.call_args_list[3][0][0] == ["git", "rev-parse", "HEAD"]
        assert mock_run.call_args_list[4][0][0] == ["git", "checkout", "main"]


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


@pytest.mark.asyncio
async def test_execute_task(goal_dir, capsys):
    """Test executing a task."""
    # Create test task file
    task_id = "G1-S1-T1"
    task_file = goal_dir / f"{task_id}.json"
    
    # Create top-level goal file
    top_level_id = "G1"
    top_level_file = goal_dir / f"{top_level_id}.json"
    
    # Write test data
    with open(task_file, 'w') as f:
        json.dump({
            "goal_id": task_id,
            "description": "Test task",
            "parent_goal": "G1-S1",
            "timestamp": "20250324_000000",
            "is_task": True,
            "requires_further_decomposition": False,
            "initial_state": {
                "git_hash": "abcdef123456",
                "repository_path": "/test/repo",
                "description": "Initial state",
                "timestamp": "20250324_000000"
            }
        }, f)
    
    with open(top_level_file, 'w') as f:
        json.dump({
            "goal_id": top_level_id,
            "description": "Test goal",
            "branch_name": "goal-G1",
            "timestamp": "20250324_000000"
        }, f)
    
    # Mock git commands
    with patch('subprocess.run') as mock_run, patch('midpoint.goal_cli.get_current_branch') as mock_get_branch:
        # Mock current branch
        mock_get_branch.return_value = "main"
        
        # Mock git status to show no changes
        mock_run.side_effect = [
            # git status
            type('Result', (), {'stdout': '', 'returncode': 0}),
            # git checkout goal-G1
            type('Result', (), {'stdout': 'Switched to branch goal-G1', 'returncode': 0}),
            # git checkout original branch
            type('Result', (), {'stdout': 'Switched to branch main', 'returncode': 0})
        ]
        
        # Mock TaskExecutor
        with patch('midpoint.goal_cli.TaskExecutor') as mock_executor:
            # Setup mock executor
            mock_executor_instance = AsyncMock()
            mock_executor.return_value = mock_executor_instance
            mock_result = ExecutionResult(
                success=True,
                branch_name="goal-G1",
                git_hash="abcdef123456",
                error_message=None,
                execution_time=0.5,
                repository_path="/test/repo",
                validation_results=["Task completed successfully"]
            )
            mock_executor_instance.execute_task.return_value = mock_result

            # Run the function
            result = await execute_task(task_id)
            
            # Assertions
            assert result is True
            mock_executor_instance.execute_task.assert_called_once()
            
            # Check that git commands were called correctly
            assert mock_run.call_count == 3
            assert mock_run.call_args_list[0][0][0] == ["git", "status", "--porcelain"]
            assert mock_run.call_args_list[1][0][0] == ["git", "checkout", "goal-G1"]
            assert mock_run.call_args_list[2][0][0] == ["git", "checkout", "main"]
            
            # Check output
            captured = capsys.readouterr()
            assert "Task G1-S1-T1 executed successfully" in captured.out
            assert "Task completed successfully" in captured.out 