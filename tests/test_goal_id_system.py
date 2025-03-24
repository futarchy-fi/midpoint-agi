import os
import json
import pytest
from pathlib import Path
from midpoint.agents.goal_decomposer import GoalDecomposer
from midpoint.agents.models import Goal, State, TaskContext

@pytest.fixture
def goal_decomposer():
    """Create a GoalDecomposer instance for testing."""
    return GoalDecomposer()

@pytest.fixture
def test_context():
    """Create a test context."""
    return TaskContext(
        state=State(
            repository_path="test_repo",
            git_hash="test_hash",
            description="Test state"
        ),
        goal=Goal(
            description="Test goal",
            validation_criteria=["Test passes"]
        ),
        iteration=0,
        execution_history=[]
    )

def test_generate_goal_id(goal_decomposer, tmp_path):
    """Test goal ID generation."""
    # Create a temporary logs directory
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    
    # Test top-level goal ID generation
    goal_id = goal_decomposer.generate_goal_id(logs_dir=str(logs_dir))
    assert goal_id == "G1"
    
    # Create a goal file
    goal_file = logs_dir / f"{goal_id}.json"
    goal_file.write_text(json.dumps({"goal_id": goal_id}))
    
    # Test subgoal ID generation
    subgoal_id = goal_decomposer.generate_goal_id(goal_id, logs_dir=str(logs_dir))
    assert subgoal_id == f"{goal_id}-S1"
    
    # Create a subgoal file
    subgoal_file = logs_dir / f"{subgoal_id}.json"
    subgoal_file.write_text(json.dumps({"goal_id": subgoal_id}))
    
    # Test another subgoal ID generation
    subgoal_id2 = goal_decomposer.generate_goal_id(goal_id, logs_dir=str(logs_dir))
    assert subgoal_id2 == f"{goal_id}-S2"

def test_create_top_goal_file(goal_decomposer, test_context, tmp_path):
    """Test top-level goal file creation."""
    # Create a temporary logs directory
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    
    # Create a top-level goal file
    filename = goal_decomposer.create_top_goal_file(test_context, logs_dir=str(logs_dir))
    
    # Verify the file was created
    goal_file = logs_dir / filename
    assert goal_file.exists()
    
    # Verify the file contents
    data = json.loads(goal_file.read_text())
    assert data["goal_id"].startswith("G")
    assert data["parent_goal"] == ""
    assert data["description"] == test_context.goal.description
    assert data["next_step"] == test_context.goal.description
    assert data["validation_criteria"] == test_context.goal.validation_criteria
    assert data["requires_further_decomposition"] is True
    assert data["iteration"] == test_context.iteration

def test_list_subgoal_files(goal_decomposer, tmp_path):
    """Test listing subgoal files."""
    # Create a temporary logs directory
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    
    # Create some test goal files
    goals = [
        {"goal_id": "G1", "next_step": "First goal", "parent_goal": "", "timestamp": "20250324_000001"},
        {"goal_id": "G1-S1", "next_step": "First subgoal", "parent_goal": "G1.json", "timestamp": "20250324_000002"},
        {"goal_id": "G2", "next_step": "Second goal", "parent_goal": "", "timestamp": "20250324_000003"}
    ]
    
    for goal in goals:
        goal_file = logs_dir / f"{goal['goal_id']}.json"
        goal_file.write_text(json.dumps(goal))
    
    # List the files
    files = goal_decomposer.list_subgoal_files(str(logs_dir))
    
    # Verify the results
    assert len(files) == 3
    file_ids = [f[4] for f in files]  # goal_id is the 5th element
    assert "G1" in file_ids
    assert "G1-S1" in file_ids
    assert "G2" in file_ids 