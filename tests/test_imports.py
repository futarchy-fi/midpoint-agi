#!/usr/bin/env python3
"""
Test file to verify that the imports work correctly.
"""

import sys
import os
import pytest
from pathlib import Path

# Add the parent directory to the path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all the necessary imports work correctly."""
    from agents.models import (
        State, Goal, StrategyPlan, FailureAnalysis, ValidationResult,
        ExecutionTrace, ExecutionSummary, BudgetAllocation, TaskContext
    )
    
    from agents.core_agents import (
        goal_decomposer, task_executor, goal_validator,
        failure_analyzer, progress_summarizer
    )
    
    from agents.orchestrator import GoalTracer, solve_problem
    
    # Create some test instances to verify the data models
    state = State(
        git_hash="test_hash",
        description="Test state"
    )
    
    goal = Goal(
        description="Test goal",
        validation_criteria=["Criterion 1", "Criterion 2"]
    )
    
    # Verify that the objects are created correctly
    assert state.git_hash == "test_hash"
    assert state.description == "Test state"
    assert goal.description == "Test goal"
    assert len(goal.validation_criteria) == 2
    assert goal.success_threshold == 0.8  # Default value
    
    print("All imports work correctly!")

if __name__ == "__main__":
    test_imports()
    print("Test completed successfully!") 