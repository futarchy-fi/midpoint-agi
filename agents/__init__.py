"""
Midpoints AGI agent orchestration system.

This package contains the core components of the Midpoints AGI system,
including agent models, orchestration logic, and specialized agents for
complex problem solving.
"""

from .models import (
    State, 
    Goal, 
    StrategyPlan,
    FailureAnalysis,
    ValidationResult,
    ExecutionTrace,
    ExecutionSummary,
    BudgetAllocation,
    TaskContext
)

from .core_agents import (
    goal_decomposer,
    task_executor,
    goal_validator,
    failure_analyzer,
    progress_summarizer,
    decompose_goal,
    execute_strategy,
    validate_goal,
    analyze_failure,
    summarize_progress
)

from .tools import (
    git_commit,
    git_checkout,
    read_file,
    write_file,
    run_command,
    list_directory,
    track_points
)

from .orchestrator import solve_problem

__all__ = [
    'State',
    'Goal',
    'StrategyPlan',
    'FailureAnalysis',
    'ValidationResult',
    'ExecutionTrace',
    'ExecutionSummary',
    'BudgetAllocation',
    'TaskContext',
    'solve_problem',
] 