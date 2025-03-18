"""
Data models for the Midpoints AGI system.

This module defines the core data structures used throughout the Midpoints AGI system,
including representations of states, goals, strategies, and execution records.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class State:
    """
    Represents a state of the world that can be modified by actions.
    
    In repository-based tasks, this typically represents a specific Git commit.
    """
    git_hash: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Goal:
    """
    Represents a goal to be achieved by the system.
    
    A goal includes a description and validation criteria that can be used
    to determine whether the goal has been achieved.
    """
    description: str
    validation_criteria: List[str]
    success_threshold: float = 0.8  # 80% of criteria must be met by default
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPlan:
    """
    A plan for achieving a goal, consisting of steps to be executed.
    
    Each step can be executed by the TaskExecutor agent and should move
    the system closer to achieving the goal.
    """
    steps: List[str]
    reasoning: str
    estimated_points: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    The result of validating a goal against the current state.
    
    Includes overall success status and detailed results for each criterion.
    """
    success: bool
    criterion_results: Dict[str, bool]
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureAnalysis:
    """
    Analysis of why a strategy failed to achieve a goal.
    
    Includes a diagnosis of the problem and suggestions for improvement.
    """
    diagnosis: str
    root_causes: List[str]
    improvement_suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    """
    A record of a single execution step in the problem-solving process.
    
    Used for debugging, analysis, and transparency.
    """
    agent: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    points_consumed: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionSummary:
    """
    A summary of the execution process, typically for a completed iteration.
    
    Provides a concise overview of what was accomplished and what remains to be done.
    """
    accomplished: List[str]
    remaining_work: List[str]
    key_insights: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetAllocation:
    """
    Allocation of the points budget across different stages of problem-solving.
    
    Used to manage resource usage throughout the execution process.
    """
    planning: int
    execution: int
    validation: int
    analysis: int
    summarization: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskContext:
    """
    Context information for a task execution.
    
    Includes the current state, goal, and any relevant metadata.
    """
    state: State
    goal: Goal
    iteration: int
    points_consumed: int
    total_budget: int
    execution_history: List[ExecutionTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict) 