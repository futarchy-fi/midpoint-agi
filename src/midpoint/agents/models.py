"""
Data models for the Midpoint system.

This module defines the core data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class State:
    """Represents a specific state of the repository."""
    git_hash: str
    description: str
    repository_path: Optional[str] = None

@dataclass
class Goal:
    """Represents a goal to be achieved."""
    description: str
    validation_criteria: List[str]
    success_threshold: float = 0.8

@dataclass
class SubgoalPlan:
    """Represents the next step toward achieving a goal."""
    next_step: str
    validation_criteria: List[str]
    reasoning: str
    requires_further_decomposition: bool = True  # Flag to indicate if more decomposition is needed
    relevant_context: Dict[str, Any] = field(default_factory=dict)  # Context to pass to child goals
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyPlan:
    """Represents a plan for achieving a goal."""
    steps: List[str]
    reasoning: str
    estimated_points: int
    metadata: Dict[str, Any]

@dataclass
class TaskContext:
    """Represents the context for a task execution."""
    state: State
    goal: Goal
    iteration: int
    points_consumed: int
    total_budget: int
    execution_history: List[Dict[str, Any]] 