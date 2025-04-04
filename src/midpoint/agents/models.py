"""
Data models for the Midpoint system.

This module defines the core data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

@dataclass
class State:
    """Represents the current state of the repository."""
    repository_path: str
    description: str
    git_hash: Optional[str] = None  # Made optional
    branch_name: Optional[str] = None  # Current branch name
    memory_hash: Optional[str] = None  # Hash of the current memory state
    memory_repository_path: Optional[str] = None  # Path to the memory repository

@dataclass
class Goal:
    """Represents a goal to be achieved."""
    description: str
    validation_criteria: List[str] = field(default_factory=list)  # Make optional with default empty list
    success_threshold: float = 0.8

class SubgoalPlan(BaseModel):
    """Represents the next step toward achieving a goal."""
    reasoning: str  # Required for both complete and incomplete goals
    goal_completed: bool = False  # Whether the goal is complete
    completion_summary: Optional[str] = None  # Summary of what was accomplished (for completed goals)
    next_step: str
    validation_criteria: List[str]
    can_be_decomposed: bool = True  # Flag to indicate if more decomposition is needed
    relevant_context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_goal: Optional[str] = None  # Reference to the parent goal file
    goal_id: Optional[str] = None  # Unique identifier for this goal
    timestamp: Optional[str] = None  # When this goal was created
    iteration: int = 0  # Current iteration in the overall process

@dataclass
class MemoryState:
    """Represents a memory repository state."""
    memory_hash: str
    repository_path: str

@dataclass
class StrategyPlan:
    """Represents a plan for achieving a goal."""
    steps: List[str]
    reasoning: str
    metadata: Dict[str, Any]

@dataclass
class TaskContext:
    """Contains all context for a task, including state, goal, and execution history."""
    state: State
    goal: Goal
    memory_state: MemoryState  # Required memory state
    iteration: int = 0
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadata for tracking decomposition context

@dataclass
class ExecutionTrace:
    """Represents the execution trace of a task."""
    task_description: str
    actions_performed: List[Dict[str, Any]]  # List of actions with their details
    tool_calls: List[Dict[str, Any]]  # List of tool calls with their details
    resulting_state: State  # The state after execution
    execution_time: float  # Time taken to execute the task
    success: bool  # Whether the execution was successful
    branch_name: str  # The git branch where execution occurred
    error_message: Optional[str] = None  # Error message if execution failed

@dataclass
class ExecutionResult:
    """Represents the result of a task execution."""
    success: bool  # Whether the execution was successful
    branch_name: str  # The git branch where execution occurred
    git_hash: str  # The git hash after execution
    error_message: Optional[str] = None  # Error message if execution failed
    execution_time: float = 0.0  # Time taken to execute the task
    repository_path: Optional[str] = None  # Path to the repository
    validation_results: List[str] = field(default_factory=list)  # Results of validation steps
    final_state: Optional['State'] = None  # Final state after execution (git_hash, memory_hash, etc.)

@dataclass
class ValidationResult:
    """Represents the result of goal validation."""
    success: bool  # Whether the validation passed
    score: float  # Score between 0 and 1
    reasoning: str  # Explanation of the validation result
    criteria_results: List[Dict[str, Any]]  # Results for each validation criterion
    git_hash: str  # The git hash that was validated
    branch_name: str  # The branch that was validated 