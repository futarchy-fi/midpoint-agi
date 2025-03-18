"""
Midpoint - An advanced AGI system for recursive goal decomposition.
"""

from .models import (
    State,
    Goal,
    StrategyPlan,
    TaskContext,
    ExecutionResult,
    ValidationResult
)

from .config import (
    get_openai_api_key,
    get_openai_org_id,
    get_anthropic_api_key,
    get_points_budget
)

from .goal_decomposer import GoalDecomposer
from .task_executor import TaskExecutor
from .validator import Validator
from .tools import (
    check_repo_state,
    create_branch,
    create_commit,
    get_current_hash,
    track_points
)

__all__ = [
    # Models
    'State',
    'Goal',
    'StrategyPlan',
    'TaskContext',
    'ExecutionResult',
    'ValidationResult',
    
    # Configuration
    'get_openai_api_key',
    'get_openai_org_id',
    'get_anthropic_api_key',
    'get_points_budget',
    
    # Agents
    'GoalDecomposer',
    'TaskExecutor',
    'Validator',
    
    # Tools
    'check_repo_state',
    'create_branch',
    'create_commit',
    'get_current_hash',
    'track_points'
] 