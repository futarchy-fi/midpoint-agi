#!/usr/bin/env python
"""
Midpoint Orchestrator - Main workflow coordinator for the Midpoint system.

IMPORTANT: This module implements a generic orchestration system that coordinates the workflow
between different agents. It MUST NOT contain any task-specific logic or hardcoded implementations.
All task-specific decisions should be handled by the specialized agents.
"""

import asyncio
import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import argparse

from .agents.models import State, Goal, TaskContext, SubgoalPlan, ExecutionResult, ValidationResult
from .agents.goal_decomposer import GoalDecomposer, validate_repository_state
from .agents.task_executor import TaskExecutor
from .agents.goal_validator import GoalValidator
from .agents.tools import get_current_hash, check_repo_state, checkout_branch
from .utils.logging import log_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Orchestrator')

@dataclass
class OrchestrationResult:
    """Result of running the orchestrator."""
    success: bool
    final_state: Optional[State] = None
    error_message: Optional[str] = None
    execution_history: List[Dict[str, Any]] = None

class Orchestrator:
    """
    Orchestrates the execution of goals through iterative decomposition and execution.
    
    The Orchestrator manages the workflow between:
    - GoalDecomposer: For breaking down complex goals
    - TaskExecutor: For implementing concrete tasks
    - GoalValidator: For verifying successful execution
    
    It handles the state transitions between agents and ensures that the repository
    remains in a consistent state throughout the process.
    """
    
    def __init__(self):
        """Initialize the Orchestrator with its component agents."""
        self.decomposer = GoalDecomposer()
        self.executor = TaskExecutor()
        self.validator = GoalValidator()
        logger.info("Orchestrator initialized")
    
    async def run(self, repo_path: str, goal: Goal, max_iterations: int = 10, 
              start_iteration: int = 0, checkpoint_path: Optional[str] = None) -> OrchestrationResult:
        """
        Run the full orchestration workflow for a given goal.
        
        Args:
            repo_path: Path to the git repository
            goal: The high-level goal to achieve
            max_iterations: Maximum number of iterations to attempt
            start_iteration: Iteration to start from (for resuming)
            checkpoint_path: Path to save checkpoints (if specified)
            
        Returns:
            OrchestrationResult with the outcome of the orchestration
        """
        logger.info(f"Starting orchestration for goal: {goal.description}")
        logger.info(f"Repository path: {repo_path}")
        
        # Verify initial repository state
        repo_state = await check_repo_state(repo_path)
        if not repo_state["is_clean"]:
            logger.error("Repository has uncommitted changes. Aborting.")
            return OrchestrationResult(
                success=False,
                error_message="Repository has uncommitted changes. Please commit or stash changes first."
            )
        
        # Get initial git hash
        initial_hash = await get_current_hash(repo_path)
        logger.info(f"Initial git hash: {initial_hash}")
        
        # Start a new logging session
        session = log_manager.start_session(
            repository_path=repo_path,
            git_hash=initial_hash,
            goal_description=goal.description
        )
        
        # Write headers for all log files
        log_manager.write_log_header("goal_hierarchy")
        log_manager.write_log_header("execution")
        
        # Initialize state
        current_state = State(
            git_hash=initial_hash,
            repository_path=repo_path,
            description="Initial state before orchestration"
        )
        
        # Initialize execution history
        execution_history = []
        
        # Main orchestration loop
        for iteration in range(start_iteration, max_iterations):
            logger.info(f"Starting iteration {iteration+1}/{max_iterations}")
            
            try:
                # 1. Decompose the goal to find the next executable task
                logger.info("Decomposing goal to find next executable task")
                subgoal_plan = await self.decomposer.determine_next_step(TaskContext(
                    state=current_state,
                    goal=goal,
                    iteration=iteration,
                    execution_history=execution_history
                ))
                
                if not subgoal_plan:
                    logger.error("Failed to decompose goal. No executable task found.")
                    return OrchestrationResult(
                        success=False,
                        final_state=current_state,
                        error_message="Failed to decompose goal. No executable task found.",
                        execution_history=execution_history
                    )
                
                logger.info(f"Found executable task: {subgoal_plan.next_step}")
                
                # 2. Execute the task
                logger.info(f"Executing task: {subgoal_plan.next_step}")
                
                # Create task context
                task_context = TaskContext(
                    state=State(
                        git_hash=current_state.git_hash,
                        repository_path=repo_path,
                        description=f"State before executing: {subgoal_plan.next_step}"
                    ),
                    goal=Goal(
                        description=subgoal_plan.next_step,
                        validation_criteria=subgoal_plan.validation_criteria,
                        success_threshold=0.8
                    ),
                    iteration=iteration,
                    execution_history=execution_history,
                    metadata={}
                )
                
                # Execute the task
                execution_result = await self.executor.execute_task(task_context, subgoal_plan.next_step)
                
                if not execution_result.success:
                    logger.error(f"Task execution failed: {execution_result.error_message}")
                    return OrchestrationResult(
                        success=False,
                        final_state=current_state,
                        error_message=f"Task execution failed: {execution_result.error_message}",
                        execution_history=execution_history
                    )
                
                logger.info(f"Task executed successfully. New git hash: {execution_result.git_hash}")
                
                # 3. Validate the execution
                logger.info("Validating task execution")
                subgoal = Goal(
                    description=subgoal_plan.next_step,
                    validation_criteria=subgoal_plan.validation_criteria,
                    success_threshold=0.8
                )
                validation_result = await self.validator.validate_execution(
                    goal=subgoal,
                    execution_result=execution_result
                )
                
                # 4. Record the execution
                execution_entry = {
                    "iteration": iteration,
                    "subgoal": subgoal_plan.next_step,
                    "git_hash": execution_result.git_hash,
                    "branch_name": execution_result.branch_name,
                    "validation_score": validation_result.score,
                    "execution_time": execution_result.execution_time,
                    "timestamp": time.time()
                }
                execution_history.append(execution_entry)
                
                # Log the execution result
                log_manager.log_execution_result(
                    iteration=iteration,
                    subgoal=subgoal_plan.next_step,
                    git_hash=execution_result.git_hash,
                    branch_name=execution_result.branch_name,
                    validation_score=validation_result.score,
                    execution_time=execution_result.execution_time
                )
                
                logger.info(f"Validation score: {validation_result.score:.2f}")
                
                # 5. Update current state
                current_state = State(
                    git_hash=execution_result.git_hash,
                    repository_path=repo_path,
                    description=f"State after iteration {iteration+1}"
                )
                
                # Save checkpoint if path is specified
                if checkpoint_path:
                    await self.save_checkpoint(
                        checkpoint_path,
                        current_state,
                        goal,
                        execution_history,
                        iteration + 1  # Save the next iteration to start from
                    )
                
                # 6. Check if the main goal is achieved
                logger.info("Checking if main goal is achieved")
                main_goal_validation = await self.validator.validate_execution(
                    goal=goal,
                    execution_result=execution_result
                )
                
                if main_goal_validation.success:
                    logger.info(f"Main goal achieved with score: {main_goal_validation.score:.2f}")
                    return OrchestrationResult(
                        success=True,
                        final_state=current_state,
                        execution_history=execution_history
                    )
                
                logger.info(f"Main goal not yet achieved. Continuing with next iteration.")
                
            except Exception as e:
                logger.error(f"Error during orchestration: {str(e)}")
                return OrchestrationResult(
                    success=False,
                    final_state=current_state,
                    error_message=f"Error during orchestration: {str(e)}",
                    execution_history=execution_history
                )
        
        # If we reach here, we've hit the maximum iterations
        logger.warning(f"Reached maximum iterations ({max_iterations}) without achieving goal")
        return OrchestrationResult(
            success=False,
            final_state=current_state,
            error_message=f"Reached maximum iterations ({max_iterations}) without achieving goal",
            execution_history=execution_history
        )

    async def save_checkpoint(self, checkpoint_path, current_state, goal, execution_history, iteration):
        """
        Save the current orchestration state to a checkpoint file.
        
        Args:
            checkpoint_path: Path to save the checkpoint
            current_state: Current state object
            goal: Current goal
            execution_history: History of execution so far
            iteration: Current iteration
        """
        checkpoint_data = {
            "state": {
                "git_hash": current_state.git_hash,
                "repository_path": current_state.repository_path,
                "description": current_state.description
            },
            "goal": {
                "description": goal.description,
                "validation_criteria": goal.validation_criteria,
                "success_threshold": goal.success_threshold
            },
            "execution_history": execution_history,
            "iteration": iteration,
            "timestamp": time.time()
        }
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    async def load_checkpoint(self, checkpoint_path):
        """
        Load orchestration state from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Tuple of (current_state, goal, execution_history, iteration)
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file {checkpoint_path} not found")
            return None
            
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                
            current_state = State(
                git_hash=checkpoint_data["state"]["git_hash"],
                repository_path=checkpoint_data["state"]["repository_path"],
                description=checkpoint_data["state"]["description"]
            )
            
            goal = Goal(
                description=checkpoint_data["goal"]["description"],
                validation_criteria=checkpoint_data["goal"]["validation_criteria"],
                success_threshold=checkpoint_data["goal"]["success_threshold"]
            )
            
            execution_history = checkpoint_data["execution_history"]
            iteration = checkpoint_data["iteration"]
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            logger.info(f"Resuming from iteration {iteration}")
            
            return current_state, goal, execution_history, iteration
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None

async def run_orchestration(
    repo_path: str,
    goal: Goal,
    max_iterations: int = 10,
    start_iteration: int = 0,
    checkpoint_path: Optional[str] = None
) -> OrchestrationResult:
    """Run the orchestration workflow."""
    orchestrator = Orchestrator()
    return await orchestrator.run(
        repo_path=repo_path,
        goal=goal,
        max_iterations=max_iterations,
        start_iteration=start_iteration,
        checkpoint_path=checkpoint_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run orchestration")
    parser.add_argument("repo_path", help="Path to the repository")
    parser.add_argument("goal", help="Goal description")
    parser.add_argument("--criteria", nargs="+", required=True, help="Validation criteria")
    parser.add_argument("--threshold", type=float, default=0.8, help="Success threshold")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--checkpoint", help="Checkpoint path")
    
    args = parser.parse_args()
    
    # Create goal object
    goal_obj = Goal(
        description=args.goal,
        validation_criteria=args.criteria,
        success_threshold=args.threshold
    )
    
    # Run orchestration
    result = asyncio.run(run_orchestration(
        repo_path=args.repo_path,
        goal=goal_obj,
        max_iterations=args.iterations,
        checkpoint_path=args.checkpoint
    ))
    
    # Print result as JSON
    print(json.dumps({
        "success": result.success,
        "final_state": {
            "git_hash": result.final_state.git_hash if result.final_state else None,
            "repository_path": result.final_state.repository_path if result.final_state else None,
            "description": result.final_state.description if result.final_state else None
        },
        "error_message": result.error_message,
        "execution_history": result.execution_history
    }, indent=2)) 