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
        Run the orchestration process to achieve a goal.
        
        Args:
            repo_path: Path to the repository to work with
            goal: The goal to achieve
            max_iterations: Maximum number of iterations to run
            start_iteration: Iteration to start from (for resuming)
            checkpoint_path: Path to save checkpoints to
            
        Returns:
            OrchestrationResult containing the final state and execution history
        """
        logger.info(f"Starting orchestration for goal: {goal.description}")
        logger.info(f"Repository: {repo_path}")
        logger.info(f"Max iterations: {max_iterations}")
        
        # Initialize state
        current_state = State(
            git_hash=await get_current_hash(repo_path),
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
                subgoal_plan = await self.decomposer.determine_next_state(TaskContext(
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
                        description=f"State before executing: {subgoal_plan.next_step}",
                        memory_hash=current_state.memory_hash,
                        memory_repository_path=current_state.memory_repository_path
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
                
                # Update current state with final state from execution result
                if execution_result.final_state:
                    current_state = execution_result.final_state
                    logger.info(f"Updated state - Git hash: {current_state.git_hash[:8]}, Memory hash: {current_state.memory_hash[:8] if current_state.memory_hash else 'None'}")
                
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
                
                # Save checkpoint if path provided
                if checkpoint_path:
                    self._save_checkpoint(checkpoint_path, current_state, execution_history)
                
                # Check if goal is complete
                if validation_result.score >= goal.success_threshold:
                    logger.info("Goal completed successfully!")
                    return OrchestrationResult(
                        success=True,
                        final_state=current_state,
                        execution_history=execution_history
                    )
                
            except Exception as e:
                logger.error(f"Error during orchestration: {str(e)}")
                return OrchestrationResult(
                    success=False,
                    final_state=current_state,
                    error_message=str(e),
                    execution_history=execution_history
                )
        
        # If we get here, we've hit the max iterations
        logger.warning(f"Reached maximum iterations ({max_iterations}) without completing goal")
        return OrchestrationResult(
            success=False,
            final_state=current_state,
            error_message=f"Maximum iterations ({max_iterations}) reached without completing goal",
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