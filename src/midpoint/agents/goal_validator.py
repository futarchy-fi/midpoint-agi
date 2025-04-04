"""
Goal Validation agent implementation.

IMPORTANT: This module implements a generic goal validation system that uses LLM to validate
task execution results. It MUST NOT contain any task-specific logic or hardcoded validation rules.
All validation decisions should be made by the LLM at runtime.
"""

import asyncio
from typing import List, Dict, Any, Optional
import re
import os
import random
import json
import logging
from pathlib import Path
import datetime

from .models import Goal, ExecutionResult, ValidationResult
from .tools import (
    get_current_hash,
    list_directory,
    read_file,
    search_code,
    run_terminal_cmd,
    get_current_branch,
    web_search,
    web_scrape
)
# Import validate_repository_state from goal_decomposer
from .goal_decomposer import validate_repository_state
from .tools.processor import ToolProcessor
from .tools.registry import ToolRegistry
from .config import get_openai_api_key
from openai import AsyncOpenAI

# Set up logging
logger = logging.getLogger('GoalValidator')

class GoalValidator:
    """
    Generic goal validation agent that uses LLM to validate execution results.
    
    This class MUST:
    - Remain completely task-agnostic
    - Not contain any hardcoded validation rules
    - Delegate all validation decisions to the LLM
    - Use the provided tools to gather evidence for validation

    As outlined in the VISION.md document, this validator:
    1. Evaluates whether a subgoal has been successfully achieved
    2. Provides a success score from 0.0 to 1.0
    3. Determines whether the result is acceptable
    4. Identifies specific areas for improvement
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the GoalValidator agent."""
        # Initialize OpenAI client with API key from config
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        if not api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
            
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Store the model name
        self.model = model
        
        # Initialize tool processor
        self.tool_processor = ToolProcessor(self.client)
        
        # System prompt for the LLM that will make all validation decisions
        self.system_prompt = """You are a goal validation agent responsible for evaluating execution results.
Your role is to:
1. Check if execution was successful
2. Validate changes against goal criteria
3. Provide detailed reasoning for validation decisions
4. Calculate a validation score

IMPORTANT: You must be thorough and objective in your validation. Your job is to ensure that each validation criterion 
has been properly satisfied. Be specific in your reasoning and cite concrete evidence rather than making general statements.

For each validation criterion:
1. Use the available tools to gather evidence
2. Analyze the evidence against the criterion
3. Make a judgment about whether the criterion is satisfied
4. Provide clear reasoning for your decision

Your response must be in JSON format with these fields:
{
    "criteria_results": [
        {
            "criterion": "string",
            "passed": boolean,
            "reasoning": "string",
            "evidence": ["string"]
        }
    ],
    "overall_score": float,  # Between 0 and 1
    "overall_reasoning": "string"
}"""

        # Get tools from registry
        self.tools = ToolRegistry.get_tool_schemas()

    async def validate_execution(self, goal: Goal, execution_result: ExecutionResult) -> ValidationResult:
        """
        Validate an execution result against a goal using LLM.
        
        This method MUST NOT contain any task-specific validation logic.
        All validation decisions should be made by the LLM.
        
        Args:
            goal: The goal to validate against
            execution_result: The result of task execution
            
        Returns:
            ValidationResult containing the validation outcome
        """
        # If execution failed, validation fails
        if not execution_result.success:
            return ValidationResult(
                success=False,
                score=0.0,
                reasoning="Execution failed: " + (execution_result.error_message or "Unknown error"),
                criteria_results=[],
                git_hash=execution_result.git_hash,
                branch_name=execution_result.branch_name
            )
        
        # Validate repository state
        try:
            await validate_repository_state(
                execution_result.repository_path,
                execution_result.git_hash
            )
        except ValueError as e:
            # Repository state validation can fail if we're not on the right branch
            # Let's try to check out the branch first and then validate
            pass
        
        # Check which branch we're on
        current_branch = await get_current_branch(execution_result.repository_path)
        
        # Switch to the execution branch if needed
        if current_branch != execution_result.branch_name:
            try:
                await run_terminal_cmd(
                    command=["git", "checkout", execution_result.branch_name],
                    cwd=execution_result.repository_path
                )
            except Exception as e:
                return ValidationResult(
                    success=False,
                    score=0.0,
                    reasoning=f"Failed to checkout branch {execution_result.branch_name}: {str(e)}",
                    criteria_results=[],
                    git_hash=execution_result.git_hash,
                    branch_name=execution_result.branch_name
                )
        
        try:
            # Get repository information
            try:
                repository_files = await list_directory(execution_result.repository_path, ".")
                if isinstance(repository_files, dict) and "files" in repository_files:
                    repository_files = repository_files["files"]
                else:
                    repository_files = []
            except Exception:
                repository_files = []
            
            # Prepare the context for the LLM
            context = f"""
Repository: {execution_result.repository_path}
Branch: {execution_result.branch_name}
Git Hash: {execution_result.git_hash}

Goal: {goal.description}

Validation Criteria:
{chr(10).join([f"{i+1}. {criterion}" for i, criterion in enumerate(goal.validation_criteria)])}

Success Threshold: {goal.success_threshold}
"""
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ]
            
            # Get validation from LLM using tool processor
            logger.info(f"Validating goal: {goal.description}")
            logger.info(f"Number of criteria: {len(goal.validation_criteria)}")
            
            # Call LLM with tools
            response, tool_calls = await self.tool_processor.run_llm_with_tools(
                messages=messages,
                model=self.model,
                validate_json_format=True
            )
            
            # Extract the response content
            if isinstance(response, list):
                # If we got a list of messages, get the last one's content
                content = response[-1]["content"] if response else "{}"
            else:
                # If we got a single message object
                content = response.get("content", "{}") if isinstance(response, dict) else response.content
            
            # Parse the validation result
            try:
                validation_data = json.loads(content)
                criteria_results = validation_data.get("criteria_results", [])
                overall_score = validation_data.get("overall_score", 0.0)
                overall_reasoning = validation_data.get("overall_reasoning", "No reasoning provided")
                
                # Calculate success based on threshold
                success = overall_score >= goal.success_threshold
                
                # Create ValidationResult
                return ValidationResult(
                    success=success,
                    score=overall_score,
                    reasoning=overall_reasoning,
                    criteria_results=criteria_results,
                    git_hash=execution_result.git_hash,
                    branch_name=execution_result.branch_name
                )
            except json.JSONDecodeError:
                # If we can't parse the response, create a fallback result
                logger.error(f"Failed to parse LLM response as JSON: {content[:100]}...")
                
                # Generate fallback criteria results
                criteria_results = []
                for criterion in goal.validation_criteria:
                    criteria_results.append({
                        "criterion": criterion,
                        "passed": False,
                        "reasoning": "Failed to parse LLM response",
                        "evidence": ["Invalid response format"]
                    })
                
                return ValidationResult(
                    success=False,
                    score=0.0,
                    reasoning="Failed to parse validation response from LLM",
                    criteria_results=criteria_results,
                    git_hash=execution_result.git_hash,
                    branch_name=execution_result.branch_name
                )
            
        except Exception as e:
            logger.error(f"Error during goal validation: {str(e)}")
            return ValidationResult(
                success=False,
                score=0.0,
                reasoning=f"Validation failed due to error: {str(e)}",
                criteria_results=[],
                git_hash=execution_result.git_hash,
                branch_name=execution_result.branch_name
            )
        finally:
            # Always switch back to main branch
            try:
                await run_terminal_cmd(
                    command=["git", "checkout", "main"],
                    cwd=execution_result.repository_path
                )
            except:
                pass
    
    def _generate_criterion_reasoning(self, criterion: str, passed: bool, evidence: List[str]) -> str:
        """Generate detailed reasoning for a single criterion validation."""
        if passed:
            return f"Criterion satisfied: {criterion}\nEvidence: {'; '.join(evidence)}"
        else:
            return f"Criterion not satisfied: {criterion}\nEvidence: {'; '.join(evidence)}"
    
    def _generate_reasoning(self, criteria_results: List[Dict[str, Any]], 
                          score: float, threshold: float) -> str:
        """Generate a human-readable reasoning for the validation result."""
        passed_count = sum(1 for result in criteria_results if result["passed"])
        total_count = len(criteria_results)
        
        reasoning = []
        reasoning.append(f"Validation {'passed' if score >= threshold else 'failed'} with score {score:.2f}/{threshold:.2f}")
        reasoning.append(f"Satisfied {passed_count}/{total_count} criteria")
        
        # Add details for failed criteria
        if passed_count < total_count:
            reasoning.append("\nFailed criteria:")
            for result in criteria_results:
                if not result["passed"]:
                    reasoning.append(f"- {result['criterion']}")
                    reasoning.append(f"  Reason: {result['reasoning']}")
        
        return "\n".join(reasoning)

    async def validate_goal(self, goal_id: str, repository_path: str = None, auto: bool = True) -> ValidationResult:
        """
        Validate a goal by ID.
        
        This is a convenience method that:
        1. Loads the goal data from the goal file
        2. Creates a Goal object with the validation criteria
        3. Creates an ExecutionResult for the current state
        4. Calls validate_execution
        
        Args:
            goal_id: The ID of the goal to validate
            repository_path: Optional repository path (defaults to current directory)
            auto: Whether this is an automated validation
            
        Returns:
            ValidationResult containing the validation outcome
        """
        # Use current directory if no repository path provided
        if not repository_path:
            repository_path = os.getcwd()
            
        # Load goal file
        from midpoint.goal_cli import ensure_goal_dir
        goal_path = ensure_goal_dir()
        goal_file = goal_path / f"{goal_id}.json"
        
        if not goal_file.exists():
            raise ValueError(f"Goal not found: {goal_id}")
            
        # Load goal data
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
            
        # Get validation criteria
        criteria = goal_data.get("validation_criteria", [])
        if not criteria:
            raise ValueError(f"No validation criteria found for goal {goal_id}")
            
        # Create Goal object
        goal = Goal(
            description=goal_data.get("description", ""),
            validation_criteria=criteria,
            success_threshold=goal_data.get("success_threshold", 0.8)
        )
        
        # Get current git state
        git_hash = await get_current_hash(repository_path)
        branch_name = await get_current_branch(repository_path)
        
        # Create ExecutionResult
        execution_result = ExecutionResult(
            success=True,
            branch_name=branch_name,
            git_hash=git_hash,
            repository_path=repository_path
        )
        
        # Validate execution
        return await self.validate_execution(goal, execution_result) 