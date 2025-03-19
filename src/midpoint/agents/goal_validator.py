import asyncio
from typing import List, Dict, Any

from .models import Goal, ExecutionResult, ValidationResult
from .tools import (
    get_current_hash,
    validate_repository_state,
    list_directory,
    read_file,
    search_code,
    run_terminal_cmd
)

class GoalValidator:
    """Agent responsible for validating execution results against goals."""
    
    def __init__(self):
        self.system_prompt = """You are a goal validation agent responsible for evaluating execution results.
Your role is to:
1. Check if execution was successful
2. Validate changes against goal criteria
3. Provide detailed reasoning for validation decisions
4. Calculate a validation score

Available tools:
- list_directory: List contents of a directory
- read_file: Read contents of a file
- search_code: Search for code patterns
- run_terminal_cmd: Run a terminal command

Always validate against the specific criteria provided in the goal.
Provide clear reasoning for your validation decisions."""

    async def validate_execution(self, goal: Goal, execution_result: ExecutionResult) -> ValidationResult:
        """
        Validate an execution result against a goal.
        
        Args:
            goal: The goal to validate against
            execution_result: The result of task execution
            
        Returns:
            ValidationResult containing the validation outcome
            
        Raises:
            ValueError: If repository validation fails
            Exception: For other errors during validation
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
        await validate_repository_state(
            execution_result.git_hash,
            execution_result.branch_name
        )
        
        # Switch to the execution branch
        await run_terminal_cmd(
            command=f"git checkout {execution_result.branch_name}",
            cwd=execution_result.git_hash  # Using git_hash as repository path for now
        )
        
        try:
            # Evaluate each criterion
            criteria_results = []
            total_score = 0.0
            
            for criterion in goal.validation_criteria:
                # Use tools to evaluate the criterion
                # This is where we'd use the LLM to evaluate the criterion
                # For now, we'll just track the criterion
                criteria_results.append({
                    "criterion": criterion,
                    "passed": True,  # This would be determined by the LLM
                    "reasoning": "Criterion evaluation would go here"
                })
                total_score += 1.0
            
            # Calculate final score
            score = total_score / len(goal.validation_criteria)
            success = score >= goal.success_threshold
            
            # Generate reasoning
            reasoning = self._generate_reasoning(criteria_results, score, goal.success_threshold)
            
            return ValidationResult(
                success=success,
                score=score,
                reasoning=reasoning,
                criteria_results=criteria_results,
                git_hash=execution_result.git_hash,
                branch_name=execution_result.branch_name
            )
            
        finally:
            # Always switch back to main branch
            try:
                await run_terminal_cmd(
                    command="git checkout main",
                    cwd=execution_result.git_hash
                )
            except:
                pass
    
    def _generate_reasoning(self, criteria_results: List[Dict[str, Any]], 
                          score: float, threshold: float) -> str:
        """Generate a human-readable reasoning for the validation result."""
        if score >= threshold:
            return f"Validation passed with score {score:.2f}/{threshold:.2f}"
        else:
            return f"Validation failed with score {score:.2f}/{threshold:.2f}" 