"""
Goal Validation agent implementation.

IMPORTANT: This module implements a generic goal validation system that uses LLM to validate
task execution results. It MUST NOT contain any task-specific logic or hardcoded validation rules.
All validation decisions should be made by the LLM at runtime.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import os
import random
from pathlib import Path
import subprocess

from openai import AsyncOpenAI

from midpoint.agents.models import Goal, ExecutionResult, CriterionResult, ValidationResult, State
from midpoint.agents.tools.git_tools import get_current_hash, get_current_branch
from midpoint.agents.tools.memory_tools import get_memory_diff
from .tools import (
    list_directory,
    read_file,
    search_code,
    run_terminal_cmd,
    web_search,
    web_scrape
)
# Import validate_repository_state from goal_decomposer
from .goal_decomposer import validate_repository_state
from .tools.processor import ToolProcessor
from .tools.registry import ToolRegistry
from .config import get_openai_api_key

# Set up logging
logger = logging.getLogger(__name__)

# System prompt for the validator
VALIDATION_SYSTEM_PROMPT = """
You are the Goal Validator, an expert at verifying whether a goal's validation criteria have been met.

Your task is to analyze the evidence provided and determine if each validation criterion has been met.
You should look for concrete evidence in the repository changes and any other information provided.

Be precise and objective in your assessment. Clearly explain your reasoning for each criterion.
You should provide specific evidence from the diffs or other sources to support your conclusions.

Your output will be used to determine if the overall goal has been successfully completed.
"""

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
1. Use the available tools to gather evidence about the changes
2. Analyze the repository and memory diffs to find relevant changes
3. Examine how the changes relate to the specific validation criterion
4. Determine if the changes satisfy the criterion
5. Provide clear reasoning with specific references to parts of the diffs

When analyzing diffs:
- Look for file additions, modifications, and deletions
- Check file content changes and their relationship to the criteria
- Examine memory documents that were added or modified
- Consider both the quantity and quality of changes

Focus on the SPECIFIC CHANGES that occurred from initial to final state, not just the final state in isolation.
Your validation must explicitly reference evidence from the diffs when available.

Your response must be in JSON format with these fields:
{
    "criteria_results": [
        {
            "criterion": "string",
            "passed": boolean,
            "reasoning": "string",
            "evidence": ["string"]  // Specific references to parts of the diffs
        }
    ],
    "overall_score": float,  // Between 0 and 1
    "overall_reasoning": "string"
}"""

        # Get tools from registry
        self.tools = ToolRegistry.get_tool_schemas()

    async def validate_execution(self, goal: Goal, execution_result: ExecutionResult) -> ValidationResult:
        """
        Validate execution results against a goal using LLM.
        
        This method takes a goal and execution result and validates the execution
        results against the goal using an LLM. This method handles repository
        state validation and content validation.
        
        Args:
            goal: The goal to validate
            execution_result: The execution result to validate
        
        Returns:
            ValidationResult object containing validation details
        """
        # If the execution failed, we don't need to validate
        if not execution_result.success:
            return ValidationResult(
                goal_id=goal.id,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                criteria_results=[],
                score=0.0,
                validated_by="System",
                automated=True,
                repository_state=None,
            )
        
        # Validate repository state
        repo_info = {}
        try:
            current_branch = get_current_branch(execution_result.repository_path)
            if current_branch != execution_result.branch_name:
                logging.info(f"Current branch {current_branch} does not match execution branch {execution_result.branch_name}")
                # Try to switch to correct branch using subprocess
                try:
                    proc = subprocess.run(
                        ["git", "checkout", execution_result.branch_name],
                        cwd=execution_result.repository_path,
                        capture_output=True,
                        text=True
                    )
                    if proc.returncode == 0:
                        logging.info(f"Switched to branch {execution_result.branch_name}")
                    else:
                        raise ValueError(f"Failed to checkout branch: {proc.stderr}")
                except Exception as e:
                    logging.error(f"Failed to switch to branch {execution_result.branch_name}: {e}")
                    # Return failed validation
                    return ValidationResult(
                        goal_id=goal.id,
                        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        criteria_results=[],
                        score=0.0,
                        validated_by="System",
                        automated=True,
                        repository_state=None,
                    )
        
            # Get repository info
            current_hash = get_current_hash(execution_result.repository_path)
            repo_info = {
                "current_hash": current_hash,
                "current_branch": current_branch,
                "timestamp": str(datetime.now().isoformat()),
                "goal_id": goal.id,
            }
        except Exception as e:
            logging.error(f"Failed to get repository info: {e}")
        
        # Get any diffs between initial and final states
        repo_diff = None
        try:
            if (hasattr(goal, 'initial_state') and goal.initial_state and 
                hasattr(goal.initial_state, 'git_hash') and goal.initial_state.git_hash):
                initial_hash = goal.initial_state.git_hash
                # Using a different approach to get the diff since get_diff is not available
                proc = subprocess.run(
                    ["git", "diff", initial_hash, current_hash],
                    cwd=execution_result.repository_path,
                    capture_output=True,
                    text=True
                )
                if proc.returncode == 0:
                    repo_diff = proc.stdout
        except Exception as e:
            logging.error(f"Failed to get repo diff: {e}")
        
        memory_diff = None
        try:
            if (hasattr(goal, 'initial_state') and goal.initial_state and 
                hasattr(goal.initial_state, 'memory_hash') and goal.initial_state.memory_hash and
                hasattr(goal, 'current_state') and goal.current_state and
                hasattr(goal.current_state, 'memory_hash') and goal.current_state.memory_hash):
                memory_diff = get_memory_diff(goal.initial_state.memory_hash, 
                                            goal.current_state.memory_hash,
                                            execution_result.repository_path)
        except Exception as e:
            logging.error(f"Failed to get memory diff: {e}")
        
        # Prepare context for LLM
        messages = [
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"""
You are tasked with validating whether the execution results match the goal criteria.

### Goal
{goal.description}

### Validation Criteria
{json.dumps(goal.validation_criteria, indent=2)}

### Evidence
""" + (f"""
#### Repository Changes
```diff
{repo_diff}
```
""" if repo_diff else "") + (f"""
#### Memory Changes
```diff
{memory_diff}
```
""" if memory_diff else "") + """

Provide a detailed validation, analyzing each criterion to determine if it has been met.
Your response MUST be a valid JSON object with the following structure:
{
  "criteria_results": [
    {
      "criterion": "The first criterion text",
      "passed": true or false,
      "reasoning": "Explanation of why the criterion is passed or failed",
      "evidence": ["Supporting evidence from the diffs or execution"]
    },
    // ... for each criterion
  ]
}
"""
            }
        ]
        
        try:
            logging.info(f"Validating goal {goal.id} with LLM")
            # Run LLM with tools
            response, tool_calls = await self.tool_processor.run_llm_with_tools(
                messages=messages,
                model=self.model
            )
            
            # Extract response from LLM
            if response and len(response) > 0:
                assistant_message = next((msg for msg in response if msg["role"] == "assistant"), None)
                if assistant_message:
                    content = assistant_message.get("content", "")
                    if content:
                        try:
                            # Try to extract JSON from the response
                            validation_data = self._extract_validation_json(content)
                            
                            # Create validation result
                            criteria_results = []
                            for result in validation_data.get("criteria_results", []):
                                criteria_results.append(
                                    CriterionResult(
                                        criterion=result["criterion"],
                                        passed=result["passed"],
                                        reasoning=result["reasoning"],
                                        evidence=result["evidence"]
                                    )
                                )
                            
                            # Calculate score
                            num_passed = sum(1 for cr in criteria_results if cr.passed)
                            total_criteria = len(criteria_results)
                            score = (num_passed / total_criteria) * 100 if total_criteria > 0 else 0
                            
                            return ValidationResult(
                                goal_id=goal.id,
                                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                                criteria_results=criteria_results,
                                score=score,
                                validated_by="LLM",
                                automated=True,
                                repository_state=repo_info,
                            )
                        except Exception as e:
                            logging.error(f"Failed to parse LLM response as JSON: {content[:500]}...")
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
                subprocess.run(
                    ["git", "checkout", "main"],
                    cwd=execution_result.repository_path,
                    capture_output=True,
                    text=True
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

    async def validate_goal(self, goal_path: str, repository_path: str = ".") -> ValidationResult:
        """
        Validate a goal using a dummy execution result.
        
        Args:
            goal_path: Path to the goal JSON file
            repository_path: Path to the repository
            
        Returns:
            ValidationResult object containing validation details
        """
        try:
            # Load goal data
            with open(goal_path, 'r') as f:
                goal_data = json.load(f)
            
            # Extract initial and current state information
            initial_state = goal_data.get('initial_state', {})
            current_state = goal_data.get('current_state', {})
            
            initial_git_hash = initial_state.get('git_hash')
            initial_memory_hash = initial_state.get('memory_hash')
            initial_timestamp = initial_state.get('timestamp')
            
            current_git_hash = current_state.get('git_hash')
            current_memory_hash = current_state.get('memory_hash')
            current_timestamp = current_state.get('timestamp')
            
            # Log hashes for debugging
            logging.debug(f"Initial hash: {initial_git_hash}")
            logging.debug(f"Current hash: {current_git_hash}")
            
            # Create Goal object
            goal = Goal(
                id=goal_data.get('goal_id', ''),
                description=goal_data.get('description', ''),
                validation_criteria=goal_data.get('validation_criteria', []),
                success_threshold=80.0,  # Default threshold
                initial_state=State(
                    git_hash=initial_git_hash,
                    memory_hash=initial_memory_hash,
                    timestamp=initial_timestamp
                ),
                current_state=State(
                    git_hash=current_git_hash,
                    memory_hash=current_memory_hash,
                    timestamp=current_timestamp
                )
            )
            
            # Create dummy execution result
            execution_result = ExecutionResult(
                success=True,
                repository_path=repository_path,
                branch_name=get_current_branch(repository_path),
                git_hash=current_git_hash,
                task_id='',
                goal_id=goal_data.get('goal_id', ''),
                error_message=None
            )
            
            # Validate execution
            return await self.validate_execution(goal, execution_result)
        except Exception as e:
            logging.error(f"Error validating goal: {e}")
            raise

    def _extract_validation_json(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON validation data from LLM response.
        
        Args:
            content: Raw content from LLM response
            
        Returns:
            Parsed JSON data as a dictionary
        """
        # Try direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, content)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try finding JSON objects using regex
        json_object_pattern = r'{[\s\S]*}'
        matches = re.findall(json_object_pattern, content)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If all attempts fail, raise an exception
        raise ValueError("Could not extract valid JSON from LLM response") 