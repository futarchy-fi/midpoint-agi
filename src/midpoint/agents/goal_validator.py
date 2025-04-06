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

YOUR RESPONSE MUST BE A VALID JSON OBJECT with this exact structure:
{
    "criteria_results": [
        {
            "criterion": "string",
            "passed": boolean,
            "reasoning": "string",
            "evidence": ["string"]
        }
    ],
    "score": float,
    "reasoning": "string"
}

Do not include any explanatory text before or after the JSON object.
Just output the JSON structure directly.
If you cannot properly validate, still return a valid JSON object with appropriate failure messages.
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
        
        # Store the last validation messages sent to the LLM
        self.last_validation_messages = None
        
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
            current_branch = await get_current_branch(execution_result.repository_path)
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
            current_hash = await get_current_hash(execution_result.repository_path)
            repo_info = {
                "git_hash": current_hash,
                "branch_name": current_branch,
                "repository_path": execution_result.repository_path,
                "description": "Goal validation state"
            }
        except Exception as e:
            logging.error(f"Failed to get repository info: {e}")
        
        # Get any diffs between initial and final states
        repo_diff = "No repository changes detected"
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
                    repo_diff = proc.stdout if proc.stdout.strip() else "No repository changes detected"
                    
                    # Auto-fail if repo diff is too large (over 100KB)
                    if len(repo_diff) > 100000:
                        logging.warning(f"Repository diff is too large ({len(repo_diff)} bytes). Auto-failing validation.")
                        failed_criteria = []
                        for criterion in goal.validation_criteria:
                            failed_criteria.append(
                                CriterionResult(
                                    criterion=criterion,
                                    passed=False,
                                    reasoning="Validation failed due to repository changes being too large to validate",
                                    evidence=["Repository diff exceeded maximum size (100KB)"]
                                )
                            )
                        return ValidationResult(
                            goal_id=goal.id,
                            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                            criteria_results=failed_criteria,
                            score=0.0,
                            validated_by="System",
                            automated=True,
                            repository_state=State(**repo_info) if repo_info else None,
                            reasoning="Repository changes were too large to validate"
                        )
        except Exception as e:
            logging.error(f"Failed to get repo diff: {e}")
            repo_diff = f"Error getting repository diff: {str(e)}"
        
        memory_diff_content = "No memory changes detected"
        memory_diff_files = []
        try:
            if (hasattr(goal, 'initial_state') and goal.initial_state and 
                hasattr(goal.initial_state, 'memory_hash') and goal.initial_state.memory_hash and
                hasattr(goal, 'current_state') and goal.current_state and
                hasattr(goal.current_state, 'memory_hash') and goal.current_state.memory_hash):
                memory_diff = await get_memory_diff(goal.initial_state.memory_hash, 
                                            goal.current_state.memory_hash,
                                            execution_result.repository_path)
                
                if memory_diff:
                    memory_diff_content = memory_diff.get('diff_content', 'No memory changes detected')
                    memory_diff_files = memory_diff.get('changed_files', [])
                    
                    # If memory hashes are different but no changes detected, something's wrong
                    # Add details about the changed files
                    if (memory_diff_content == "No memory changes detected" or not memory_diff_content.strip()) and \
                       goal.initial_state.memory_hash != goal.current_state.memory_hash:
                        logging.warning(f"Memory hashes differ ({goal.initial_state.memory_hash} vs {goal.current_state.memory_hash}) but no diff content found")
                        
                        # Get the content of any changed files
                        if memory_diff_files:
                            memory_repo_path = memory_diff.get('memory_repo_path', execution_result.repository_path)
                            additional_content = ["Memory hashes differ but standard diff shows no changes. Showing file contents instead:"]
                            
                            for file_path in memory_diff_files:
                                try:
                                    # Save current state
                                    current_branch = await get_current_branch(memory_repo_path)
                                    
                                    # Try to checkout the final hash to read files
                                    checkout_proc = subprocess.run(
                                        ["git", "checkout", goal.current_state.memory_hash],
                                        cwd=memory_repo_path,
                                        capture_output=True,
                                        text=True
                                    )
                                    
                                    if checkout_proc.returncode == 0:
                                        # Try to read the file content
                                        file_full_path = os.path.join(memory_repo_path, file_path)
                                        if os.path.exists(file_full_path):
                                            with open(file_full_path, 'r') as f:
                                                file_content = f.read()
                                            additional_content.append(f"\n--- FILE: {file_path} ---\n{file_content}")
                                    
                                    # Restore original branch
                                    subprocess.run(
                                        ["git", "checkout", current_branch],
                                        cwd=memory_repo_path,
                                        capture_output=True,
                                        text=True
                                    )
                                except Exception as file_error:
                                    logging.error(f"Error reading changed file {file_path}: {file_error}")
                                    additional_content.append(f"Error reading {file_path}: {str(file_error)}")
                        
                            if len(additional_content) > 1:  # If we have any file content
                                memory_diff_content = "\n".join(additional_content)
            
                    # Auto-fail if memory diff is too large (over 100KB)
                    if len(memory_diff_content) > 100000:
                        logging.warning(f"Memory diff is too large ({len(memory_diff_content)} bytes). Auto-failing validation.")
                        failed_criteria = []
                        for criterion in goal.validation_criteria:
                            failed_criteria.append(
                                CriterionResult(
                                    criterion=criterion,
                                    passed=False,
                                    reasoning="Validation failed due to memory changes being too large to validate",
                                    evidence=["Memory diff exceeded maximum size (100KB)"]
                                )
                            )
                        return ValidationResult(
                            goal_id=goal.id,
                            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                            criteria_results=failed_criteria,
                            score=0.0,
                            validated_by="System",
                            automated=True,
                            repository_state=State(**repo_info) if repo_info else None,
                            reasoning="Memory changes were too large to validate"
                        )
        except Exception as e:
            logging.error(f"Failed to get memory diff: {e}")
            memory_diff_content = f"Error getting memory diff: {str(e)}"
        
        # Prepare additional evidence if needed
        additional_evidence = []

        # If memory hashes are different but no diff content, provide additional context
        if (hasattr(goal, 'initial_state') and goal.initial_state and 
            hasattr(goal.initial_state, 'memory_hash') and goal.initial_state.memory_hash and
            hasattr(goal, 'current_state') and goal.current_state and
            hasattr(goal.current_state, 'memory_hash') and goal.current_state.memory_hash and
            goal.initial_state.memory_hash != goal.current_state.memory_hash and
            (memory_diff_content == "No memory changes detected" or memory_diff_content.startswith("Error getting memory diff"))):
            
            additional_evidence.append(f"Memory hash changed from {goal.initial_state.memory_hash} to {goal.current_state.memory_hash}")
            
            # Try to get changed files directly
            try:
                memory_repo_path = execution_result.repository_path
                if hasattr(goal.initial_state, 'memory_repository_path') and goal.initial_state.memory_repository_path:
                    memory_repo_path = goal.initial_state.memory_repository_path
                    
                cmd = ["git", "diff", "--name-only", goal.initial_state.memory_hash, goal.current_state.memory_hash]
                proc = subprocess.run(
                    cmd,
                    cwd=memory_repo_path,
                    capture_output=True,
                    text=True
                )
                
                if proc.returncode == 0 and proc.stdout.strip():
                    changed_files = proc.stdout.strip().split("\n")
                    additional_evidence.append(f"Changed files: {', '.join(changed_files)}")
            except Exception as e:
                logging.error(f"Failed to get direct changed files: {e}")

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

#### Repository Changes
```diff
{repo_diff}
```

#### Memory Repository Changes
```diff
{memory_diff_content}
```

#### Memory Changed Files
{json.dumps(memory_diff_files, indent=2) if memory_diff_files else "No files changed in memory"}

{("#### Additional Evidence\n" + "\n".join(additional_evidence)) if additional_evidence else ""}

Evaluate the evidence above to determine if the goal's validation criteria are met. 
Do not perform any new actions or tool calls to satisfy the criteria.
Only validate based on the evidence provided from previous execution.
"""
        },
    ]
        
        # Store the messages for later saving
        self.last_validation_messages = messages
        
        try:
            # Use the LLM to validate the goal
            validation_response = await self._validate_with_tools(goal, messages)
            
            # Log the raw response for debugging
            logging.info(f"Raw validation response (first 200 chars): {validation_response[:200] if validation_response else 'empty'}")
            
            # Extract criteria_results from validation_response and convert to ValidationResult
            criteria_data = self._extract_validation_json(validation_response)
            
            if not criteria_data or "criteria_results" not in criteria_data:
                logging.error("No validation criteria results found in response")
                # Return failed validation with error message
                failed_criteria = []
                for criterion in goal.validation_criteria:
                    failed_criteria.append(
                        CriterionResult(
                            criterion=criterion,
                            passed=False,
                            reasoning=f"Validation failed due to error: Could not extract valid JSON from LLM response",
                            evidence=[]
                        )
                    )
                return ValidationResult(
                    goal_id=goal.id,
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    criteria_results=failed_criteria,
                    score=0.0,
                    validated_by="LLM",
                    automated=True,
                    repository_state=State(**repo_info) if repo_info else None,
                )
                
            # Create criterion results from validation data
            criteria_results = []
            for criteria in criteria_data.get("criteria_results", []):
                criteria_results.append(
                    CriterionResult(
                        criterion=criteria.get("criterion", ""),
                        passed=criteria.get("passed", False),
                        reasoning=criteria.get("reasoning", ""),
                        evidence=criteria.get("evidence", [])
                    )
                )
                
            # Calculate score
            score = criteria_data.get("score", 0.0)
            if score > 1.0:  # Normalize score if it's given as percentage
                score = score / 100.0
                
            if score == 0.0 and criteria_results:
                # Calculate score based on passed criteria if not provided
                passed_count = sum(1 for cr in criteria_results if cr.passed)
                total_count = len(criteria_results)
                if total_count > 0:
                    score = passed_count / total_count
            
            # Return validation result
            return ValidationResult(
                goal_id=goal.id,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                criteria_results=criteria_results,
                score=score,
                validated_by="LLM",
                automated=True,
                repository_state=State(**repo_info) if repo_info else None,
            )
        except Exception as e:
            logging.error(f"Error during automated validation: {e}")
            # Create failed criteria results for each validation criterion
            failed_criteria = []
            for criterion in goal.validation_criteria:
                failed_criteria.append(
                    CriterionResult(
                        criterion=criterion,
                        passed=False,
                        reasoning=f"Validation failed due to error: {str(e)}",
                        evidence=[]
                    )
                )
            
            # Return failed validation
            return ValidationResult(
                goal_id=goal.id,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                criteria_results=failed_criteria,
                score=0.0,
                validated_by="LLM",
                automated=True,
                repository_state=State(**repo_info) if repo_info else None,
            )
    
    def _generate_criterion_reasoning(self, criterion: str, passed: bool, evidence: List[str]) -> str:
        """Generate detailed reasoning for a single criterion validation."""
        if passed:
            return f"Criterion satisfied: {criterion}\nEvidence: {'; '.join(evidence)}"
        else:
            return f"Criterion not satisfied: {criterion}\nEvidence: {'; '.join(evidence)}"
    
    def _generate_reasoning(self, criteria_results: List[Any], 
                          score: float, threshold: float) -> str:
        """Generate a human-readable reasoning for the validation result."""
        # Handle both CriterionResult objects and dictionaries
        passed_count = 0
        total_count = len(criteria_results)
        
        for result in criteria_results:
            if hasattr(result, 'passed'):
                # It's a CriterionResult object
                if result.passed:
                    passed_count += 1
            elif isinstance(result, dict) and 'passed' in result:
                # It's a dictionary
                if result['passed']:
                    passed_count += 1
        
        reasoning = []
        reasoning.append(f"Validation {'passed' if score >= threshold else 'failed'} with score {score:.2f}/{threshold:.2f}")
        reasoning.append(f"Satisfied {passed_count}/{total_count} criteria")
        
        # Add details for failed criteria
        if passed_count < total_count:
            reasoning.append("\nFailed criteria:")
            for result in criteria_results:
                if hasattr(result, 'passed') and not result.passed:
                    # It's a CriterionResult object
                    reasoning.append(f"- {result.criterion}")
                    reasoning.append(f"  Reason: {result.reasoning}")
                elif isinstance(result, dict) and not result.get('passed', True):
                    # It's a dictionary
                    reasoning.append(f"- {result.get('criterion', 'Unknown criterion')}")
                    reasoning.append(f"  Reason: {result.get('reasoning', 'No reason provided')}")
        
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

    def _extract_validation_json(self, content: str) -> dict:
        """
        Extract a JSON dictionary containing validation criteria results from the LLM response.
        Returns an empty dictionary if no valid response is processed.
        
        Args:
            content: The content to extract JSON from, typically the LLM response.
            
        Returns:
            A dictionary containing the validation results with criteria_results, score, and all_passed fields.
            Returns an empty dictionary if no valid response could be processed.
        """
        if not content:
            logging.error("Cannot extract validation JSON from empty content")
            return {"criteria_results": []}

        logging.debug(f"Extracting validation JSON from content of length {len(content)}")
        
        # Check if content has debug prefix and strip it
        debug_prefix_match = re.search(r'^.*?DEBUG:\s+Raw\s+LLM\s+response\s+content:\s+', content)
        if debug_prefix_match:
            logging.debug(f"Detected debug prefix, stripping it out")
            content = content[debug_prefix_match.end():]
        
        # Check for other common prefixes like terminal output
        term_prefix_match = re.search(r'^[.\s\w]+%\s+', content)
        if term_prefix_match:
            logging.debug(f"Detected terminal output prefix, stripping: {term_prefix_match.group(0)}")
            content = content[term_prefix_match.end():]
        
        # First try to parse the content as JSON directly
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "criteria_results" in parsed:
                logging.debug("Successfully parsed content as direct JSON")
                return parsed
        except json.JSONDecodeError:
            logging.debug("Failed to parse content as direct JSON")
        
        # Look for JSON in markdown code blocks
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content)
        logging.debug(f"Found {len(code_blocks)} code blocks")
        
        for block in code_blocks:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "criteria_results" in parsed:
                    logging.debug("Successfully parsed JSON from code block")
                    return parsed
            except json.JSONDecodeError as e:
                logging.debug(f"Failed to parse code block as JSON ({str(e)})")
                # Try to fix truncated code blocks
                if '{' in block and '}' not in block.split('{', 1)[1]:
                    logging.debug("Detected potentially truncated JSON in code block, attempting to fix")
                    try:
                        # Find the last complete nested object
                        parts = block.split('{')
                        reconstructed = parts[0]
                        depth = 0
                        for i, part in enumerate(parts[1:], 1):
                            reconstructed += '{'
                            for j, char in enumerate(part):
                                if char == '{':
                                    depth += 1
                                    reconstructed += char
                                elif char == '}':
                                    depth -= 1
                                    reconstructed += char
                                else:
                                    reconstructed += char
                            
                            # If we've closed all braces, we've found a complete object
                            if depth == 0:
                                break
                        
                        # Add missing closing braces if needed
                        if depth > 0:
                            reconstructed += '}' * depth
                        
                        # Try to parse the reconstructed JSON
                        fixed_json = json.loads(reconstructed)
                        if isinstance(fixed_json, dict) and "criteria_results" in fixed_json:
                            logging.debug("Successfully parsed fixed JSON from truncated code block")
                            return fixed_json
                    except Exception as fix_error:
                        logging.debug(f"Failed to fix truncated JSON: {str(fix_error)}")
        
        # Try to extract just the criteria_results array using regex
        criteria_results_match = re.search(r'"criteria_results"\s*:\s*(\[\s*\{.*?\}\s*\])', content, re.DOTALL)
        if criteria_results_match:
            criteria_json = criteria_results_match.group(1)
            logging.debug(f"Found criteria_results JSON array")
            
            try:
                criteria_results = json.loads(criteria_json)
                result = {"criteria_results": criteria_results}
                
                # Calculate score
                passed_count = sum(1 for cr in criteria_results if isinstance(cr, dict) and cr.get("passed", False))
                total_count = len(criteria_results)
                result["score"] = (passed_count / total_count) if total_count > 0 else 0
                result["all_passed"] = passed_count == total_count
                
                logging.debug(f"Successfully parsed criteria_results array with {passed_count}/{total_count} passed")
                return result
            except json.JSONDecodeError as e:
                logging.debug(f"Failed to parse criteria_results array: {str(e)}")
                
                # If parsing failed, try to fix truncated JSON
                if '{' in criteria_json and '}' in criteria_json:
                    # Count braces to check if JSON is truncated
                    open_braces = criteria_json.count('{')
                    close_braces = criteria_json.count('}')
                    
                    if open_braces > close_braces:
                        logging.debug("Attempting to fix truncated JSON")
                        # Find the last complete object in the array
                        last_complete_index = -1
                        depth = 0
                        in_object = False
                        for i, char in enumerate(criteria_json):
                            if char == '{':
                                depth += 1
                                in_object = True
                            elif char == '}':
                                depth -= 1
                                if depth == 0 and in_object:
                                    last_complete_index = i
                                    in_object = False
                        
                        if last_complete_index > 0:
                            fixed_json = criteria_json[:last_complete_index + 1] + ']'
                            try:
                                criteria_results = json.loads(fixed_json)
                                result = {"criteria_results": criteria_results}
                                
                                # Calculate score
                                passed_count = sum(1 for cr in criteria_results if isinstance(cr, dict) and cr.get("passed", False))
                                total_count = len(criteria_results)
                                result["score"] = (passed_count / total_count) if total_count > 0 else 0
                                result["all_passed"] = passed_count == total_count
                                
                                logging.debug(f"Successfully parsed fixed truncated JSON with {passed_count}/{total_count} passed")
                                return result
                            except json.JSONDecodeError:
                                logging.debug("Failed to parse fixed truncated JSON")
        
        # As a last resort, try to extract individual criterion patterns
        criterion_pattern = r'"criterion"\s*:\s*"([^"]+)".*?"passed"\s*:\s*(true|false)'
        criterion_matches = re.findall(criterion_pattern, content, re.DOTALL)
        
        if criterion_matches:
            logging.debug(f"Found {len(criterion_matches)} criteria matches using regex pattern")
            
            criteria_results = []
            for criterion, passed in criterion_matches:
                criteria_entry = {
                    "criterion": criterion,
                    "passed": passed.lower() == "true",
                    "reasoning": "Extracted from truncated JSON response",
                    "evidence": ["Evidence extracted from truncated response"]
                }
                
                # Try to find specific reasoning for this criterion
                reasoning_pattern = rf'"criterion"\s*:\s*"{re.escape(criterion)}".*?"reasoning"\s*:\s*"([^"]+?)"'
                reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
                if reasoning_match:
                    criteria_entry["reasoning"] = reasoning_match.group(1)
                
                # Try to find specific evidence for this criterion
                evidence_pattern = rf'"criterion"\s*:\s*"{re.escape(criterion)}".*?"evidence"\s*:\s*\[(.*?)\]'
                evidence_match = re.search(evidence_pattern, content, re.DOTALL)
                if evidence_match:
                    # Extract individual evidence items
                    evidence_items = re.findall(r'"([^"]+)"', evidence_match.group(1))
                    if evidence_items:
                        criteria_entry["evidence"] = evidence_items
                
                criteria_results.append(criteria_entry)
            
            result = {"criteria_results": criteria_results}
            
            # Calculate score
            passed_count = sum(1 for cr in criteria_results if isinstance(cr, dict) and cr.get("passed", False))
            total_count = len(criteria_results)
            result["score"] = (passed_count / total_count) if total_count > 0 else 0
            result["all_passed"] = passed_count == total_count
            
            logging.debug(f"Successfully extracted {len(criteria_results)} criteria with {passed_count}/{total_count} passed")
            return result
        
        # Special handling for truncated JSON where we can't extract any criteria
        if '{"criteria_results":' in content and not '"criterion"' in content:
            logging.warning("Found criteria_results JSON object but could not extract criteria")
            # Extract any criterion from validation_criteria and mark as failed due to truncation
            return {
                "criteria_results": [
                    {
                        "criterion": "Extracted from partial JSON response",
                        "passed": False,
                        "reasoning": "Extracted from partial JSON response",
                        "evidence": ["Partial JSON extraction"]
                    }
                ],
                "overall_score": 0,
                "overall_reasoning": "Extracted from partial validation response"
            }
            
        # If we got here, we couldn't extract the validation data
        logging.warning("No valid JSON validation response could be processed")
        return {"criteria_results": []} 

    async def _validate_with_tools(self, goal: Goal, messages: List[Dict[str, str]]) -> str:
        """
        Use LLM with tools to validate a goal.
        
        Args:
            goal: The goal to validate
            messages: The messages to send to the LLM, including system prompt and user query
            
        Returns:
            The response content from the LLM
        """
        logging.info(f"Validating goal {goal.id} with LLM")
        
        # Run LLM with tools
        response, tool_calls = await self.tool_processor.run_llm_with_tools(
            messages=messages,
            model=self.model
        )
        
        # Store the complete conversation including tools for context saving
        self.last_validation_messages = response
        
        # Extract response from LLM
        if response and len(response) > 0:
            assistant_message = next((msg for msg in response if msg["role"] == "assistant"), None)
            if assistant_message and assistant_message.get("content"):
                return assistant_message.get("content", "")
        
        # If we didn't get a content response but got tool calls instead, report an error
        if tool_calls:
            logging.warning("Validator used tools instead of validating based on provided evidence")
            
            # Create a description of tools used, for debugging purposes
            tool_descriptions = []
            for i, tool_call in enumerate(tool_calls):
                if tool_call.get("tool") and tool_call.get("args"):
                    tool_name = tool_call.get("tool")
                    args = tool_call.get("args", {})
                    tool_descriptions.append(f"Used tool: {tool_name} with arguments: {json.dumps(args)}")
            
            # Return a validation response indicating tool usage isn't valid for validation
            return json.dumps({
                "criteria_results": [
                    {
                        "criterion": "Validation based on evidence",
                        "passed": False,
                        "reasoning": "The validator attempted to use tools instead of validating based on the provided evidence",
                        "evidence": tool_descriptions
                    }
                ],
                "score": 0.0,
                "reasoning": "Validation should examine the provided evidence, not execute tools"
            })
        
        # If we get here, we have neither content nor tool calls to work with
        logging.warning("No content or tool calls in LLM response")
        return json.dumps({
            "criteria_results": [
                {
                    "criterion": "Valid response required",
                    "passed": False,
                    "reasoning": "No content or tool calls in LLM response",
                    "evidence": ["LLM did not provide a meaningful response"]
                }
            ],
            "score": 0.0,
            "reasoning": "Validation failed due to empty LLM response"
        }) 