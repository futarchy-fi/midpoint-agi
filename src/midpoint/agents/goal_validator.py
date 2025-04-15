"""
Goal Validation agent implementation.

IMPORTANT: This module implements a generic goal validation system that uses LLM to validate
task execution results. It MUST NOT contain any task-specific logic or hardcoded validation rules.
All validation decisions should be made by the LLM at runtime.
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import os
import random
from pathlib import Path

from openai import AsyncOpenAI

from midpoint.agents.models import Goal, ExecutionResult, CriterionResult, ValidationResult, State
from midpoint.agents.tools.git_tools import get_current_hash, get_current_branch
from .tools import (
    list_directory,
    read_file,
    search_code,
    run_terminal_cmd,
    web_search,
    web_scrape
)
from .tools.processor import ToolProcessor
from .tools.registry import ToolRegistry
from .config import get_openai_api_key

# Set up logging
logger = logging.getLogger(__name__)

# System prompt for the validator - MODIFIED
VALIDATION_SYSTEM_PROMPT = """
You are the Goal Validator, an expert at verifying whether a goal's validation criteria have been met.

Your task is to determine if the work done for a goal meets its validation criteria.
Follow an OODA loop:

1.  **OBSERVE:** You will be given the initial and potentially final state hashes (git and memory). Use available tools (`run_terminal_cmd`, `read_file`, potentially others) to:
    *   Verify you are on the correct git branch specified in the execution result.
    *   Get the repository diff (`git diff <initial_hash> <final_hash> | cat`).
    *   Get the memory diff (e.g., using `git diff` in the memory repo if applicable, or by reading relevant files if needed).
    *   Read specific files if needed to understand the changes related to the criteria.

2.  **ORIENT:** Analyze the evidence gathered (diffs, file contents) in relation to the goal's validation criteria.

3.  **DECIDE:** For each validation criterion, determine if the evidence shows it has been met. Be precise and objective.

4.  **OUTPUT:** Provide ONLY a valid JSON object with this exact structure:
    {
        "criteria_results": [
            {
                "criterion": "string",
                "passed": boolean,
                "reasoning": "string",
                "evidence": ["string"] // Specific references to diffs or file contents
            }
        ],
        "score": float, // Overall score 0.0-1.0 based on passed criteria
        "reasoning": "string" // Overall reasoning for the score
    }

IMPORTANT:
- Base your validation ONLY on the changes between the initial and final states provided.
- Cite specific evidence from the diffs or files you examined.
- Do not include any explanatory text before or after the JSON object.
- If you cannot gather necessary evidence with tools (e.g., tool error, insufficient info), reflect this failure in the reasoning and mark relevant criteria as failed.
"""

class GoalValidator:
    """
    Generic goal validation agent that uses LLM to validate execution results.
    
    This class MUST:
    - Remain completely task-agnostic
    - Not contain any hardcoded validation rules
    - Delegate all validation decisions to the LLM
    - Let the LLM use tools to gather evidence for validation

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
        self.system_prompt = VALIDATION_SYSTEM_PROMPT # Use updated prompt

        # Get tools from registry
        self.tools = ToolRegistry.get_tool_schemas()

    def validate_execution(self, goal: Goal, execution_result: ExecutionResult) -> ValidationResult:
        """
        Validate execution results against a goal using LLM.
        
        NOTE: This synchronous method relies on its caller providing an asyncio event loop
        because it internally calls async helper functions/tools.
        
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
                reasoning="Execution reported as failed."
            )
        
        # Get current state information to provide context to the LLM
        current_repo_path = execution_result.repository_path
        current_branch = "unknown"
        current_hash = "unknown"
        try:
            # Call async helper directly, relies on caller's loop
            current_branch = get_current_branch(current_repo_path)
            # Call async helper directly, relies on caller's loop
            current_hash = get_current_hash(current_repo_path)
            logging.info(f"Current state for validation: Branch='{current_branch}', Hash='{current_hash[:8]}'")
        except Exception as e:
            logging.error(f"Failed to get initial repository state: {e}")
            # Proceed, but LLM will need to verify/handle this

        # Extract initial state info from the goal object
        initial_git_hash = None
        initial_memory_hash = None
        if hasattr(goal, 'initial_state') and goal.initial_state:
            initial_git_hash = getattr(goal.initial_state, 'git_hash', None)
            initial_memory_hash = getattr(goal.initial_state, 'memory_hash', None)

        # Extract current memory hash from the goal object (if available)
        current_memory_hash = None
        if hasattr(goal, 'current_state') and goal.current_state:
             current_memory_hash = getattr(goal.current_state, 'memory_hash', None)
        # Fallback to execution result if not on goal
        if not current_memory_hash and execution_result.final_state:
             current_memory_hash = getattr(execution_result.final_state, 'memory_hash', None)

        # Prepare the user prompt for the LLM
        prompt_lines = [
            f"Please validate the goal: '{goal.description}'",
            "Based on the changes between the initial and current states.",
            "\n**Goal Details:**",
            f"- Description: {goal.description}",
            f"- Validation Criteria:",
        ]
        prompt_lines.extend([f"  - {c}" for c in goal.validation_criteria])

        prompt_lines.append("\n**State Information:**")
        prompt_lines.append(f"- Target Branch: {execution_result.branch_name}") # Branch LLM should be on
        prompt_lines.append(f"- Initial Git Hash: {initial_git_hash or 'Not specified'}")
        prompt_lines.append(f"- Final Git Hash: {current_hash}") # The hash LLM should currently see
        prompt_lines.append(f"- Initial Memory Hash: {initial_memory_hash or 'Not specified'}")
        prompt_lines.append(f"- Final Memory Hash: {current_memory_hash or 'Not specified'}")
        prompt_lines.append(f"- Repository Path: {current_repo_path}")
        # We might need to provide memory repo path if it's separate and needed for tools
        # memory_repo_path = ... # How to get this reliably? Assume tools can infer for now or add if needed.

        prompt_lines.append("\n**Instructions:**")
        prompt_lines.append("1. Verify you are on the target git branch using tools.")
        prompt_lines.append("2. Use tools to get the repository diff between the initial and final git hashes.")
        prompt_lines.append("3. Use tools to investigate memory changes between the initial and final memory hashes (e.g., diff or reading files)." if initial_memory_hash and current_memory_hash else "3. No memory comparison needed or possible.")
        prompt_lines.append("4. Analyze the gathered evidence against the validation criteria.")
        prompt_lines.append("5. Provide your assessment in the required JSON format.")

        user_prompt = "\n".join(prompt_lines)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Store the messages for later saving (consider if this is still needed/useful)
        self.last_validation_messages = messages

        try:
            # Use the LLM to perform validation, allowing tool use
            validation_response_content = self._validate_with_tools(goal, messages)

            # Log the raw response for debugging
            logging.info(f"Raw validation response (first 200 chars): {validation_response_content[:200] if validation_response_content else 'empty'}")

            # Extract criteria_results from validation_response
            criteria_data = self._extract_validation_json(validation_response_content)

            if not criteria_data or "criteria_results" not in criteria_data:
                logging.error("No valid validation criteria results found in LLM response")
                # Create a generic failure if JSON is invalid
                failed_criteria = [
                    CriterionResult(
                        criterion=c,
                        passed=False,
                        reasoning="Validation failed: Could not parse valid JSON from LLM response.",
                        evidence=[f"Raw response: {validation_response_content[:200]}..."]
                    ) for c in goal.validation_criteria
                ]
                return ValidationResult(
                    goal_id=goal.id,
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    criteria_results=failed_criteria,
                    score=0.0,
                    validated_by="LLM",
                    automated=True,
                    repository_state=State(git_hash=current_hash, branch_name=current_branch, repository_path=current_repo_path) if current_hash != "unknown" else None,
                    reasoning="Could not extract valid JSON from LLM response"
                )

            # Create criterion results from validation data
            criteria_results = []
            all_criteria_found = True
            goal_criteria_set = set(goal.validation_criteria)
            response_criteria_set = set()
            for criteria in criteria_data.get("criteria_results", []):
                criterion_text = criteria.get("criterion", "")
                response_criteria_set.add(criterion_text)
                criteria_results.append(
                    CriterionResult(
                        criterion=criterion_text,
                        passed=criteria.get("passed", False),
                        reasoning=criteria.get("reasoning", ""),
                        evidence=criteria.get("evidence", [])
                    )
                )
            # Check if LLM missed any criteria
            missing_criteria = goal_criteria_set - response_criteria_set
            if missing_criteria:
                all_criteria_found = False
                for criterion in missing_criteria:
                    criteria_results.append(
                        CriterionResult(
                            criterion=criterion,
                            passed=False,
                            reasoning="Validation failed: LLM did not provide an assessment for this criterion.",
                            evidence=[]
                        )
                    )

            # Calculate score
            score = criteria_data.get("score", criteria_data.get("overall_score")) # Try both keys
            reasoning = criteria_data.get("reasoning", criteria_data.get("overall_reasoning", ""))

            # If score is missing or invalid, recalculate based on results
            if score is None or not isinstance(score, (float, int)):
                logging.warning("Score missing or invalid in LLM response, recalculating.")
                passed_count = sum(1 for cr in criteria_results if cr.passed)
                total_count = len(criteria_results)
                score = (passed_count / total_count) if total_count > 0 else 0.0
            elif score > 1.0: # Normalize score if it's given as percentage
                score = score / 100.0

            # Add note about missing criteria to reasoning if needed
            if not all_criteria_found:
                reasoning += f"\nNote: LLM did not evaluate all criteria ({len(missing_criteria)} missing)."

            # Return validation result
            return ValidationResult(
                goal_id=goal.id,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                criteria_results=criteria_results,
                score=score,
                validated_by="LLM",
                automated=True,
                # Provide the state as observed *before* LLM interaction
                repository_state=State(git_hash=current_hash, branch_name=current_branch, repository_path=current_repo_path) if current_hash != "unknown" else None,
                reasoning=reasoning
            )

        except Exception as e:
            logging.error(f"Error during automated validation: {e}", exc_info=True) # Add traceback
            # Create failed criteria results for each validation criterion
            failed_criteria = []
            for criterion in goal.validation_criteria:
                failed_criteria.append(
                    CriterionResult(
                        criterion=criterion,
                        passed=False,
                        reasoning=f"Validation failed due to system error: {str(e)}",
                        evidence=[]
                    )
                )
            
            # Return failed validation
            return ValidationResult(
                goal_id=goal.id,
                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                criteria_results=failed_criteria,
                score=0.0,
                validated_by="System", # System error, not LLM error
                automated=True,
                repository_state=State(git_hash=current_hash, branch_name=current_branch, repository_path=current_repo_path) if current_hash != "unknown" else None,
                reasoning=f"Automated validation failed with exception: {str(e)}"
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

    def validate_goal(self, goal_path: str, repository_path: str = ".") -> ValidationResult:
        """
        Validate a goal using a dummy execution result.
        
        NOTE: This synchronous method relies on its caller providing an asyncio event loop
        for the internal calls to async helpers and self.validate_execution.
        
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
            
            # Use current_state hash from goal file as the definitive 'final' hash
            current_git_hash = current_state.get('git_hash')
            current_memory_hash = current_state.get('memory_hash')
            current_timestamp = current_state.get('timestamp')

            if not current_git_hash:
                logging.warning(f"Current git hash not found in goal file {goal_path}, attempting to get from repo.")
                current_git_hash = get_current_hash(repository_path)
            if not current_git_hash:
                 raise ValueError(f"Could not determine current git hash for validation from {goal_path} or repository.")

            logging.debug(f"Using Initial Git Hash: {initial_git_hash}")
            logging.debug(f"Using Final Git Hash:   {current_git_hash}")
            logging.debug(f"Using Initial Mem Hash: {initial_memory_hash}")
            logging.debug(f"Using Final Mem Hash:   {current_memory_hash}")

            # Get the branch name associated with the current state hash
            # This might require a separate lookup or be stored in the goal file?
            # For now, fetch from the repo, assuming we're on the right branch.
            current_branch_name = get_current_branch(repository_path)
            logging.debug(f"Assuming current branch is: {current_branch_name}")

            # Create Goal object
            goal = Goal(
                id=goal_data.get('goal_id', ''),
                description=goal_data.get('description', ''),
                validation_criteria=goal_data.get('validation_criteria', []),
                success_threshold=goal_data.get("success_threshold", 80.0),
                initial_state=State(
                    git_hash=initial_git_hash,
                    memory_hash=initial_memory_hash,
                    timestamp=initial_timestamp
                ),
                current_state=State( # Pass current state from goal file
                    git_hash=current_git_hash,
                    memory_hash=current_memory_hash,
                    timestamp=current_timestamp
                )
            )
            
            # Create ExecutionResult reflecting the state defined in the goal file
            execution_result = ExecutionResult(
                success=True, # Assume success to trigger validation
                repository_path=repository_path,
                branch_name=current_branch_name, # Branch associated with final state
                git_hash=current_git_hash,    # Final hash
                task_id='',
                goal_id=goal_data.get('goal_id', ''),
                error_message=None,
                # Include final state based on goal file's current_state
                final_state=State(
                    git_hash=current_git_hash,
                    memory_hash=current_memory_hash,
                    timestamp=current_timestamp,
                    repository_path=repository_path,
                    branch_name=current_branch_name
                )
            )
            
            # Validate execution (sync call)
            return self.validate_execution(goal, execution_result)
        except Exception as e:
            logging.error(f"Error validating goal: {e}", exc_info=True)
            # Return a structured error
            return ValidationResult(
                 goal_id=goal_path, # Use path as fallback ID
                 timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                 criteria_results=[],
                 score=0.0,
                 validated_by="System",
                 automated=True,
                 repository_state=None,
                 reasoning=f"Failed to set up validation for {goal_path}: {str(e)}"
             )

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

    # Keep this method synchronous
    def _validate_with_tools(self, goal: Goal, messages: List[Dict[str, str]]) -> str:
        """
        Use LLM with tools to validate a goal.
        
        NOTE: This synchronous method relies on its caller providing an asyncio event loop
        for the internal call to self.tool_processor.run_llm_with_tools.
        
        Args:
            goal: The goal to validate
            messages: The messages to send to the LLM, including system prompt and user query
            
        Returns:
            The response content from the LLM
        """
        logging.info(f"Validating goal {goal.id} with LLM using tools")
        
        # Run LLM with tools
        # Call async tool processor directly, relies on caller's loop
        response, tool_calls = self.tool_processor.run_llm_with_tools(
            messages=messages,
            model=self.model
        )
        
        # Store the complete conversation including tools for context saving
        self.last_validation_messages = response
        
        # Extract response from LLM
        if response and len(response) > 0:
            assistant_message = next((msg for msg in response if msg["role"] == "assistant"), None)
            # The final message might just be the JSON, or could be preceded by tool calls
            # We return the content of the *last* assistant message assuming it holds the final JSON
            if assistant_message and assistant_message.get("content"):
                return assistant_message.get("content", "")
            else:
                # If the last message has no content (e.g., only tool calls), return empty
                logging.warning("Final assistant message had no content.")
                return ""
        
        # If we get here, something went wrong, no response list or empty
        logging.warning("No response messages received from LLM interaction.")
        return json.dumps({
            "criteria_results": [],
            "score": 0.0,
            "reasoning": "Validation failed: No response received from LLM."
        }) 