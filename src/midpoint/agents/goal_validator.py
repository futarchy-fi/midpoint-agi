"""
Goal Validation agent implementation.

IMPORTANT: This module implements a generic goal validation system that uses LLM to validate
task execution results. It MUST NOT contain any task-specific logic or hardcoded validation rules.
All validation decisions should be made by the LLM at runtime.
"""

import asyncio
from typing import List, Dict, Any
import re
import os
import random
import json

from .models import Goal, ExecutionResult, ValidationResult
from .tools import (
    get_current_hash,
    validate_repository_state,
    list_directory,
    read_file,
    search_code,
    run_terminal_cmd,
    get_current_branch,
    web_search,
    web_scrape
)
from .config import get_openai_api_key
from openai import AsyncOpenAI

class GoalValidator:
    """
    Generic goal validation agent that uses LLM to validate execution results.
    
    This class MUST:
    - Remain completely task-agnostic
    - Not contain any hardcoded validation rules
    - Delegate all validation decisions to the LLM
    - Use the provided tools to gather evidence for validation
    """
    
    def __init__(self):
        """Initialize the GoalValidator agent."""
        # Initialize OpenAI client with API key from config
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        if not api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
            
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # System prompt for the LLM that will make all validation decisions
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
- web_search: Search the web using DuckDuckGo's API
- web_scrape: Scrape content from a webpage

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

        # Define tool schema for the OpenAI API
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List the contents of a directory in the repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to the git repository"
                            },
                            "directory": {
                                "type": "string",
                                "description": "Directory to list within the repository",
                                "default": "."
                            }
                        },
                        "required": ["repo_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file in the repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to the git repository"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file within the repository"
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "First line to read (0-indexed)",
                                "default": 0
                            },
                            "max_lines": {
                                "type": "integer",
                                "description": "Maximum number of lines to read",
                                "default": 100
                            }
                        },
                        "required": ["repo_path", "file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": "Search the codebase for patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to the git repository"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Regular expression pattern to search for"
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": "Pattern for files to include (e.g., '*.py')",
                                "default": "*"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 20
                            }
                        },
                        "required": ["repo_path", "pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_terminal_cmd",
                    "description": "Run a terminal command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to run"
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Working directory for the command"
                            }
                        },
                        "required": ["command", "cwd"]
                    }
                }
            }
        ]

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
            # Create the user prompt for the LLM
            user_prompt = f"""Goal: {goal.description}

Validation Criteria:
{chr(10).join(f"- {criterion}" for criterion in goal.validation_criteria)}

Repository Path: {execution_result.repository_path}
Branch: {execution_result.branch_name}
Git Hash: {execution_result.git_hash}

Please validate the execution result against each criterion.
Use the available tools to gather evidence and make validation decisions."""

            # Initialize messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Chat completion with tool use
            try:
                final_output = None
                
                # Loop until we get a final output
                while final_output is None:
                    # Call OpenAI API
                    response = await self.client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        temperature=0.1,
                        max_tokens=4000
                    )
                    
                    # Get the model's message
                    message = response.choices[0].message
                    
                    # Add the message to our conversation
                    messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})
                    
                    # If the model wants to use tools
                    if message.tool_calls:
                        # Handle each tool call
                        for tool_call in message.tool_calls:
                            # Get the function call details
                            func_name = tool_call.function.name
                            func_args = json.loads(tool_call.function.arguments)
                            
                            # Execute the appropriate tool
                            if func_name == "list_directory":
                                result = await list_directory(func_args["repo_path"], func_args.get("directory", "."))
                            elif func_name == "read_file":
                                result = await read_file(
                                    func_args["repo_path"],
                                    func_args["file_path"],
                                    func_args.get("start_line", 0),
                                    func_args.get("max_lines", 100)
                                )
                            elif func_name == "search_code":
                                result = await search_code(
                                    func_args["repo_path"],
                                    func_args["pattern"],
                                    func_args.get("file_pattern", "*"),
                                    func_args.get("max_results", 20)
                                )
                            elif func_name == "run_terminal_cmd":
                                result = await run_terminal_cmd(
                                    command=func_args["command"],
                                    cwd=func_args["cwd"]
                                )
                            
                            # Add the result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": func_name,
                                "content": str(result)
                            })
                    else:
                        # Try to parse the final output
                        try:
                            final_output = json.loads(message.content)
                            
                            # Validate the output format
                            if not all(key in final_output for key in ["criteria_results", "overall_score", "overall_reasoning"]):
                                final_output = None
                                messages.append({
                                    "role": "user",
                                    "content": "Please provide your response in the correct JSON format with all required fields."
                                })
                        except json.JSONDecodeError:
                            messages.append({
                                "role": "user",
                                "content": "Please provide your response in valid JSON format."
                            })
                
                # Create the validation result
                return ValidationResult(
                    success=final_output["overall_score"] >= goal.success_threshold,
                    score=final_output["overall_score"],
                    reasoning=final_output["overall_reasoning"],
                    criteria_results=final_output["criteria_results"],
                    git_hash=execution_result.git_hash,
                    branch_name=execution_result.branch_name
                )
            except Exception as e:
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