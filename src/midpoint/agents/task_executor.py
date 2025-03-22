"""
Task Executor - Generic task execution agent for the Midpoint system.

IMPORTANT: This module implements a generic task execution system that uses LLM to interpret
and execute tasks. It MUST NOT contain any task-specific logic or hardcoded implementations.
All task-specific decisions and implementations should be handled by the LLM at runtime.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import os
import logging

from .models import TaskContext, ExecutionTrace, State, Goal, ExecutionResult
from .goal_decomposer import validate_repository_state
from .tools import (
    check_repo_state,
    create_branch,
    create_commit,
    get_current_hash,
    get_current_branch,
    checkout_branch,
    list_directory,
    read_file,
    search_code,
    edit_file,
    run_terminal_cmd,
    validate_repository_state,
    web_search,
    web_scrape
)
from .config import get_openai_api_key
from openai import AsyncOpenAI

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TaskExecutor')

class TaskExecutor:
    """
    Generic task execution agent that uses LLM to implement changes.
    
    This class MUST:
    - Remain completely task-agnostic
    - Not contain any hardcoded task implementations
    - Delegate all task-specific decisions to the LLM
    - Use the provided tools to implement changes based on LLM decisions
    
    The workflow is:
    1. LLM analyzes the task and current repository state
    2. LLM decides what changes are needed
    3. LLM uses the available tools to implement those changes
    4. LLM validates the changes
    5. Changes are committed if valid
    """
    
    def __init__(self):
        """Initialize the TaskExecutor agent."""
        logger.info("Initializing TaskExecutor")
        # Initialize OpenAI client with API key from config
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        if not api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
            
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        logger.info("TaskExecutor initialized successfully")
        
        # System prompt for the LLM that will make all task execution decisions
        self.system_prompt = """You are a task execution agent responsible for implementing code changes.
Your role is to:
1. Execute tasks using available tools
2. Track progress and report status
3. Handle errors gracefully
4. Maintain clean git state

IMPORTANT: A new branch has been created for you to work on. You are responsible for:
1. Making all necessary changes to implement the task
2. Creating commits with meaningful messages
3. Ensuring all changes are committed before completing
4. Returning the final commit hash in your output

Available tools:
- list_directory: List contents of a directory
- read_file: Read contents of a file
- search_code: Search for code patterns
- create_commit: Create a git commit
- run_terminal_cmd: Run a terminal command
- edit_file: Edit the contents of a file
- web_search: Search the web using DuckDuckGo's API
- web_scrape: Scrape content from a webpage

IMPORTANT TOOL USAGE NOTES:
1. When using tools, provide only the name without any prefixes (e.g., use "list_directory" not "functions.list_directory")
2. The create_commit tool will return the new commit hash - you must save this hash and use it as the final_commit_hash in your response
3. Always make your changes and verify them before creating a commit
4. After creating a commit, store the returned hash and use it in your final output

For each task:
1. Analyze the current repository state
2. Determine what changes are needed
3. Use the available tools to implement changes
4. Validate the changes
5. Create appropriate commits
6. Return the final commit hash from the create_commit call

Your response must be in JSON format with these fields:
{
    "actions": [
        {
            "tool": "string",  # Name of the tool to use (without any prefixes)
            "args": {},  # Arguments for the tool
            "purpose": "string"  # Why this action is needed
        }
    ],
    "final_commit_hash": "string",  # Hash returned from the create_commit call
    "validation_steps": [  # Steps to validate the changes
        "string"
    ],
    "task_completed": boolean,  # Whether the task was successfully completed
    "completion_reason": "string"  # Explanation if task was not completed
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
                    "name": "edit_file",
                    "description": "Edit the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to the git repository"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "content": {
                                "type": "string",
                                "description": "New content for the file"
                            }
                        },
                        "required": ["repo_path", "file_path", "content"]
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
            },
            {
                "type": "function",
                "function": {
                    "name": "create_commit",
                    "description": "Create a git commit with the given message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to the git repository"
                            },
                            "message": {
                                "type": "string",
                                "description": "Commit message"
                            }
                        },
                        "required": ["repo_path", "message"]
                    }
                }
            }
        ]

    async def execute_task(self, context: TaskContext, task: str) -> ExecutionResult:
        """
        Execute a task using LLM-driven decision making.
        
        This method MUST NOT contain any task-specific logic.
        All task-specific decisions and implementations should be made by the LLM.
        
        The LLM will:
        1. Analyze the task and repository state
        2. Decide what changes are needed
        3. Use the available tools to implement changes
        4. Validate the changes
        5. Create appropriate commits
        
        Args:
            context: The current task context
            task: The task description to execute
            
        Returns:
            ExecutionResult containing the execution outcome
        """
        logger.info(f"Starting task execution: {task}")
        logger.info(f"Repository: {context.state.repository_path}")
        logger.info(f"Current Git Hash: {context.state.git_hash}")
        
        # Initialize execution
        start_time = time.time()
        base_name = f"task-{context.iteration}"
        
        try:
            # Create a new branch for this execution
            logger.info(f"Creating new branch: {base_name}")
            branch_name = await create_branch(context.state.repository_path, base_name)
            logger.info(f"Created branch: {branch_name}")
            
            # Update state with branch name
            context.state.branch_name = branch_name
            
            # Create the user prompt for the LLM
            user_prompt = f"""Task: {task}

Repository Path: {context.state.repository_path}
Branch: {branch_name} (newly created for this task)
Git Hash: {context.state.git_hash}

IMPORTANT: Commits are not allowed in this system. All changes must be reviewed and committed manually.
Please analyze the task and determine what changes are needed.
Use the available tools to implement the changes and validate them.
Do not attempt to create commits - just make the necessary changes to the files."""

            logger.debug(f"User prompt: {user_prompt}")

            # Initialize messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Add a timeout for LLM interactions
            async with asyncio.timeout(30):  # 30 second timeout for LLM interactions
                while True:
                    try:
                        # Get LLM response
                        response = await self.client.chat.completions.create(
                            model="gpt-4-turbo-preview",
                            messages=messages,
                            temperature=0.7,
                            max_tokens=2000
                        )
                        
                        # Parse the response
                        content = response.choices[0].message.content
                        logger.debug(f"LLM response: {content}")
                        
                        try:
                            # Try to parse as JSON
                            final_output = json.loads(content)
                        except json.JSONDecodeError:
                            # If not JSON, try to extract JSON from the content
                            json_match = re.search(r'\{.*\}', content, re.DOTALL)
                            if json_match:
                                final_output = json.loads(json_match.group())
                            else:
                                raise ValueError("No valid JSON found in response")
                        
                        # Validate required fields
                        required_fields = ["task_completed", "final_commit_hash"]
                        if not all(field in final_output for field in required_fields):
                            logger.warning("Final output missing required fields")
                            logger.debug(f"Missing fields: {[field for field in required_fields if field not in final_output]}")
                            continue
                            
                        # Return result based on task completion
                        return ExecutionResult(
                            success=final_output["task_completed"],
                            branch_name=branch_name,
                            git_hash=context.state.git_hash,  # Use current hash since commits are disabled
                            error_message=None if final_output["task_completed"] else final_output.get("completion_reason"),
                            execution_time=time.time() - start_time,
                            repository_path=context.state.repository_path
                        )
                        
                    except asyncio.TimeoutError:
                        logger.error("LLM interaction timed out")
                        return ExecutionResult(
                            success=False,
                            branch_name=branch_name,
                            git_hash=context.state.git_hash,
                            error_message="LLM interaction timed out",
                            execution_time=time.time() - start_time,
                            repository_path=context.state.repository_path
                        )
                    except Exception as e:
                        logger.error(f"Error during LLM interaction: {str(e)}")
                        return ExecutionResult(
                            success=False,
                            branch_name=branch_name,
                            git_hash=context.state.git_hash,
                            error_message=f"Error during LLM interaction: {str(e)}",
                            execution_time=time.time() - start_time,
                            repository_path=context.state.repository_path
                        )
            
        except Exception as e:
            logger.error(f"Fatal error during task execution: {str(e)}")
            # If execution fails, clean up the branch and go back to main
            try:
                await run_terminal_cmd(
                    command=["git", "checkout", "main"], 
                    cwd=context.state.repository_path
                )
                # Don't assume we know the branch name if create_branch failed
                if 'branch_name' in locals():
                    await run_terminal_cmd(
                        command=["git", "branch", "-D", branch_name],
                        cwd=context.state.repository_path
                    )
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
                
            return ExecutionResult(
                success=False,
                branch_name=base_name,  # Use base_name since we don't have actual branch name
                git_hash=context.state.git_hash,
                error_message=str(e),
                execution_time=time.time() - start_time,
                repository_path=context.state.repository_path
            ) 