"""
Task Executor - Generic task execution agent for the Midpoint system.

IMPORTANT: This module implements a generic task execution system that uses LLM to interpret
and execute tasks. It MUST NOT contain any task-specific logic or hardcoded implementations.
All task-specific decisions and implementations should be handled by the LLM at runtime.
"""

import asyncio
import time
import json
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import os
import logging
from pathlib import Path

from .models import TaskContext, ExecutionTrace, State, Goal, ExecutionResult
from .tools import initialize_all_tools
from .tools.processor import ToolProcessor
from .tools.registry import ToolRegistry
from .tools.git_tools import create_branch, create_commit, get_current_hash, get_current_branch
from .tools.filesystem_tools import list_directory, read_file
from .tools.code_tools import search_code
from .tools.web_tools import web_search, web_scrape
from .tools.terminal_tools import run_terminal_cmd
from .config import get_openai_api_key
from openai import AsyncOpenAI

# Set up logging
logger = logging.getLogger('TaskExecutor')

def configure_logging(debug=False, quiet=False, log_dir_path="logs"):
    """
    Configure logging for the TaskExecutor.
    
    Args:
        debug: Whether to show debug messages on console
        quiet: Whether to only show warnings and errors
        log_dir_path: Directory to store log files
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir_path)
    log_dir.mkdir(exist_ok=True)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"task_executor_{timestamp}.log"
    
    # Create file handler for full logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    
    # Create console handler with a more concise format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Set console handler level based on arguments
    if debug:
        console_handler.setLevel(logging.DEBUG)
    elif quiet:
        console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(logging.INFO)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs
    
    # Remove existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our custom handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create a special formatter for task execution events
    class TaskExecutorFilter(logging.Filter):
        def filter(self, record):
            # Format specific log records for better clarity
            if hasattr(record, 'taskstep') and record.taskstep:
                record.msg = f"📋 Task Step: {record.msg}"
            if hasattr(record, 'tool') and record.tool:
                record.msg = f"🔧 Tool: {record.tool} - {record.msg}"
            if hasattr(record, 'validation') and record.validation:
                record.msg = f"✓ Validation: {record.msg}"
            return True
    
    console_handler.addFilter(TaskExecutorFilter())
    logger.info(f"Logging configured. Full logs will be saved to: {log_file}")
    return log_file

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
        extra = {'taskstep': True}
        logger.info(f"Starting task execution: {task}", extra=extra)
        logger.info(f"Repository: {context.state.repository_path}")
        logger.info(f"Current Git Hash: {context.state.git_hash}")
        
        # Initialize execution
        start_time = time.time()
        base_name = f"task-{context.iteration}"
        branch_name = None
        
        try:
            # Validate repository state
            try:
                current_hash = await get_current_hash(context.state.repository_path)
                current_branch = await get_current_branch(context.state.repository_path)
                logger.info(f"Current branch: {current_branch}, hash: {current_hash[:8]}")
                
                if current_hash != context.state.git_hash:
                    logger.warning(f"Repository hash mismatch. Expected: {context.state.git_hash[:8]}, Got: {current_hash[:8]}")
                    # Continue anyway but log the warning
            except Exception as e:
                logger.error(f"Error validating repository state: {str(e)}")
                raise ValueError(f"Repository validation failed: {str(e)}")
            
            # Create a new branch for this execution
            try:
                logger.info(f"Creating new branch: {base_name}", extra=extra)
                branch_name = await create_branch(context.state.repository_path, base_name)
                logger.info(f"Created branch: {branch_name}", extra=extra)
                
                # Update state with branch name
                context.state.branch_name = branch_name
            except Exception as e:
                logger.error(f"Failed to create branch: {str(e)}")
                raise ValueError(f"Branch creation failed: {str(e)}")
            
            # Create the user prompt for the LLM
            user_prompt = f"""Task: {task}

Repository Path: {context.state.repository_path}
Branch: {branch_name} (newly created for this task)
Git Hash: {context.state.git_hash}"""

            logger.debug(f"User prompt: {user_prompt}")

            # Initialize messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Maximum retries for LLM interactions
            max_retries = 3
            retry_count = 0
            
            # Add a timeout for LLM interactions
            try:
                # Use a different timeout approach for compatibility
                # with different Python versions
                async def execute_with_timeout():
                    nonlocal retry_count
                    while retry_count < max_retries:
                        try:
                            logger.info(f"Sending request to LLM (attempt {retry_count + 1}/{max_retries})", extra=extra)
                            # Get LLM response
                            response = await self.client.chat.completions.create(
                                model="gpt-4-turbo-preview",
                                messages=messages,
                                temperature=0.7,
                                max_tokens=2000
                            )
                            
                            # Parse the response
                            content = response.choices[0].message.content
                            logger.debug(f"LLM response received (length: {len(content)})")
                            
                            try:
                                # Try to parse as JSON
                                final_output = json.loads(content)
                                logger.info("Successfully parsed LLM response as JSON", extra=extra)
                            except json.JSONDecodeError as json_err:
                                # If not JSON, try to extract JSON from the content
                                logger.warning(f"Failed to parse response as JSON: {str(json_err)}")
                                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                                if json_match:
                                    try:
                                        logger.info("Attempting to extract JSON from response", extra=extra)
                                        final_output = json.loads(json_match.group())
                                        logger.info("Successfully extracted and parsed JSON from response", extra=extra)
                                    except json.JSONDecodeError as extract_err:
                                        logger.error(f"Failed to parse extracted JSON: {str(extract_err)}")
                                        raise ValueError(f"Failed to parse response: {str(extract_err)}")
                                else:
                                    logger.error("No valid JSON found in response")
                                    raise ValueError("No valid JSON found in response")
                            
                            # Validate required fields
                            required_fields = ["task_completed", "final_commit_hash"]
                            missing_fields = [field for field in required_fields if field not in final_output]
                            
                            if missing_fields:
                                logger.warning(f"Response missing required fields: {', '.join(missing_fields)}")
                                retry_count += 1
                                if retry_count >= max_retries:
                                    raise ValueError(f"Response missing required fields after {max_retries} attempts")
                                continue
                            
                            # Log validation steps if available
                            if "validation_steps" in final_output and final_output["validation_steps"]:
                                logger.info("Validation steps:", extra={'validation': True})
                                for step in final_output["validation_steps"]:
                                    logger.info(f"- {step}", extra={'validation': True})
                            
                            # Return result based on task completion
                            task_completed = final_output.get("task_completed", False)
                            final_hash = final_output.get("final_commit_hash", context.state.git_hash)
                            completion_reason = final_output.get("completion_reason", "No reason provided")
                            
                            if task_completed:
                                logger.info(f"✅ Task completed successfully", extra=extra)
                            else:
                                logger.warning(f"❌ Task failed: {completion_reason}", extra=extra)
                            
                            return ExecutionResult(
                                success=task_completed,
                                branch_name=branch_name,
                                git_hash=final_hash,
                                error_message=None if task_completed else completion_reason,
                                execution_time=time.time() - start_time,
                                repository_path=context.state.repository_path,
                                validation_results=final_output.get("validation_steps", [])
                            )
                            
                        except asyncio.TimeoutError:
                            logger.error("LLM interaction timed out")
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise
                            
                        except Exception as e:
                            logger.error(f"Error during LLM interaction: {str(e)}")
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise
                
                # Create a task with a timeout
                try:
                    return await asyncio.wait_for(execute_with_timeout(), timeout=300)  # 5-minute timeout
                except asyncio.TimeoutError:
                    logger.error("Task execution timed out after 5 minutes")
                    return ExecutionResult(
                        success=False,
                        branch_name=branch_name or base_name,
                        git_hash=context.state.git_hash,
                        error_message="Task execution timed out after 5 minutes",
                        execution_time=time.time() - start_time,
                        repository_path=context.state.repository_path
                    )
            except Exception as e:
                logger.error(f"Error during LLM interaction: {str(e)}")
                return ExecutionResult(
                    success=False,
                    branch_name=branch_name or base_name,
                    git_hash=context.state.git_hash,
                    error_message=f"Error during LLM interaction: {str(e)}",
                    execution_time=time.time() - start_time,
                    repository_path=context.state.repository_path
                )
            
        except Exception as e:
            logger.error(f"Fatal error during task execution: {str(e)}")
            import traceback
            logger.debug(f"Exception traceback: {traceback.format_exc()}")
            
            # If execution fails, clean up the branch and go back to main
            if branch_name:
                try:
                    logger.info("Cleaning up after error - switching back to main branch", extra=extra)
                    await run_terminal_cmd(
                        command="git checkout main", 
                        cwd=context.state.repository_path
                    )
                    
                    logger.info(f"Removing branch: {branch_name}", extra=extra)
                    await run_terminal_cmd(
                        command=f"git branch -D {branch_name}",
                        cwd=context.state.repository_path
                    )
                    logger.info("Cleanup completed", extra=extra)
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup: {str(cleanup_error)}")
                    logger.debug(f"Cleanup exception traceback: {traceback.format_exc()}")
            else:
                logger.info("No branch was created, so no cleanup needed", extra=extra)
                
            return ExecutionResult(
                success=False,
                branch_name=branch_name or base_name,
                git_hash=context.state.git_hash,
                error_message=str(e),
                execution_time=time.time() - start_time,
                repository_path=context.state.repository_path
            ) 