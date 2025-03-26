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
from .tools.memory_tools import store_memory_document, retrieve_memory_documents
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
            if hasattr(record, 'memory') and record.memory:
                record.msg = f"📚 Memory: {record.msg}"
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
        
        # Initialize tool registry and processor
        self.tool_registry = ToolRegistry()
        initialize_all_tools()
        self.tool_processor = ToolProcessor(self.client)
        
        # Create a lock for concurrent execution
        self._execution_lock = asyncio.Lock()
        
        # Generate system prompt after tool registry is initialized
        self.system_prompt = self._generate_system_prompt()
        
        # Initialize conversation buffer for memory storage
        self.conversation_buffer = []
        
        logger.info("TaskExecutor initialized successfully")

    async def _save_interaction_to_memory(self, context: TaskContext):
        """
        Save the conversation to memory, excluding memory messages and system prompts.
        """
        if not context.state.memory_repository_path or not context.state.memory_hash:
            logger.info("No memory repo or hash available, skipping _save_interaction_to_memory")
            return

        try:
            # Format conversation content
            content = "## Task Execution Conversation\n\n"
            
            # Add each relevant entry from the conversation buffer
            for entry in self.conversation_buffer:
                entry_type = entry.get("type", "unknown")
                
                # Skip tool usage entries as they're already reflected in the conversation
                if entry_type == "tool_usage":
                    continue
                    
                content += f"### {entry_type.title()}\n\n"
                
                if "user_prompt" in entry:
                    content += f"User:\n{entry['user_prompt']}\n\n"
                if "partial_response" in entry:
                    content += f"Assistant:\n{entry['partial_response']}\n\n"
                if "final_content" in entry:
                    content += f"Final Response:\n{entry['final_content']}\n\n"
            
            # Add metadata
            content += "## Metadata\n\n"
            content += f"- Task ID: {context.metadata.get('task_id', 'unknown')}\n"
            content += f"- Parent Goal: {context.metadata.get('parent_goal', 'none')}\n"
            content += f"- Repository: {context.state.repository_path}\n"
            content += f"- Branch: {context.state.branch_name}\n"
            content += f"- Git Hash: {context.state.git_hash}\n"
            content += f"- Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Store in memory
            await store_memory_document(
                content=content,
                category="task_execution",
                metadata={
                    "task_id": context.metadata.get("task_id"),
                    "parent_goal": context.metadata.get("parent_goal"),
                    "repository": context.state.repository_path,
                    "branch": context.state.branch_name,
                    "git_hash": context.state.git_hash,
                    "timestamp": datetime.datetime.now().isoformat()
                },
                memory_repo_path=context.state.memory_repository_path,
                memory_hash=context.state.memory_hash
            )
            logger.info("Successfully saved task execution conversation to memory", extra={'memory': True})
            
        except Exception as e:
            logger.error(f"Failed to store conversation to memory: {str(e)}")

    def _generate_system_prompt(self) -> str:
        """
        Generate a system prompt that includes dynamically generated tool descriptions.
        
        Returns:
            The system prompt with tool descriptions.
        """
        # Generate list of available tools and their descriptions
        tool_descriptions = []
        for tool_schema in self.tool_registry.get_tool_schemas():
            if tool_schema['type'] == 'function' and 'function' in tool_schema:
                function = tool_schema['function']
                name = function.get('name', '')
                description = function.get('description', '')
                tool_descriptions.append(f"- {name}: {description}")
        
        # Join the tool descriptions with newlines
        tool_descriptions_text = "\n".join(tool_descriptions)
        
        # Create the system prompt with the tool descriptions
        system_prompt = f"""You are a task execution agent responsible for implementing changes.
Your role is to:
1. Execute tasks using available tools
2. Track progress and report status
3. Handle errors gracefully
4. Maintain clean git state
5. Store important findings and observations in the memory repository

IMPORTANT: A new branch has been created for you to work on. You are responsible for:
1. Understanding the required changes for the task
2. EITHER making code changes in the repository OR storing information in the memory repository (or both)
3. If making code changes: Create commits with meaningful messages and return the commit hash
4. If only storing information in memory: Use memory repository tools and set task_completed to true
5. Ensuring all code changes are committed before completing

You have the following tools available:
{tool_descriptions_text}

Task completion criteria:
1. A task is considered complete if EITHER:
   a. You made code changes and committed them, OR
   b. You stored information in the memory repository without code changes
2. A task is considered failed if:
   a. You cannot complete the requested task for any reason, OR
   b. You have uncommitted code changes, OR
   c. You neither made code changes nor stored anything in memory

For each task:
1. Analyze the current repository state
2. Determine what changes are needed (code changes, memory storage, or both)
3. Use the available tools to implement those changes
4. Validate the changes
5. Create appropriate commits if code was changed
6. Store relevant information in the memory repository

When you've completed the task or determined it cannot be completed, provide a final response with:
1. A summary of what you did
2. Whether the task was completed successfully
3. Any validation steps that can be taken to verify the changes
4. The git hash of the final commit, if you made code changes
5. References to any memory documents you created

For exploratory or study tasks, use the memory repository to store your findings rather than making code changes."""
        
        return system_prompt

    async def execute_task(self, context: TaskContext) -> ExecutionResult:
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
        6. Store relevant information in memory repository
        
        Args:
            context: The current task context
            
        Returns:
            ExecutionResult containing the execution outcome
        """
        try:
            # Log task information
            logger.info(f"Executing task: {context.task}")
            logger.info(f"Repository: {context.state.repository_path}")
            logger.info(f"Branch: {context.state.branch_name}")
            
            # Create the user prompt for the LLM
            user_prompt = f"""Task: {context.task}

Repository Path: {context.state.repository_path}
Branch: {context.state.branch_name}
Git Hash: {context.state.git_hash}"""
            
            # Execute the task
            result = await self._execute_task_with_llm(user_prompt, context)
            
            # Return the result
            return ExecutionResult(
                success=True,
                result=result,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            return ExecutionResult(
                success=False,
                result=None,
                error=str(e)
            )

    async def _execute_task_with_llm(self, user_prompt: str, context: TaskContext) -> str:
        """Execute a task using the LLM."""
        # TODO: Implement LLM-based task execution
        # For now, just return a placeholder result
        return "Task executed successfully"

    async def is_git_ancestor(self, repo_path: str, ancestor_hash: str, descendant_hash: str) -> bool:
        """
        Check if one hash is an ancestor of another in a git repository.
        
        Args:
            repo_path: Path to the git repository
            ancestor_hash: The hash to check if it's an ancestor
            descendant_hash: The hash to check if it's a descendant
            
        Returns:
            True if ancestor_hash is an ancestor of descendant_hash, False otherwise
        """
        if ancestor_hash == descendant_hash:
            return True
        
        try:
            # Use git merge-base --is-ancestor to check if one hash is an ancestor of another
            process = await asyncio.create_subprocess_exec(
                "git", "merge-base", "--is-ancestor", ancestor_hash, descendant_hash,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for the command to complete
            await process.communicate()
            
            # The command returns 0 if it's an ancestor, 1 if not
            return process.returncode == 0
        except Exception as e:
            logger.error(f"Error checking git ancestry: {str(e)}")
            return False 