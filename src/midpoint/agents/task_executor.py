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
        
        logger.info("TaskExecutor initialized successfully")

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
        6. Store relevant information in memory repository
        
        Args:
            context: The current task context
            task: The task description to execute
            
        Returns:
            ExecutionResult containing the execution outcome
        """
        # Use a lock to prevent concurrent execution of the same task
        async with self._execution_lock:
            extra = {'taskstep': True}
            logger.info(f"Starting task execution: {task}", extra=extra)
            logger.info(f"Repository: {context.state.repository_path}")
            logger.info(f"Current Git Hash: {context.state.git_hash}")
            
            # Log memory repository information if available
            if context.state.memory_repository_path and context.state.memory_hash:
                logger.info(f"Memory Repository: {context.state.memory_repository_path}")
                logger.info(f"Memory Hash: {context.state.memory_hash}")
            
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

                # Add memory repository info if available
                if context.state.memory_repository_path and context.state.memory_hash:
                    user_prompt += f"""

Memory Repository Path: {context.state.memory_repository_path}
Memory Hash: {context.state.memory_hash}

You should use the memory repository to store important findings, decisions, and observations. 
Use the store_memory_document tool to save this information with appropriate categories:
- 'reasoning' for explanations of your thought process
- 'observations' for interesting discoveries
- 'decisions' for important choices you make
- 'study' for research and learning outcomes"""

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
                                logger.info(f"Sending request to LLM with tool processing (attempt {retry_count + 1}/{max_retries})", extra=extra)
                                
                                # Use the ToolProcessor to handle the complete tool execution loop
                                final_message, tool_usage = await self.tool_processor.run_llm_with_tools(
                                    messages=messages,
                                    model="gpt-4-turbo-preview",
                                    temperature=0.7,
                                    max_tokens=2000,
                                    validate_json_format=False
                                )
                                
                                # Log tool usage
                                if tool_usage:
                                    logger.info(f"Model used {len(tool_usage)} tools during execution", extra=extra)
                                    for tool in tool_usage:
                                        logger.debug(f"Used tool: {tool['tool']} with args: {tool['args']}")
                                
                                # Get the content from the final message
                                content = final_message.content
                                logger.debug(f"LLM final response received (length: {len(content)})")
                                
                                # Analyze the content to determine task status and outcomes
                                # This is in free-form text now, not structured JSON
                                
                                # Check for task completion indicators
                                task_completed = True  # Default to true unless we find indicators of failure
                                completion_reason = ""
                                memory_documents = []
                                made_code_changes = False
                                final_hash = context.state.git_hash  # Default to original hash
                                validation_steps = []
                                
                                # Extract patterns for validation steps
                                validation_pattern = r"validation steps?:?\s*[\n\*\-]+(.*?)(?:\n\n|\n#|\Z)"
                                validation_match = re.search(validation_pattern, content, re.IGNORECASE | re.DOTALL)
                                if validation_match:
                                    validation_text = validation_match.group(1).strip()
                                    # Split by newlines and bullet points
                                    validation_steps = [step.strip().lstrip("-*•").strip() for step in re.split(r"\n+|\*|\-|•", validation_text) if step.strip()]
                                
                                # Extract if task is completed or not
                                if re.search(r"task (was |has been )?(not|couldn'?t) (be )?(completed|finished|successful)", content, re.IGNORECASE):
                                    task_completed = False
                                    # Try to find the reason
                                    reason_pattern = r"(reason|because|due to|error)[:\s]+(.*?)(?:\n\n|\n#|\Z)"
                                    reason_match = re.search(reason_pattern, content, re.IGNORECASE | re.DOTALL)
                                    if reason_match:
                                        completion_reason = reason_match.group(2).strip()
                                    else:
                                        completion_reason = "Task could not be completed"
                                
                                # Check for commit hash references
                                commit_pattern = r"(commit hash|final hash|git hash)[:\s]+([a-f0-9]{7,40})"
                                commit_match = re.search(commit_pattern, content, re.IGNORECASE)
                                if commit_match:
                                    final_hash = commit_match.group(2).strip()
                                    made_code_changes = True
                                
                                # Check for memory document references
                                memory_pattern = r"memory document[s]?[:\s]+(.*?)(?:\n\n|\n#|\Z)"
                                memory_match = re.search(memory_pattern, content, re.IGNORECASE | re.DOTALL)
                                if memory_match:
                                    memory_text = memory_match.group(1).strip()
                                    # Try to extract individual documents with categories
                                    memory_items = re.findall(r"['\"]?([a-zA-Z0-9_]+)['\"]?\s*\(([a-zA-Z0-9_]+)\)", memory_text)
                                    for path, category in memory_items:
                                        memory_documents.append({"document_path": path, "category": category})
                                    
                                    # If we didn't find any with that pattern, try a simpler approach
                                    if not memory_documents:
                                        # Just split by commas, newlines, or bullets
                                        memory_items = [item.strip().lstrip("-*•").strip() for item in re.split(r"\n+|\*|\-|•|,", memory_text) if item.strip()]
                                        for item in memory_items:
                                            # Try to extract category if in format "path (category)"
                                            cat_match = re.search(r"(.*?)\s*\(([a-zA-Z0-9_]+)\)", item)
                                            if cat_match:
                                                path = cat_match.group(1).strip().strip("'\"")
                                                category = cat_match.group(2).strip()
                                                memory_documents.append({"document_path": path, "category": category})
                                            else:
                                                # Just use the item as path and guess category
                                                memory_documents.append({"document_path": item, "category": "unknown"})
                                
                                # Check for tool errors that might indicate task failure
                                if "Error:" in content and not task_completed:
                                    if not completion_reason:
                                        error_match = re.search(r"Error:?\s+(.*?)(?:\n\n|\n#|\Z)", content, re.IGNORECASE | re.DOTALL)
                                        if error_match:
                                            completion_reason = error_match.group(1).strip()
                                
                                # Detect uncommitted changes in the code repository
                                try:
                                    status_result = await run_terminal_cmd(
                                        command="git status --porcelain",
                                        cwd=context.state.repository_path
                                    )
                                    # Extract stdout from the result dictionary
                                    status_output = status_result.get("stdout", "")
                                    has_uncommitted_changes = bool(status_output.strip())
                                    
                                    if has_uncommitted_changes and task_completed:
                                        # Remove the exception for informational tasks
                                        logger.warning("Task marked as completed but uncommitted changes exist in repository")
                                        task_completed = False
                                        completion_reason = "Task has uncommitted changes in the repository"
                                except Exception as e:
                                    logger.error(f"Error checking for uncommitted changes: {str(e)}")
                                
                                # If no memory documents and no code changes, but marked complete, it's an error
                                if task_completed and not memory_documents and not made_code_changes:
                                    # Remove the exception for informational tasks
                                    logger.warning("Task marked as completed but no changes were made to code or memory")
                                    task_completed = False
                                    completion_reason = "Task did not make any changes to code or memory repositories"
                                
                                # Check if this is an exploratory task that's just getting started
                                # We'll look for phrases indicating the model is still exploring
                                is_exploratory = False
                                exploration_phrases = ["begin exploring", "start by", "first step", "initial exploration", 
                                                       "starting with", "gather information", "looking through", "search for"]
                                
                                for phrase in exploration_phrases:
                                    if phrase in content.lower():
                                        is_exploratory = True
                                        break
                                
                                # For exploratory tasks, create an initial memory document if needed
                                if not task_completed and not memory_documents and not made_code_changes and is_exploratory:
                                    logger.info("This appears to be an exploratory task in its initial step.")
                                    logger.info("Creating an initial memory document to track progress.")
                                    
                                    # Create a memory document to track the task progress
                                    try:
                                        # Create a summary based on the model's response
                                        memory_content = f"""## Study Session Progress Tracker
                                        
                                        ### Task
                                        {task}
                                        
                                        ### Current Status
                                        Initial exploration phase. The task is in progress.
                                        
                                        ### Model's Response
                                        {content[:2000]}  # Truncate if too long
                                        
                                        ### Findings So Far
                                        No findings yet, exploration just beginning.
                                        """
                                        
                                        await store_memory_document(
                                            content=memory_content,
                                            category="study",
                                            metadata={"task": task, "status": "in_progress"},
                                            memory_repo_path=context.state.memory_repository_path
                                        )
                                        
                                        # Update the response data
                                        memory_documents = [{"category": "study", "document_path": "task_progress.md"}]
                                        task_completed = True
                                        logger.info("Created initial progress tracking document in memory repository")
                                    except Exception as e:
                                        logger.error(f"Failed to create initial memory document: {str(e)}")
                                
                                if task_completed:
                                    logger.info(f"✅ Task completed successfully", extra=extra)
                                else:
                                    logger.warning(f"❌ Task failed: {completion_reason}", extra=extra)
                                
                                # Create final state object
                                current_hash = context.state.git_hash
                                if made_code_changes:
                                    # Get the latest hash if we made code changes
                                    try:
                                        current_hash = await get_current_hash(context.state.repository_path)
                                    except Exception as e:
                                        logger.error(f"Failed to get current hash: {str(e)}")
                                
                                # Also get current memory hash if applicable
                                memory_hash = context.state.memory_hash
                                if context.state.memory_repository_path:
                                    try:
                                        memory_hash = await get_current_hash(context.state.memory_repository_path)
                                    except Exception as e:
                                        logger.error(f"Failed to get memory hash: {str(e)}")
                                
                                # Create final state
                                final_state = State(
                                    git_hash=current_hash,
                                    description=f"State after executing task: {task}",
                                    repository_path=context.state.repository_path,
                                    memory_hash=memory_hash,
                                    memory_repository_path=context.state.memory_repository_path
                                )
                                
                                return ExecutionResult(
                                    success=task_completed,
                                    branch_name=branch_name,
                                    git_hash=final_hash,
                                    error_message=None if task_completed else completion_reason,
                                    execution_time=time.time() - start_time,
                                    repository_path=context.state.repository_path,
                                    validation_results=validation_steps,
                                    final_state=final_state
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
                    
                    # Use asyncio.wait_for with a timeout
                    return await asyncio.wait_for(execute_with_timeout(), timeout=300)  # 5 minute timeout
                    
                except asyncio.TimeoutError:
                    logger.error("Task execution timed out after 5 minutes")
                    return ExecutionResult(
                        success=False,
                        branch_name=branch_name,
                        git_hash=context.state.git_hash,
                        error_message="Task execution timed out after 5 minutes",
                        execution_time=time.time() - start_time,
                        repository_path=context.state.repository_path
                    )
                    
            except Exception as e:
                logger.error(f"Task execution failed: {str(e)}")
                return ExecutionResult(
                    success=False,
                    branch_name=branch_name,
                    git_hash=context.state.git_hash,
                    error_message=str(e),
                    execution_time=time.time() - start_time,
                    repository_path=context.state.repository_path
                ) 