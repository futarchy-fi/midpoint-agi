import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from .models import TaskContext, ExecutionTrace, State, Goal, ExecutionResult
from .goal_decomposer import validate_repository_state
from .tools import (
    get_current_hash,
    track_points,
    list_directory,
    read_file,
    search_code,
    create_branch,
    create_commit,
    run_terminal_cmd,
    edit_file
)

class TaskExecutor:
    """Agent responsible for executing tasks identified by the GoalDecomposer."""
    
    def __init__(self):
        self.system_prompt = """You are a task execution agent responsible for implementing code changes.
Your role is to:
1. Execute tasks using available tools
2. Track progress and report status
3. Handle errors gracefully
4. Maintain clean git state

Available tools:
- list_directory: List contents of a directory
- read_file: Read contents of a file
- search_code: Search for code patterns
- create_branch: Create a new git branch
- create_commit: Create a git commit
- run_terminal_cmd: Run a terminal command
- edit_file: Edit the contents of a file

Always ensure the repository is in a clean state before and after execution.
Create meaningful git commits for each successful execution step."""

    async def execute_task(self, context: TaskContext, task: str) -> ExecutionResult:
        """
        Execute a task and return the execution result.
        
        Args:
            context: The current task context
            task: The task description to execute
            
        Returns:
            ExecutionResult containing the execution outcome
            
        Raises:
            ValueError: If repository validation fails
            Exception: For other errors during execution
        """
        # Validate repository state
        await validate_repository_state(
            context.state.repository_path,
            context.state.git_hash
        )
        
        # Initialize execution
        start_time = time.time()
        points_consumed = 0
        branch_name = f"task-{context.iteration}"
        
        try:
            # Create a new branch for this execution
            await create_branch(context.state.repository_path, branch_name)
            points_consumed += 5  # Points for branch creation
            
            # Execute the task using available tools
            task_lower = task.lower()
            
            # First gather information about what we need to do
            files_to_check = []
            if "file" in task_lower:
                # Search for file names in quotes
                file_matches = re.findall(r'["\']([^"\']+)["\']', task)
                files_to_check.extend(file_matches)
            
            # If we need to search code
            if "search" in task_lower or "find" in task_lower:
                search_results = await search_code(
                    context.state.repository_path,
                    task.split("search for ")[-1].split("find ")[-1].split()[0],
                    max_results=10
                )
                points_consumed += 10  # Points for code search
            
            # If we need to read files
            for file_path in files_to_check:
                try:
                    content = await read_file(
                        context.state.repository_path,
                        file_path,
                        max_lines=100
                    )
                    points_consumed += 5  # Points for reading file
                except ValueError:
                    # File doesn't exist, might need to create it
                    pass
            
            # Handle different types of tasks
            if "create" in task_lower and "folder" in task_lower:
                # Extract folder name from task
                folder_name = next((m for m in re.findall(r'["\']([^"\']+)["\']', task)), None)
                if not folder_name:
                    raise ValueError("Could not determine folder name from task description")
                
                await run_terminal_cmd(
                    command=f"mkdir -p {folder_name}",
                    cwd=context.state.repository_path
                )
                points_consumed += 5  # Points for folder creation
                
            elif "create" in task_lower and "file" in task_lower:
                # Extract file name and content
                file_name = files_to_check[0] if files_to_check else None
                if not file_name:
                    raise ValueError("Could not determine file name from task description")
                
                # Extract content between triple quotes if present
                content_match = re.search(r'"""(.*?)"""', task, re.DOTALL)
                content = content_match.group(1) if content_match else ""
                
                await edit_file(
                    context.state.repository_path,
                    file_name,
                    content,
                    create_if_missing=True
                )
                points_consumed += 10  # Points for file creation
                
            elif "edit" in task_lower or "modify" in task_lower:
                file_name = files_to_check[0] if files_to_check else None
                if not file_name:
                    raise ValueError("Could not determine file name from task description")
                
                # Get current content
                try:
                    current_content = await read_file(
                        context.state.repository_path,
                        file_name,
                        max_lines=1000
                    )
                    points_consumed += 5  # Points for reading file
                except ValueError as e:
                    raise ValueError(f"Cannot edit non-existent file: {file_name}")
                
                # Extract new content between triple quotes if present
                content_match = re.search(r'"""(.*?)"""', task, re.DOTALL)
                new_content = content_match.group(1) if content_match else current_content
                
                await edit_file(
                    context.state.repository_path,
                    file_name,
                    new_content,
                    create_if_missing=False
                )
                points_consumed += 15  # Points for file modification
                
            else:
                # For more complex tasks, we might need to:
                # 1. Parse the task more carefully
                # 2. Use multiple tools in sequence
                # 3. Handle dependencies between actions
                raise NotImplementedError(f"Task type not yet implemented: {task}")
            
            # Get the final git hash after execution
            final_hash = await get_current_hash(context.state.repository_path)
            
            # Create a commit with the changes
            await create_commit(
                context.state.repository_path,
                f"Task execution completed: {task}"
            )
            points_consumed += 5  # Points for commit
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create the execution result
            result = ExecutionResult(
                success=True,
                branch_name=branch_name,
                git_hash=final_hash,
                execution_time=execution_time,
                points_consumed=points_consumed
            )
            
            return result
            
        except Exception as e:
            # If execution fails, clean up the branch
            try:
                await run_terminal_cmd(
                    command=f"git checkout main && git branch -D {branch_name}",
                    cwd=context.state.repository_path
                )
            except:
                pass  # Ignore cleanup errors
                
            # Create failure result
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                branch_name=branch_name,
                git_hash=context.state.git_hash,  # Keep original hash on failure
                error_message=str(e),
                execution_time=execution_time,
                points_consumed=points_consumed
            ) 