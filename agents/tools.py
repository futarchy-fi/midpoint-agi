from typing import Optional, Dict, Any, Callable
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime
import asyncio
import functools

def function_tool(func: Callable) -> Callable:
    """Decorator to mark a function as a tool.
    
    This is a placeholder for the actual function_tool decorator from OpenAI.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    
    # Add metadata to the function
    wrapper.is_tool = True
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    
    return wrapper

@function_tool
async def git_commit(message: str) -> str:
    """Commit changes to git with a descriptive message"""
    try:
        # Stage all changes
        process = await asyncio.create_subprocess_exec(
            "git", "add", ".",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return f"Error staging changes: {stderr.decode()}"
        
        # Commit with message
        process = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return f"Error committing changes: {stderr.decode()}"
        
        # Get the commit hash
        process = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return f"Error getting commit hash: {stderr.decode()}"
        
        return stdout.decode().strip()
    except Exception as e:
        return f"Error committing changes: {str(e)}"

@function_tool
async def git_checkout(hash: str) -> str:
    """Checkout a specific git commit"""
    try:
        process = await asyncio.create_subprocess_exec(
            "git", "checkout", hash,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return f"Error checking out commit: {stderr.decode()}"
        
        return f"Successfully checked out {hash}"
    except Exception as e:
        return f"Error checking out commit: {str(e)}"

@function_tool
async def read_file(file_path: str) -> str:
    """Read the contents of a file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@function_tool
async def write_file(file_path: str, content: str) -> str:
    """Write content to a file"""
    try:
        # Create parent directories if they don't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@function_tool
async def run_command(command: str) -> str:
    """Run a shell command and return its output"""
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            return f"Error running command: {stderr.decode()}"
        
        return stdout.decode()
    except Exception as e:
        return f"Error running command: {str(e)}"

@function_tool
async def list_directory(path: str) -> str:
    """List contents of a directory"""
    try:
        items = os.listdir(path)
        return json.dumps(items, indent=2)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

# Points tracking tool
@function_tool
async def track_points(action: str, points: int) -> Dict[str, Any]:
    """Track points consumed by an action"""
    return {
        "action": action,
        "points_consumed": points,
        "timestamp": str(datetime.now())
    } 