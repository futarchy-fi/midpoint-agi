"""
Terminal tools for the Midpoint agent system.

This module provides tools for interacting with the terminal,
such as running commands in a subprocess.
"""

import os
import asyncio
import logging
import shlex
from typing import List, Dict, Any, Optional, Union

from .base import Tool
from .registry import ToolRegistry

class RunTerminalCmdTool(Tool):
    """Tool for running terminal commands."""
    
    @property
    def name(self) -> str:
        return "run_terminal_cmd"
    
    @property
    def description(self) -> str:
        return "Run a terminal command in a subprocess"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to run (as a string)"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command",
                    "default": None
                },
                "env": {
                    "type": "object",
                    "description": "Environment variables to set for the command (optional)",
                    "additionalProperties": {"type": "string"}
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 60)",
                    "default": 60
                }
            },
            "required": ["command"]
        }
    
    async def execute(self, command: str, cwd: Optional[str] = None, 
                     env: Optional[Dict[str, str]] = None, 
                     timeout: int = 60) -> Dict[str, Any]:
        """Run a terminal command."""
        try:
            # Normalize working directory
            if cwd:
                cwd = os.path.abspath(cwd)
                if not os.path.exists(cwd):
                    return {
                        "success": False,
                        "error": f"Working directory does not exist: {cwd}",
                        "stdout": "",
                        "stderr": "",
                        "exit_code": None
                    }
            
            # Set up environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            # Parse command into args if it's a string
            if isinstance(command, str):
                try:
                    # Try to parse the command string into arguments
                    args = shlex.split(command)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to parse command: {str(e)}",
                        "stdout": "",
                        "stderr": "",
                        "exit_code": None
                    }
            else:
                args = command  # Assume it's already a list
            
            # Create and run the process
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=process_env
            )
            
            try:
                # Set timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                # Get the results
                stdout_text = stdout.decode('utf-8', errors='replace')
                stderr_text = stderr.decode('utf-8', errors='replace')
                exit_code = process.returncode
                
                success = exit_code == 0
                
                return {
                    "success": success,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "exit_code": exit_code
                }
                
            except asyncio.TimeoutError:
                # Try to terminate the process
                try:
                    process.terminate()
                    await asyncio.sleep(0.5)
                    if process.returncode is None:
                        process.kill()
                except:
                    pass
                
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "stdout": "",
                    "stderr": "",
                    "exit_code": None
                }
                
        except Exception as e:
            logging.error(f"Error running command: {str(e)}")
            return {
                "success": False,
                "error": f"Error running command: {str(e)}",
                "stdout": "",
                "stderr": "",
                "exit_code": None
            }

# Instantiate and register the tools
run_terminal_cmd_tool = RunTerminalCmdTool()

ToolRegistry.register_tool(run_terminal_cmd_tool)

# Export async functions for external use
async def run_terminal_cmd(command: Union[str, List[str]], cwd: Optional[str] = None, 
                          env: Optional[Dict[str, str]] = None, 
                          timeout: int = 60) -> Dict[str, Any]:
    """Run a terminal command."""
    return await run_terminal_cmd_tool.execute(
        command=command, 
        cwd=cwd, 
        env=env, 
        timeout=timeout
    ) 