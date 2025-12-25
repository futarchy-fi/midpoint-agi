"""
Filesystem tools for the Midpoint agent system.

This module provides tools for interacting with the filesystem,
such as listing directories and reading files.
"""

import os
import glob
import logging
from typing import List, Dict, Any, Optional

from .base import Tool
from .registry import ToolRegistry

class ListDirectoryTool(Tool):
    """Tool for listing the contents of a directory."""
    
    @property
    def name(self) -> str:
        return "list_directory"
    
    @property
    def description(self) -> str:
        return "List contents of a directory"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files",
                    "default": "*"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list contents recursively",
                    "default": False
                }
            },
            "required": ["path"]
        }
    
    def execute(self, path: str, pattern: str = "*", recursive: bool = False) -> Dict[str, Any]:
        """List files and directories at the given path."""
        try:
            # Validate path exists
            if not os.path.exists(path):
                return {
                    "error": f"Path does not exist: {path}",
                    "items": []
                }
                
            # Normalize path
            path = os.path.abspath(path)
            
            # Get items based on recursive flag
            items = []
            if recursive:
                # Use recursive glob with **
                search_pattern = os.path.join(path, "**", pattern)
                matches = glob.glob(search_pattern, recursive=True)
                
                # Convert to relative paths from the base directory
                items = [os.path.relpath(match, path) for match in matches]
            else:
                # Use regular glob
                search_pattern = os.path.join(path, pattern)
                matches = glob.glob(search_pattern)
                
                # Convert to relative paths from the base directory
                items = [os.path.relpath(match, path) for match in matches]
            
            # Sort items
            items.sort()
            
            # Get details for each item (type, size)
            item_details = []
            for item in items:
                full_path = os.path.join(path, item)
                is_dir = os.path.isdir(full_path)
                
                if is_dir:
                    item_type = "directory"
                    size = None
                else:
                    item_type = "file"
                    try:
                        size = os.path.getsize(full_path)
                    except:
                        size = None
                
                item_details.append({
                    "name": item,
                    "type": item_type,
                    "size": size
                })
            
            return {
                "path": path,
                "items": item_details
            }
            
        except Exception as e:
            logging.error(f"Error listing directory: {str(e)}")
            return {
                "error": f"Error listing directory: {str(e)}",
                "items": []
            }

class ReadFileTool(Tool):
    """Tool for reading the contents of a file."""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read the contents of a file"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-indexed)",
                    "default": 0
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                    "default": 100
                }
            },
            "required": ["file_path"]
        }
    
    def execute(self, file_path: str, start_line: int = 0, max_lines: int = 100) -> Dict[str, Any]:
        """Read a file and return its contents."""
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                return {
                    "error": f"File does not exist: {file_path}",
                    "content": None,
                    "start_line": start_line,
                    "lines_read": 0,
                    "total_lines": 0
                }
                
            # Normalize path
            file_path = os.path.abspath(file_path)
            
            # Check if it's a file
            if not os.path.isfile(file_path):
                return {
                    "error": f"Not a file: {file_path}",
                    "content": None,
                    "start_line": start_line,
                    "lines_read": 0,
                    "total_lines": 0
                }
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()
                
            total_lines = len(all_lines)
            
            # Validate start_line
            if start_line < 0:
                start_line = 0
            elif start_line >= total_lines:
                start_line = max(0, total_lines - 1)
            
            # Calculate end line
            end_line = min(start_line + max_lines, total_lines)
            
            # Extract the requested lines
            selected_lines = all_lines[start_line:end_line]
            content = ''.join(selected_lines)
            
            lines_read = len(selected_lines)
            
            return {
                "content": content,
                "start_line": start_line,
                "lines_read": lines_read,
                "total_lines": total_lines
            }
            
        except Exception as e:
            logging.error(f"Error reading file: {str(e)}")
            return {
                "error": f"Error reading file: {str(e)}",
                "content": None,
                "start_line": start_line,
                "lines_read": 0,
                "total_lines": 0
            }

class EditFileTool(Tool):
    """Tool for editing the contents of a file."""
    
    @property
    def name(self) -> str:
        return "edit_file"
    
    @property
    def description(self) -> str:
        return "Edit the contents of a file"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "content": {
                    "type": "string",
                    "description": "New content for the file"
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Whether to create parent directories if they don't exist",
                    "default": True
                }
            },
            "required": ["file_path", "content"]
        }
    
    def execute(self, file_path: str, content: str, create_dirs: bool = True) -> Dict[str, Any]:
        """Edit the contents of a file."""
        try:
            # Normalize path
            file_path = os.path.abspath(file_path)
            
            # Create parent directories if needed
            if create_dirs:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write the content to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Verify file was written
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"Failed to create file: {file_path}",
                    "file_path": file_path
                }
            
            # Get file info
            file_size = os.path.getsize(file_path)
            
            return {
                "success": True,
                "file_path": file_path,
                "size": file_size
            }
            
        except Exception as e:
            logging.error(f"Error editing file: {str(e)}")
            return {
                "success": False,
                "error": f"Error editing file: {str(e)}",
                "file_path": file_path
            }

# Instantiate and register the tools
list_directory_tool = ListDirectoryTool()
read_file_tool = ReadFileTool()
edit_file_tool = EditFileTool()

ToolRegistry.register_tool(list_directory_tool)
ToolRegistry.register_tool(read_file_tool)
ToolRegistry.register_tool(edit_file_tool)

# Export functions for external use
def list_directory(path: str, pattern: str = "*", recursive: bool = False) -> Dict[str, Any]:
    """List contents of a directory."""
    return list_directory_tool.execute(path=path, pattern=pattern, recursive=recursive)

def read_file(file_path: str, start_line: int = 0, max_lines: int = 100) -> Dict[str, Any]:
    """Read the contents of a file."""
    return read_file_tool.execute(file_path=file_path, start_line=start_line, max_lines=max_lines)

def edit_file(file_path: str, content: str, create_dirs: bool = True) -> Dict[str, Any]:
    """Edit the contents of a file."""
    return edit_file_tool.execute(file_path=file_path, content=content, create_dirs=create_dirs)