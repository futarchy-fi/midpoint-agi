"""
Code tools for the Midpoint agent system.

This module provides tools for working with code, including code search.
"""

import os
import re
import asyncio
import logging
import platform
from typing import List, Dict, Any, Optional

from .base import Tool
from .registry import ToolRegistry

class SearchCodeTool(Tool):
    """Tool for searching code in the repository."""
    
    @property
    def name(self) -> str:
        return "search_code"
    
    @property
    def description(self) -> str:
        return "Search code in the repository for a given pattern"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Pattern for files to include (e.g., '*.py')",
                    "default": "*"
                },
                "repo_path": {
                    "type": "string", 
                    "description": "Path to the repository to search in",
                    "default": "."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 20
                }
            },
            "required": ["pattern"]
        }
    
    @property
    def required_parameters(self) -> List[str]:
        return ["pattern"]
    
    async def execute(self, pattern: str, repo_path: str = ".", file_pattern: str = "*", max_results: int = 20) -> str:
        """Search the codebase for patterns."""
        # Sanitize inputs
        if not pattern:
            raise ValueError("Search pattern cannot be empty")
            
        # Use different search methods depending on platform availability
        search_methods = [
            self._search_ripgrep,
            self._search_grep,
            self._search_findstr
        ]
        
        for search_method in search_methods:
            try:
                result = await search_method(repo_path, pattern, file_pattern, max_results)
                return result
            except Exception as e:
                logging.debug(f"Search method {search_method.__name__} failed: {str(e)}")
                continue
                
        # If all methods fail, use a simple Python-based search
        return await self._search_python(repo_path, pattern, file_pattern, max_results)
    
    async def _search_ripgrep(self, repo_path: str, pattern: str, file_pattern: str, max_results: int) -> str:
        """Search using ripgrep if available."""
        cmd = ["rg", "--no-heading", "--line-number", "--max-count", str(max_results)]
        
        # Add file pattern if specified
        if file_pattern and file_pattern != "*":
            cmd.extend(["-g", file_pattern])
            
        # Add the pattern to search for
        cmd.append(pattern)
        
        # Execute the search
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode > 1:  # rg returns 1 if no matches found
            raise RuntimeError(f"Ripgrep search failed: {result.stderr}")
            
        if not result.stdout.strip():
            return f"No results found for pattern '{pattern}'"
            
        return f"Search results for '{pattern}':\n\n{result.stdout}"
    
    async def _search_grep(self, repo_path: str, pattern: str, file_pattern: str, max_results: int) -> str:
        """Search using grep if available."""
        # Prepare the file pattern
        file_glob = ""
        if file_pattern and file_pattern != "*":
            file_glob = f" --include='{file_pattern}'"
            
        # Build the grep command
        cmd = f"grep -rn {file_glob} --max-count={max_results} '{pattern}' ."
        
        # Execute the search
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode > 1:  # grep returns 1 if no matches found
            raise RuntimeError(f"Grep search failed: {result.stderr}")
            
        if not result.stdout.strip():
            return f"No results found for pattern '{pattern}'"
            
        return f"Search results for '{pattern}':\n\n{result.stdout}"
    
    async def _search_findstr(self, repo_path: str, pattern: str, file_pattern: str, max_results: int) -> str:
        """Search using findstr on Windows."""
        # Prepare the file pattern
        file_glob = "*.*"
        if file_pattern and file_pattern != "*":
            file_glob = file_pattern
            
        # Build the findstr command
        cmd = f"findstr /s /n /p /c:\"{pattern}\" {file_glob}"
        
        # Execute the search
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode > 1:  # findstr returns 1 if no matches found
            raise RuntimeError(f"Findstr search failed: {result.stderr}")
            
        if not result.stdout.strip():
            return f"No results found for pattern '{pattern}'"
            
        # Limit results
        lines = result.stdout.strip().split("\n")[:max_results]
        
        return f"Search results for '{pattern}':\n\n" + "\n".join(lines)
    
    async def _search_python(self, repo_path: str, pattern: str, file_pattern: str, max_results: int) -> str:
        """Fallback Python-based search implementation."""
        results = []
        pattern_re = re.compile(pattern)
        
        # Convert file pattern to regex if needed
        file_regex = None
        if file_pattern and file_pattern != "*":
            file_regex = re.compile(file_pattern.replace("*", ".*").replace("?", "."))
            
        # Walk through the repository
        for root, _, files in os.walk(repo_path):
            # Skip .git directory
            if ".git" in root.split(os.path.sep):
                continue
                
            for file in files:
                # Skip non-matching files
                if file_regex and not file_regex.match(file):
                    continue
                    
                # Get full file path
                file_path = os.path.join(root, file)
                
                try:
                    # Read file contents
                    with open(file_path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f, 1):
                            if pattern_re.search(line):
                                # Get relative path to repo
                                rel_path = os.path.relpath(file_path, repo_path)
                                results.append(f"{rel_path}:{i}:{line.rstrip()}")
                                
                                # Check if we've reached the max results
                                if len(results) >= max_results:
                                    break
                                    
                    # Stop if we've reached max results
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    logging.debug(f"Error reading file {file_path}: {str(e)}")
                    continue
                    
            # Stop if we've reached max results
            if len(results) >= max_results:
                break
                
        if not results:
            return f"No results found for pattern '{pattern}'"
            
        return f"Search results for '{pattern}':\n\n" + "\n".join(results)

# Instantiate and register the tools
search_code_tool = SearchCodeTool()
ToolRegistry.register_tool(search_code_tool)

# Export the tool function
async def search_code(repo_path: str, pattern: str, file_pattern: str = "*", max_results: int = 20) -> str:
    """Search the codebase for patterns."""
    return await search_code_tool.execute(
        repo_path=repo_path,
        pattern=pattern,
        file_pattern=file_pattern,
        max_results=max_results
    ) 