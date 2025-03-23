"""
Git operations and utility functions for the Midpoint system.
"""

import os
import asyncio
import random
import string
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

async def check_repo_state(repo_path: str) -> Dict[str, bool]:
    """Check the current state of the repository."""
    repo_path = Path(repo_path)
    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
        
    # Check if it's a git repository
    if not (repo_path / ".git").exists():
        raise ValueError(f"Not a git repository: {repo_path}")
        
    # Get git status
    result = await asyncio.create_subprocess_exec(
        "git", "status", "--porcelain",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Git status failed: {stderr.decode()}")
        
    status = stdout.decode()
    
    # Check for various states
    has_uncommitted = bool(status)
    has_untracked = any(line.startswith("??") for line in status.splitlines())
    
    return {
        "is_clean": not any([has_uncommitted, has_untracked]),
        "has_uncommitted": has_uncommitted,
        "has_untracked": has_untracked,
        "has_merge_conflicts": False,
        "has_rebase_conflicts": False
    }

async def create_branch(repo_path: str, base_name: str) -> str:
    """Create a new branch with a random suffix."""
    # Generate random suffix
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    branch_name = f"{base_name}-{suffix}"
    
    # Create and checkout branch
    result = await asyncio.create_subprocess_exec(
        "git", "checkout", "-b", branch_name,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create branch: {stderr.decode()}")
        
    return branch_name

async def revert_to_hash(repo_path: str, git_hash: str) -> None:
    """Revert the repository to a specific git hash."""
    # First check if we have uncommitted changes
    state = await check_repo_state(repo_path)
    if not state["is_clean"]:
        raise RuntimeError("Cannot revert: repository has uncommitted changes")
        
    # Hard reset to the hash
    result = await asyncio.create_subprocess_exec(
        "git", "reset", "--hard", git_hash,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to revert to hash: {stderr.decode()}")

async def create_commit(repo_path: str, message: str) -> str:
    """Create a commit with the given message."""
    # Add all changes
    add_result = await asyncio.create_subprocess_exec(
        "git", "add", ".",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, add_stderr = await add_result.communicate()
    
    if add_result.returncode != 0:
        raise RuntimeError(f"Failed to add changes: {add_stderr.decode()}")
        
    # Create commit
    commit_result = await asyncio.create_subprocess_exec(
        "git", "commit", "-m", message,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, commit_stderr = await commit_result.communicate()
    
    if commit_result.returncode != 0:
        raise RuntimeError(f"Failed to create commit: {commit_stderr.decode()}")
        
    # Get the commit hash
    hash_result = await asyncio.create_subprocess_exec(
        "git", "rev-parse", "HEAD",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, hash_stderr = await hash_result.communicate()
    
    if hash_result.returncode != 0:
        raise RuntimeError(f"Failed to get commit hash: {hash_stderr.decode()}")
        
    return stdout.decode().strip()

async def get_current_hash(repo_path: str) -> str:
    """Get the current git hash."""
    result = await asyncio.create_subprocess_exec(
        "git", "rev-parse", "HEAD",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get current hash: {stderr.decode()}")
        
    return stdout.decode().strip()

async def get_current_branch(repo_path: str) -> str:
    """Get the current branch name."""
    result = await asyncio.create_subprocess_exec(
        "git", "rev-parse", "--abbrev-ref", "HEAD",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get current branch: {stderr.decode()}")
        
    return stdout.decode().strip()

async def checkout_branch(repo_path: str, branch_name: str) -> None:
    """Checkout a specific branch."""
    # First check if we have uncommitted changes
    state = await check_repo_state(repo_path)
    if not state["is_clean"]:
        raise RuntimeError("Cannot checkout: repository has uncommitted changes")
        
    result = await asyncio.create_subprocess_exec(
        "git", "checkout", branch_name,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to checkout branch: {stderr.decode()}")

# Repository exploration tools
async def list_directory(repo_path: str, directory: str = ".") -> Dict[str, List[str]]:
    """
    List the contents of a directory in the repository.
    
    Args:
        repo_path: Path to the git repository
        directory: Directory to list within the repository
        
    Returns:
        Dictionary with 'files' and 'directories' keys
    """
    full_path = Path(repo_path) / directory
    logging.debug(f"Attempting to list directory: {full_path}")
    
    if not full_path.exists():
        error_msg = f"Directory does not exist: {directory}"
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    result = {
        "files": [],
        "directories": []
    }
    
    try:
        for item in full_path.iterdir():
            if item.is_file():
                result["files"].append(item.name)
            elif item.is_dir() and item.name != ".git":
                result["directories"].append(item.name)
                
        logging.debug(f"Successfully listed directory {directory}: found {len(result['files'])} files and {len(result['directories'])} directories")
    except Exception as e:
        logging.error(f"Error listing directory {directory}: {str(e)}")
        raise
            
    return result

async def read_file(repo_path: str, file_path: str, start_line: int = 0, max_lines: int = 100) -> str:
    """
    Read the contents of a file in the repository.
    
    Args:
        repo_path: Path to the git repository
        file_path: Path to the file within the repository
        start_line: First line to read (0-indexed)
        max_lines: Maximum number of lines to read
        
    Returns:
        Contents of the file as a string
    """
    full_path = Path(repo_path) / file_path
    
    if not full_path.exists():
        raise ValueError(f"File does not exist: {full_path}")
        
    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Calculate end line, ensuring we don't go past the end of the file
    end_line = min(start_line + max_lines, len(lines))
    
    # Get the requested lines
    content = "".join(lines[start_line:end_line])
    
    # Add line count information
    total_lines = len(lines)
    result = f"File: {file_path} (Lines {start_line+1}-{end_line} of {total_lines})\n\n{content}"
    
    if end_line < total_lines:
        result += f"\n\n[...{total_lines - end_line} more lines...]"
        
    return result

async def search_code(repo_path: str, pattern: str, file_pattern: str = "*", max_results: int = 20) -> str:
    """
    Search the codebase for a pattern.
    
    Args:
        repo_path: Path to the git repository
        pattern: Regular expression pattern to search for
        file_pattern: Pattern for files to include (e.g., "*.py")
        max_results: Maximum number of results to return
        
    Returns:
        Search results as a string
    """
    repo_path = Path(repo_path)
    
    # Use grep-like command for efficiency
    try:
        # Use different commands based on platform
        if os.name == "nt":  # Windows
            cmd = ["findstr", "/s", "/n", pattern]
            if file_pattern != "*":
                # Windows findstr doesn't support file patterns directly
                # This is a simplification; in practice you might need more complex handling
                pass
        else:  # Unix-like
            cmd = ["grep", "-r", "-n", pattern, "--include=" + file_pattern]
            
        result = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode > 1:  # grep returns 1 if no matches, 0 if matches, >1 if error
            raise RuntimeError(f"Search failed: {stderr.decode()}")
            
        output = stdout.decode()
        lines = output.splitlines()
        
        # Truncate if too many results
        if len(lines) > max_results:
            return f"Found {len(lines)} matches. Showing first {max_results}:\n\n" + "\n".join(lines[:max_results])
        elif len(lines) > 0:
            return f"Found {len(lines)} matches:\n\n" + output
        else:
            return "No matches found."
            
    except Exception as e:
        # Fallback to Python-based search if command-line tools fail
        matches = []
        count = 0
        
        for path in repo_path.rglob(file_pattern):
            if path.is_file() and ".git" not in str(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f, 1):
                            if re.search(pattern, line):
                                rel_path = path.relative_to(repo_path)
                                matches.append(f"{rel_path}:{i}: {line.strip()}")
                                count += 1
                                if count >= max_results:
                                    break
                except Exception:
                    # Skip files that can't be read
                    pass
                    
                if count >= max_results:
                    break
                    
        if matches:
            return f"Found {count} matches:\n\n" + "\n".join(matches)
        else:
            return "No matches found."

async def edit_file(repo_path: str, file_path: str, content: str, create_if_missing: bool = False) -> None:
    """
    Edit or create a file in the repository.
    
    Args:
        repo_path: Path to the git repository
        file_path: Path to the file within the repository
        content: New content for the file
        create_if_missing: Whether to create the file if it doesn't exist
    """
    full_path = Path(repo_path) / file_path
    
    # Check if file exists
    if not full_path.exists() and not create_if_missing:
        raise ValueError(f"File does not exist: {full_path}")
    
    # Create parent directories if they don't exist
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the content
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

async def run_terminal_cmd(command: str, cwd: str) -> Tuple[str, str]:
    """
    Run a terminal command in the specified directory.
    
    Args:
        command: The command to run
        cwd: The working directory to run the command in
        
    Returns:
        Tuple of (stdout, stderr) as strings
        
    Raises:
        RuntimeError: If the command fails
    """
    # Split command into parts
    if isinstance(command, str):
        command = command.split()
        
    result = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {stderr.decode()}")
        
    return stdout.decode(), stderr.decode()

async def validate_repository_state(repo_path: str, expected_hash: str) -> None:
    """
    Validate that the repository is in the expected state.
    
    Args:
        repo_path: Path to the git repository
        expected_hash: Expected git hash to validate against
        
    Raises:
        ValueError: If repository is not in the expected state
    """
    # First check if repository is clean
    state = await check_repo_state(repo_path)
    if not state["is_clean"]:
        raise ValueError("Repository is not in a clean state")
        
    # Get current hash
    current_hash = await get_current_hash(repo_path)
    if current_hash != expected_hash:
        raise ValueError(f"Repository hash mismatch. Expected: {expected_hash}, Got: {current_hash}")

async def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily's API, which is optimized for AI agents.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        Search results as a formatted string
    """
    from tavily import TavilyClient
    import os
    
    # Get API key from environment
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable not set"
        
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
        
        # Format the results
        results = []
        
        # Add results
        for result in response.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            
            if title and content:
                results.append(f"Title: {title}\nContent: {content}\nURL: {url}\n")
        
        if results:
            return "\n".join(results)
        else:
            return "No results found."
            
    except Exception as e:
        return f"Error performing Tavily search: {str(e)}"

async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using both DuckDuckGo and Tavily APIs for comprehensive results.
    Falls back to DuckDuckGo if Tavily is not available or API key is not set.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        Combined search results as a formatted string
    """
    # Get results from DuckDuckGo first
    duck_results = await duckduckgo_search(query, max_results)
    
    # Try to get Tavily results if available
    try:
        tavily_results = await tavily_search(query, max_results)
    except Exception:
        tavily_results = None
    
    # Combine results
    combined_results = []
    
    if duck_results and duck_results != "No results found.":
        combined_results.append("=== DuckDuckGo Results ===\n" + duck_results)
    
    if tavily_results and tavily_results != "No results found." and "Error:" not in tavily_results:
        combined_results.append("\n=== Tavily Results ===\n" + tavily_results)
    
    if combined_results:
        return "\n".join(combined_results)
    else:
        return "No results found from either source."

async def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo's API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        Search results as a formatted string
    """
    import aiohttp
    import json
    
    # DuckDuckGo API endpoint
    url = "https://api.duckduckgo.com/"
    
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return f"Error: Search failed with status {response.status}"
                    
                data = await response.json()
                
                # Format the results
                results = []
                
                # Add abstract if available
                if data.get("Abstract"):
                    results.append(f"Abstract: {data['Abstract']}")
                
                # Add related topics
                if data.get("RelatedTopics"):
                    for topic in data["RelatedTopics"][:max_results]:
                        if "Text" in topic:
                            results.append(f"- {topic['Text']}")
                
                if results:
                    return "\n\n".join(results)
                else:
                    return "No results found."
                    
    except Exception as e:
        return f"Error performing DuckDuckGo search: {str(e)}"

async def web_scrape(url: str) -> str:
    """
    Scrape content from a webpage.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Scraped content as a string
    """
    import aiohttp
    from bs4 import BeautifulSoup
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"Error: Failed to fetch URL with status {response.status}"
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                return text
                
    except Exception as e:
        return f"Error scraping webpage: {str(e)}" 