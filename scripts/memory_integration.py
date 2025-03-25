#!/usr/bin/env python
"""
Script demonstrating integration of memory system with the agent system.

This file shows how the memory system could be integrated with the existing
agent system to provide memory context to agents.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import argparse
import logging
from typing import Optional, Dict, Any
import subprocess

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import from memory tools
from scripts.memory_tools import (
    get_repo_path,
    get_current_hash,
    store_document,
    retrieve_documents,
    update_cross_reference,
    get_memory_for_code_hash
)

# Import from the existing agent system
from src.midpoint.agents.models import State
from src.midpoint.agents.agent import Agent

# Define memory state classes (these could be moved to a separate module)
from scripts.memory_example import MemoryState, ExtendedState, get_context_for_state

def get_memory_context_provider(memory_repo_path: Optional[str] = None):
    """
    Create a memory context provider function for the agent.
    
    Args:
        memory_repo_path: Path to the memory repository.
        
    Returns:
        A function that takes a state and returns memory context.
    """
    def memory_context_provider(state: State, query: Optional[str] = None) -> str:
        """
        Provide memory context for the agent based on the current state.
        
        Args:
            state: The current state of the agent.
            query: Optional query to focus the memory retrieval.
            
        Returns:
            A string containing relevant memory context.
        """
        # Get the memory repository path
        repo_path = memory_repo_path or get_repo_path()
        
        # Create a memory state
        memory_state = MemoryState.from_repo_path(repo_path)
        
        # Create an extended state
        extended_state = ExtendedState.from_base_state(state, memory_state)
        
        # Get context for the state
        context = get_context_for_state(extended_state, query)
        
        return context
    
    return memory_context_provider

def create_memory_enabled_agent(code_repo_path: str, memory_repo_path: Optional[str] = None) -> Agent:
    """
    Create an agent with memory capabilities.
    
    Args:
        code_repo_path: Path to the code repository.
        memory_repo_path: Path to the memory repository (optional).
        
    Returns:
        An agent with memory capabilities.
    """
    from src.midpoint.agents.context import get_code_context
    
    # Create a regular agent
    agent = Agent(
        model_name="gpt-4",
        system_prompt="You are a helpful AI assistant with access to memory.",
        state=State(
            git_hash="HEAD",
            repository_path=code_repo_path,
            description="Current state of the code repository"
        )
    )
    
    # Add memory context provider to the agent's context providers
    memory_provider = get_memory_context_provider(memory_repo_path)
    
    # Create a combined context provider
    def combined_context_provider(state: State, query: Optional[str] = None) -> str:
        code_context = get_code_context(state)
        memory_context = memory_provider(state, query)
        
        return f"{code_context}\n\n{memory_context}"
    
    # Set the agent's context provider
    agent.set_context_provider(combined_context_provider)
    
    return agent

def store_agent_memory(
    agent: Agent,
    content: str,
    category: str,
    metadata: Optional[Dict[str, Any]] = None,
    memory_repo_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Store a memory document for the agent.
    
    Args:
        agent: The agent to store memory for.
        content: The content of the memory document.
        category: The category of the memory document.
        metadata: Additional metadata for the document.
        memory_repo_path: Path to the memory repository (optional).
        
    Returns:
        Information about the stored document.
    """
    # Get the code hash from the agent's state
    code_hash = agent.state.git_hash
    
    # Get the memory repository path
    repo_path = memory_repo_path or get_repo_path()
    
    # Check for untracked files
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    if result.stdout:
        untracked_files = [line[3:] for line in result.stdout.splitlines() if line.startswith("??")]
        if untracked_files:
            error_msg = f"Cannot store agent memory: Found untracked files in memory repository:\n{chr(10).join(untracked_files)}\nPlease commit or remove these files before proceeding."
            raise RuntimeError(error_msg)
    
    # Create metadata if not provided
    if metadata is None:
        metadata = {}
    
    # Add required metadata
    if "id" not in metadata:
        import uuid
        metadata["id"] = f"{category}_{str(uuid.uuid4())[:8]}"
    
    metadata["code_hash"] = code_hash
    if "commit_message" not in metadata:
        metadata["commit_message"] = f"Add {category} document"
    
    # Store the document
    result = store_document(
        content=content,
        category=category,
        metadata=metadata,
        repo_path=repo_path
    )
    
    # Get the current memory hash
    memory_hash = get_current_hash(repo_path)
    
    # Update the cross-reference
    update_cross_reference(
        code_hash=code_hash,
        memory_hash=memory_hash,
        repo_path=repo_path
    )
    
    return result

def main():
    """Run the memory integration example."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Memory integration with agent system")
    parser.add_argument("--code-repo", default=os.getcwd(), help="Path to code repository")
    parser.add_argument("--memory-repo", default=get_repo_path(), help="Path to memory repository")
    parser.add_argument("--init", action="store_true", help="Initialize memory repository")
    parser.add_argument("--store", action="store_true", help="Store sample memories")
    parser.add_argument("--query", type=str, help="Query to test with the agent")
    args = parser.parse_args()
    
    # Initialize memory repository if requested
    if args.init:
        from scripts.init_memory_repo import init_memory_repo
        memory_repo_info = init_memory_repo(args.memory_repo, None, None)
        logging.info(f"Initialized memory repository at {memory_repo_info['path']}")
    
    # Create an agent with memory capabilities
    agent = create_memory_enabled_agent(args.code_repo, args.memory_repo)
    logging.info(f"Created agent with memory capabilities")
    
    # Store sample memories if requested
    if args.store:
        categories = ["reasoning", "observations", "decisions"]
        samples = [
            "I'm reasoning about how to implement the agent memory system. It seems like using a separate Git repository would provide good versioning and history tracking.",
            "I've observed that accessing the memory repository directly from the agent might cause performance issues in some cases. We might need to add caching.",
            "I've decided to organize memory by category and use cross-references to link code and memory states."
        ]
        
        for category, content in zip(categories, samples):
            result = store_agent_memory(
                agent=agent,
                content=content,
                category=category,
                memory_repo_path=args.memory_repo
            )
            logging.info(f"Stored {category} document: {result['id']}")
    
    # Test the agent with a query if provided
    if args.query:
        logging.info(f"Testing agent with query: {args.query}")
        
        # Get context for the query
        context = agent.get_context(args.query)
        logging.info(f"Context for query: {context}")
        
        # Simulate agent response (not actually calling the LLM)
        logging.info("Agent would respond based on the code and memory context")
        
        # In a real implementation, we would call the LLM here:
        # response = agent.run(args.query)
        # logging.info(f"Agent response: {response}")

if __name__ == "__main__":
    main() 