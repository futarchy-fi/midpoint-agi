#!/usr/bin/env python
"""
Example script demonstrating memory integration with Midpoint.

This script shows how memory state could be tracked alongside code state.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Add the src directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import from existing Midpoint modules
from src.midpoint.agents.models import State

# Import from memory tools
from memory_tools import (
    get_repo_path,
    get_current_hash,
    store_document,
    retrieve_documents,
    update_cross_reference,
    get_memory_for_code_hash
)

@dataclass
class MemoryState:
    """Represents a memory repository state."""
    memory_hash: str
    repository_path: str
    
    @classmethod
    def from_repo_path(cls, repo_path=None):
        """Create a memory state from a repository path."""
        repo_path = repo_path or get_repo_path()
        return cls(
            memory_hash=get_current_hash(repo_path),
            repository_path=repo_path
        )

@dataclass
class ExtendedState:
    """Extended state with memory repository information."""
    # Base state properties
    git_hash: str
    repository_path: str
    description: str
    branch_name: Optional[str] = None
    
    # Memory state
    memory_state: Optional[MemoryState] = None
    
    @classmethod
    def from_base_state(cls, base_state, memory_state=None):
        """Create an extended state from a base state."""
        return cls(
            git_hash=base_state.git_hash,
            repository_path=base_state.repository_path,
            description=base_state.description,
            branch_name=base_state.branch_name,
            memory_state=memory_state
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary."""
        result = {
            "git_hash": self.git_hash,
            "repository_path": self.repository_path,
            "description": self.description,
            "branch_name": self.branch_name
        }
        
        # Add memory state information if available
        if self.memory_state:
            memory_hash = getattr(self.memory_state, "memory_hash", None)
            memory_path = getattr(self.memory_state, "repository_path", None)
            if memory_hash:
                result["memory_hash"] = memory_hash
            if memory_path:
                result["memory_repository_path"] = memory_path
        
        return result

def get_context_for_state(state, query=None):
    """Get memory context for a given state."""
    if not state.memory_state:
        return "No memory available for this state."
    
    # This is a simple implementation that retrieves recent documents
    reasoning_docs = retrieve_documents(category="reasoning", limit=3, repo_path=state.memory_state.repository_path)
    observation_docs = retrieve_documents(category="observations", limit=3, repo_path=state.memory_state.repository_path)
    decision_docs = retrieve_documents(category="decisions", limit=2, repo_path=state.memory_state.repository_path)
    
    # Format documents for context
    context_parts = ["## Agent Memory\n"]
    
    if decision_docs:
        context_parts.append("### Previous Decisions")
        for i, (path, doc) in enumerate(decision_docs):
            context_parts.append(f"**Decision {i+1}** ({path}):\n{doc[:500]}")
            if len(doc) > 500:
                context_parts.append("... (truncated)")
            context_parts.append("")
    
    if reasoning_docs:
        context_parts.append("### Previous Reasoning")
        for i, (path, doc) in enumerate(reasoning_docs):
            context_parts.append(f"**Reasoning {i+1}** ({path}):\n{doc[:500]}")
            if len(doc) > 500:
                context_parts.append("... (truncated)")
            context_parts.append("")
    
    if observation_docs:
        context_parts.append("### Previous Observations")
        for i, (path, doc) in enumerate(observation_docs):
            context_parts.append(f"**Observation {i+1}** ({path}):\n{doc[:500]}")
            if len(doc) > 500:
                context_parts.append("... (truncated)")
            context_parts.append("")
    
    return "\n".join(context_parts)

def print_state_info(state: ExtendedState) -> None:
    """Print information about the state."""
    print(f"State Information:")
    print(f"  Git Hash: {state.git_hash}")
    print(f"  Repository: {state.repository_path}")
    print(f"  Branch: {state.branch_name}")
    print(f"  Description: {state.description}")
    
    if state.memory_state:
        memory_hash = getattr(state.memory_state, "memory_hash", None)
        memory_path = getattr(state.memory_state, "repository_path", None)
        if memory_hash:
            print(f"  Memory Hash: {memory_hash}")
        if memory_path:
            print(f"  Memory Repository: {memory_path}")

def main():
    """Run the example."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Memory integration example")
    parser.add_argument("--code-repo", default=os.getcwd(), help="Path to code repository")
    parser.add_argument("--memory-repo", default=get_repo_path(), help="Path to memory repository")
    parser.add_argument("--init", action="store_true", help="Initialize memory repository")
    parser.add_argument("--store", action="store_true", help="Store a sample document")
    args = parser.parse_args()
    
    # Initialize memory repository if requested
    if args.init:
        from init_memory_repo import init_memory_repo
        memory_repo_info = init_memory_repo(args.memory_repo, None, None)
        print(f"Initialized memory repository at {memory_repo_info['path']}")
    
    # Get code repository hash
    code_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.code_repo,
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()
    
    # Create base state
    base_state = State(
        git_hash=code_hash,
        repository_path=args.code_repo,
        description="Current state of the code repository"
    )
    
    # Create memory state
    memory_state = MemoryState.from_repo_path(args.memory_repo)
    
    # Create extended state
    extended_state = ExtendedState.from_base_state(base_state, memory_state)
    
    # Print state information
    print_state_info(extended_state)
    
    # Store a sample document if requested
    if args.store:
        categories = ["reasoning", "observations", "decisions"]
        
        for i, category in enumerate(categories):
            content = f"""# Sample {category.capitalize()} Document

This is a sample document in the {category} category.
It demonstrates how memory documents would be stored and retrieved.

## Details

- Category: {category}
- Created at: {subprocess.check_output(['date']).decode().strip()}
- Related code hash: {code_hash}

## Content

This document would contain the agent's {category} about tasks,
code changes, or other relevant information.
"""
            
            # Store the document
            store_document(
                content=content,
                category=category,
                metadata={
                    "id": f"sample_{category}_{i+1}",
                    "code_hash": code_hash,
                    "commit_message": f"Add sample {category} document"
                },
                repo_path=args.memory_repo
            )
            
            print(f"Stored sample {category} document")
    
    # Get and print context for the current state
    context = get_context_for_state(extended_state)
    print("\nContext for Current State:")
    print(context)
    
    # Show how to look up memory hash for a code hash
    print("\nMemory Hash Lookup:")
    memory_hash = get_memory_for_code_hash(code_hash, args.memory_repo)
    
    if memory_hash:
        print(f"The code hash {code_hash[:7]} is linked to memory hash {memory_hash[:7]}")
    else:
        print(f"No memory hash found for code hash {code_hash[:7]}")

if __name__ == "__main__":
    main() 