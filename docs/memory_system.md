# Agent Memory System

This document describes the memory system for the Midpoint agent.

## Overview

The memory system allows the agent to retain and retrieve information across sessions. It uses a separate Git repository to store memory documents, which are organized by category and linked to the code repository through commit hashes.

## Key Concepts

### Memory Repository

The memory repository is a separate Git repository that stores memory documents. It is organized into the following directories:

- `reasoning`: Contains documents that capture the agent's reasoning process
- `observations`: Contains documents that record the agent's observations
- `decisions`: Contains documents that record the agent's decisions
- `cross_references`: Contains files that link code commit hashes to memory commit hashes

### Memory State

The memory state is a representation of the current state of the memory repository. It includes:

- The current commit hash of the memory repository
- The path to the memory repository

### Extended State

The extended state combines the code state and memory state, allowing the agent to reason about both. It includes:

- The code repository hash
- The code repository path
- The memory repository hash
- The memory repository path

## Implementation

The memory system is implemented in the following files:

- `scripts/init_memory_repo.py`: Script to initialize a memory repository
- `scripts/memory_tools.py`: Utility functions for working with the memory repository
- `scripts/memory_example.py`: Example script demonstrating memory integration

### Memory Tools

The memory tools module provides the following functions:

- `get_repo_path`: Get the path to the memory repository
- `get_current_hash`: Get the current commit hash of a repository
- `store_document`: Store a document in the memory repository
- `retrieve_documents`: Retrieve documents from the memory repository
- `update_cross_reference`: Update the cross-reference between code and memory hashes
- `get_memory_for_code_hash`: Get the memory hash corresponding to a code hash

## Usage

### Initializing a Memory Repository

```bash
python scripts/init_memory_repo.py --path /path/to/memory/repo
```

### Storing a Document

```bash
python scripts/memory_tools.py store --category reasoning --id reasoning_1 --content "This is my reasoning" --code-hash abcdef1234567890
```

### Retrieving Documents

```bash
python scripts/memory_tools.py retrieve --category reasoning --limit 3
```

### Running the Example

```bash
python scripts/memory_example.py --init --store
```

## Integration

The memory system can be integrated with the existing Midpoint agent by:

1. Extending the `State` class to include memory state
2. Using the memory tools to store and retrieve documents
3. Incorporating memory context into agent prompts

## Future Work

- Integration with vector database for semantic search
- Automatic linking of code and memory states
- Enhanced memory summarization and retrieval
- Multi-agent memory sharing 