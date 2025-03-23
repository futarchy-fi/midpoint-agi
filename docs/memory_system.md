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

## Cross-Reference System

The memory system maintains a cross-reference between code repository states and memory repository states. This allows retrieval of memories that were relevant at a specific point in the code's history.

### Structure

The cross-reference information is stored in `metadata/cross-reference.json` with the following structure:

```json
{
  "mappings": [
    {
      "code_hash": "1234567890abcdef",
      "memory_hash": "abcdef1234567890",
      "timestamp": 1679012345
    },
    {
      "code_hash": "1234567890abcdef",
      "memory_hash": "fedcba0987654321",
      "timestamp": 1679012400
    }
  ],
  "latest": {
    "1234567890abcdef": "fedcba0987654321"
  }
}
```

This format supports:
- Full history of all mappings, including timestamps
- Multiple memory states for a single code state
- Quick lookup of the latest memory state for any code state

### Usage

The system provides several ways to access the cross-reference data:

1. **Get latest memory for code**:
   ```python
   memory_hash = get_memory_for_code_hash(code_hash)
   ```

2. **Get full history of memories for code**:
   ```python
   historical_mappings = get_memory_for_code_hash(code_hash, historical=True)
   ```

3. **Get memory from specific time period**:
   ```python
   memory_hash = get_memory_for_code_hash(code_hash, timestamp=specific_time)
   ```

### CLI Commands

You can also interact with the cross-reference system using the CLI:

```bash
# Link a code hash to a memory hash
python -m scripts.memory_tools link CODE_HASH MEMORY_HASH

# Look up the latest memory hash for a code hash
python -m scripts.memory_tools lookup CODE_HASH

# View all historical mappings for a code hash
python -m scripts.memory_tools lookup CODE_HASH --historical

# Find memory hash from closest timestamp
python -m scripts.memory_tools lookup CODE_HASH --timestamp=1679012345

# View complete cross-reference history
python -m scripts.memory_tools history
```

### State Reversion

The cross-reference system's historical tracking allows precise reversion to previous states:

1. **Identifying a state to revert to**:
   ```bash
   python -m scripts.memory_tools history
   ```

2. **Reverting to that state**:
   ```python
   # In code
   code_hash = "1234567890abcdef"
   timestamp = 1679012345  # Timestamp from history
   
   # Get the memory hash from that time
   memory_hash = get_memory_for_code_hash(code_hash, timestamp=timestamp)
   
   # Checkout both repositories to that state
   subprocess.run(["git", "checkout", code_hash], cwd=CODE_REPO_PATH)
   subprocess.run(["git", "checkout", memory_hash], cwd=MEMORY_REPO_PATH)
   ```

This enables the Midpoint system to travel back in time to any specific state in its development history, with both code and memory correctly synchronized. 