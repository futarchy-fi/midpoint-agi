#!/usr/bin/env python
"""
Basic tools for working with the memory repository.

This script provides simple functions for storing and retrieving documents.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def get_repo_path():
    """Get the memory repository path from environment variables."""
    load_dotenv()
    return os.getenv("MEMORY_REPO_PATH", os.path.expanduser("~/.midpoint/memory"))

def get_current_hash(repo_path):
    """Get the current commit hash of the memory repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

def store_document(content, category, metadata=None, repo_path=None, memory_hash=None):
    """
    Store a document in the memory repository.
    
    Args:
        content: Content to store in the document
        category: Category to store the document under
        metadata: Additional metadata to store with the document
        repo_path: Path to the memory repository
        memory_hash: Required hash to operate on - must check out this hash before making changes
        
    Returns:
        Dictionary with document path, memory hash, and other metadata
    """
    # Get repository path
    repo_path = repo_path or get_repo_path()
    repo_path = Path(repo_path)
    
    # Default metadata
    metadata = metadata or {}
    
    # Validate there's either memory_hash parameter or in metadata
    target_hash = memory_hash or metadata.get("memory_hash")
    if not target_hash:
        raise ValueError("Memory hash is required - either as parameter or in metadata['memory_hash']")
    
    # Create a filename based on metadata
    filename = f"{metadata.get('id', 'doc')}_{int(time.time())}.md"
    doc_path = repo_path / "documents" / category / filename
    
    # Ensure directory exists
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep track of original position to restore later
    original_branch = None

    try:
        # Save the current branch or commit we're on
        try:
            result = subprocess.run(
                ["git", "symbolic-ref", "--short", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            # If symbolic-ref succeeds, we're on a branch
            if result.returncode == 0:
                original_branch = result.stdout.strip()
            else:
                # If symbolic-ref fails, we're in detached HEAD state
                original_branch = get_current_hash(repo_path)
            
            logging.debug(f"Memory: Saved original position: {original_branch}")
        except Exception as e:
            logging.warning(f"Memory: Failed to get current branch: {str(e)}")
            original_branch = None
        
        # Check if already at target_hash
        current_hash = get_current_hash(repo_path)
        if current_hash != target_hash:
            # Check if the target hash exists
            hash_check = subprocess.run(
                ["git", "cat-file", "-t", target_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            
            if hash_check.returncode == 0:
                logging.debug(f"Memory: Checking out target hash: {target_hash}")
                # Checkout the target hash
                checkout_result = subprocess.run(
                    ["git", "checkout", target_hash],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                
                if checkout_result.returncode != 0:
                    error_msg = checkout_result.stderr
                    logging.error(f"Memory: Failed to checkout target hash: {target_hash}, error: {error_msg}")
                    raise ValueError(f"Cannot checkout memory hash: {target_hash}. Error: {error_msg}")
            else:
                error_msg = hash_check.stderr
                logging.error(f"Memory: Target hash does not exist: {target_hash}, error: {error_msg}")
                raise ValueError(f"Memory hash does not exist: {target_hash}. Error: {error_msg}")
        else:
            logging.debug(f"Memory: Already at target hash: {target_hash}")
        
        # Write content
        with open(doc_path, "w") as f:
            f.write(content)
        
        # Add to git
        subprocess.run(["git", "add", str(doc_path)], cwd=repo_path, check=True)
        
        # Commit
        message = metadata.get("commit_message", f"Add {category} document: {filename}")
        commit_result = subprocess.run(["git", "commit", "-m", message], cwd=repo_path, capture_output=True, text=True, check=True)
        
        # Log the git commit output at debug level
        if hasattr(logging, 'debug') and commit_result.stdout:
            logging.debug(commit_result.stdout.strip())
        
        # Get commit hash
        commit_hash = get_current_hash(repo_path)
        
        # Update cross-reference if code hash is provided
        if "code_hash" in metadata:
            update_cross_reference(metadata["code_hash"], commit_hash, repo_path)
        
        # Log storage information at debug level instead of printing
        if hasattr(logging, 'debug'):
            logging.debug(f"Stored document at: {doc_path}")
            logging.debug(f"Commit hash: {commit_hash}")
        
        relative_path = str(doc_path.relative_to(repo_path))
        
        # Return comprehensive information about the stored document
        return {
            "path": relative_path,
            "memory_hash": commit_hash,
            "category": category,
            "filename": filename
        }
    finally:
        # Restore original branch if we changed it
        if original_branch and current_hash != target_hash:
            try:
                logging.debug(f"Memory: Restoring original position: {original_branch}")
                subprocess.run(
                    ["git", "checkout", original_branch],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                logging.warning(f"Memory: Failed to restore original branch: {str(e)}")

def retrieve_documents(category=None, limit=10, repo_path=None, memory_hash=None):
    """
    Retrieve documents from the memory repository.
    
    Args:
        category: Category to retrieve documents from
        limit: Maximum number of documents to retrieve
        repo_path: Path to the memory repository
        memory_hash: Required hash to operate on - must check out this hash before reading documents
        
    Returns:
        List of (path, content) tuples for documents
    """
    # Get repository path
    repo_path = repo_path or get_repo_path()
    repo_path = Path(repo_path)
    
    # Validate memory_hash parameter
    if not memory_hash:
        raise ValueError("Memory hash is required")
    
    # Keep track of original position to restore later
    original_branch = None
    
    try:
        # Save the current branch or commit we're on
        try:
            result = subprocess.run(
                ["git", "symbolic-ref", "--short", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            # If symbolic-ref succeeds, we're on a branch
            if result.returncode == 0:
                original_branch = result.stdout.strip()
            else:
                # If symbolic-ref fails, we're in detached HEAD state
                original_branch = get_current_hash(repo_path)
            
            logging.debug(f"Memory: Saved original position: {original_branch}")
        except Exception as e:
            logging.warning(f"Memory: Failed to get current branch: {str(e)}")
            original_branch = None
        
        # Check if already at memory_hash
        current_hash = get_current_hash(repo_path)
        if current_hash != memory_hash:
            # Check if the target hash exists
            hash_check = subprocess.run(
                ["git", "cat-file", "-t", memory_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            
            if hash_check.returncode == 0:
                logging.debug(f"Memory: Checking out memory hash: {memory_hash}")
                # Checkout the memory hash
                checkout_result = subprocess.run(
                    ["git", "checkout", memory_hash],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                
                if checkout_result.returncode != 0:
                    error_msg = checkout_result.stderr
                    logging.error(f"Memory: Failed to checkout memory hash: {memory_hash}, error: {error_msg}")
                    raise ValueError(f"Cannot checkout memory hash: {memory_hash}. Error: {error_msg}")
            else:
                error_msg = hash_check.stderr
                logging.error(f"Memory: Memory hash does not exist: {memory_hash}, error: {error_msg}")
                raise ValueError(f"Memory hash does not exist: {memory_hash}. Error: {error_msg}")
        else:
            logging.debug(f"Memory: Already at memory hash: {memory_hash}")
        
        # Set search path
        if category:
            search_path = repo_path / "documents" / category
        else:
            search_path = repo_path / "documents"
        
        if not search_path.exists():
            if hasattr(logging, 'debug'):
                logging.debug(f"Path does not exist: {search_path}")
            return []
        
        # Find all .md files
        files = list(search_path.glob("**/*.md"))
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Limit results
        files = files[:limit]
        
        # Read contents
        results = []
        for file in files:
            with open(file, "r") as f:
                results.append((str(file.relative_to(repo_path)), f.read()))
        
        return results
    finally:
        # Restore original branch if we changed it
        if original_branch and current_hash != memory_hash:
            try:
                logging.debug(f"Memory: Restoring original position: {original_branch}")
                subprocess.run(
                    ["git", "checkout", original_branch],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                logging.warning(f"Memory: Failed to restore original branch: {str(e)}")

def update_cross_reference(code_hash, memory_hash, repo_path=None, base_memory_hash=None):
    """
    Update the cross-reference between code and memory repositories.
    
    Args:
        code_hash: The code repository commit hash
        memory_hash: The memory repository commit hash to reference
        repo_path: Path to the memory repository
        base_memory_hash: Required hash to operate on - must check out this hash before updating cross-reference
    """
    # Get repository path
    repo_path = repo_path or get_repo_path()
    repo_path = Path(repo_path)
    
    # Validate base_memory_hash parameter
    if not base_memory_hash:
        base_memory_hash = memory_hash  # Default to the memory hash we're referencing
    
    # Keep track of original position to restore later
    original_branch = None
    
    try:
        # Save the current branch or commit we're on
        try:
            result = subprocess.run(
                ["git", "symbolic-ref", "--short", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            # If symbolic-ref succeeds, we're on a branch
            if result.returncode == 0:
                original_branch = result.stdout.strip()
            else:
                # If symbolic-ref fails, we're in detached HEAD state
                original_branch = get_current_hash(repo_path)
            
            logging.debug(f"Memory: Saved original position: {original_branch}")
        except Exception as e:
            logging.warning(f"Memory: Failed to get current branch: {str(e)}")
            original_branch = None
        
        # Check if already at base_memory_hash
        current_hash = get_current_hash(repo_path)
        if current_hash != base_memory_hash:
            # Check if the base hash exists
            hash_check = subprocess.run(
                ["git", "cat-file", "-t", base_memory_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            
            if hash_check.returncode == 0:
                logging.debug(f"Memory: Checking out base memory hash: {base_memory_hash}")
                # Checkout the base memory hash
                checkout_result = subprocess.run(
                    ["git", "checkout", base_memory_hash],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                
                if checkout_result.returncode != 0:
                    error_msg = checkout_result.stderr
                    logging.error(f"Memory: Failed to checkout base memory hash: {base_memory_hash}, error: {error_msg}")
                    raise ValueError(f"Cannot checkout base memory hash: {base_memory_hash}. Error: {error_msg}")
            else:
                error_msg = hash_check.stderr
                logging.error(f"Memory: Base memory hash does not exist: {base_memory_hash}, error: {error_msg}")
                raise ValueError(f"Base memory hash does not exist: {base_memory_hash}. Error: {error_msg}")
        else:
            logging.debug(f"Memory: Already at base memory hash: {base_memory_hash}")
        
        cross_ref_path = repo_path / "metadata" / "cross-reference.json"
        
        # Ensure directory exists
        cross_ref_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing cross-reference
        if cross_ref_path.exists():
            with open(cross_ref_path, "r") as f:
                cross_ref = json.load(f)
        else:
            cross_ref = {
                "mappings": [],
                "latest": {}
            }
        
        # Ensure the structure has both mappings and latest
        if "mappings" not in cross_ref:
            cross_ref["mappings"] = []
        if "latest" not in cross_ref:
            cross_ref["latest"] = {}
        
        # Add to mappings history with timestamp
        mapping = {
            "code_hash": code_hash,
            "memory_hash": memory_hash,
            "timestamp": int(time.time())
        }
        cross_ref["mappings"].append(mapping)
        
        # Update the latest mapping
        cross_ref["latest"][code_hash] = memory_hash
        
        # Save updated cross-reference
        with open(cross_ref_path, "w") as f:
            json.dump(cross_ref, f, indent=2)
        
        # We don't add cross-reference.json to git or commit it
        # since it's in .gitignore and is managed separately
        
        # Log cross-reference update at debug level instead of printing
        if hasattr(logging, 'debug'):
            logging.debug(f"Updated cross-reference: {code_hash[:7]} -> {memory_hash[:7]}")
        else:
            print(f"Updated cross-reference: {code_hash[:7]} -> {memory_hash[:7]}")
    finally:
        # Restore original branch if we changed it
        if original_branch and current_hash != base_memory_hash:
            try:
                logging.debug(f"Memory: Restoring original position: {original_branch}")
                subprocess.run(
                    ["git", "checkout", original_branch],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                logging.warning(f"Memory: Failed to restore original branch: {str(e)}")

def get_memory_for_code_hash(code_hash, repo_path=None, historical=False, timestamp=None, memory_hash=None):
    """
    Get the memory hash corresponding to a code hash.
    
    Args:
        code_hash: The code repository commit hash
        repo_path: Path to the memory repository
        historical: If True, returns all historical mappings for this code hash
        timestamp: If provided, returns the mapping closest to this timestamp
        memory_hash: Required hash to operate on - must check out this hash before reading cross-reference
        
    Returns:
        If historical=True: List of mappings for this code hash
        If timestamp provided: The mapping closest to the timestamp
        Otherwise: The latest memory hash for this code hash
    """
    # Get repository path
    repo_path = repo_path or get_repo_path()
    repo_path = Path(repo_path)
    
    # Validate memory_hash parameter
    if not memory_hash:
        raise ValueError("Memory hash is required")
    
    # Keep track of original position to restore later
    original_branch = None
    
    try:
        # Save the current branch or commit we're on
        try:
            result = subprocess.run(
                ["git", "symbolic-ref", "--short", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            # If symbolic-ref succeeds, we're on a branch
            if result.returncode == 0:
                original_branch = result.stdout.strip()
            else:
                # If symbolic-ref fails, we're in detached HEAD state
                original_branch = get_current_hash(repo_path)
            
            logging.debug(f"Memory: Saved original position: {original_branch}")
        except Exception as e:
            logging.warning(f"Memory: Failed to get current branch: {str(e)}")
            original_branch = None
        
        # Check if already at memory_hash
        current_hash = get_current_hash(repo_path)
        if current_hash != memory_hash:
            # Check if the memory hash exists
            hash_check = subprocess.run(
                ["git", "cat-file", "-t", memory_hash],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            
            if hash_check.returncode == 0:
                logging.debug(f"Memory: Checking out memory hash: {memory_hash}")
                # Checkout the memory hash
                checkout_result = subprocess.run(
                    ["git", "checkout", memory_hash],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                
                if checkout_result.returncode != 0:
                    error_msg = checkout_result.stderr
                    logging.error(f"Memory: Failed to checkout memory hash: {memory_hash}, error: {error_msg}")
                    raise ValueError(f"Cannot checkout memory hash: {memory_hash}. Error: {error_msg}")
            else:
                error_msg = hash_check.stderr
                logging.error(f"Memory: Memory hash does not exist: {memory_hash}, error: {error_msg}")
                raise ValueError(f"Memory hash does not exist: {memory_hash}. Error: {error_msg}")
        else:
            logging.debug(f"Memory: Already at memory hash: {memory_hash}")
        
        cross_ref_path = repo_path / "metadata" / "cross-reference.json"
        
        if not cross_ref_path.exists():
            if hasattr(logging, 'debug'):
                logging.debug(f"Cross-reference file not found: {cross_ref_path}")
            else:
                print(f"Cross-reference file not found: {cross_ref_path}")
            return None
        
        # Load cross-reference
        with open(cross_ref_path, "r") as f:
            cross_ref = json.load(f)
        
        # Handle old format
        if not isinstance(cross_ref, dict) or ("mappings" not in cross_ref and "latest" not in cross_ref):
            # Old format - convert
            old_data = cross_ref
            cross_ref = {
                "mappings": [
                    {"code_hash": k, "memory_hash": v, "timestamp": int(time.time())}
                    for k, v in old_data.items()
                ],
                "latest": old_data
            }
        
        # Return all historical mappings if requested
        if historical:
            historical_mappings = [
                mapping for mapping in cross_ref.get("mappings", [])
                if mapping["code_hash"] == code_hash
            ]
            if historical_mappings:
                if hasattr(logging, 'debug'):
                    logging.debug(f"Found {len(historical_mappings)} historical mappings for code hash {code_hash[:7]}")
                else:
                    print(f"Found {len(historical_mappings)} historical mappings for code hash {code_hash[:7]}")
            else:
                if hasattr(logging, 'debug'):
                    logging.debug(f"No historical mappings found for code hash: {code_hash[:7]}")
                else:
                    print(f"No historical mappings found for code hash: {code_hash[:7]}")
            return historical_mappings
        
        # Return mapping closest to timestamp if provided
        if timestamp:
            mappings = [
                mapping for mapping in cross_ref.get("mappings", [])
                if mapping["code_hash"] == code_hash
            ]
            if mappings:
                # Find closest mapping to timestamp
                closest_mapping = min(mappings, key=lambda m: abs(m["timestamp"] - timestamp))
                if hasattr(logging, 'debug'):
                    logging.debug(f"Found memory hash from {closest_mapping['timestamp']} for code hash {code_hash[:7]}: {closest_mapping['memory_hash'][:7]}")
                else:
                    print(f"Found memory hash from {closest_mapping['timestamp']} for code hash {code_hash[:7]}: {closest_mapping['memory_hash'][:7]}")
                return closest_mapping["memory_hash"]
        
        # Otherwise return latest
        memory_hash = cross_ref.get("latest", {}).get(code_hash)
        
        if memory_hash:
            if hasattr(logging, 'debug'):
                logging.debug(f"Found latest memory hash for code hash {code_hash[:7]}: {memory_hash[:7]}")
            else:
                print(f"Found latest memory hash for code hash {code_hash[:7]}: {memory_hash[:7]}")
        else:
            if hasattr(logging, 'debug'):
                logging.debug(f"No memory hash found for code hash: {code_hash[:7]}")
            else:
                print(f"No memory hash found for code hash: {code_hash[:7]}")
        
        return memory_hash
    finally:
        # Restore original branch if we changed it
        if original_branch and current_hash != memory_hash:
            try:
                logging.debug(f"Memory: Restoring original position: {original_branch}")
                subprocess.run(
                    ["git", "checkout", original_branch],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                logging.warning(f"Memory: Failed to restore original branch: {str(e)}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Memory repository tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Store document command
    store_parser = subparsers.add_parser("store", help="Store a document")
    store_parser.add_argument("category", help="Document category")
    store_parser.add_argument("--file", help="File to read content from")
    store_parser.add_argument("--content", help="Content to store")
    store_parser.add_argument("--id", help="Document ID")
    store_parser.add_argument("--code-hash", help="Code hash to link to")
    store_parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    # Retrieve documents command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve documents")
    retrieve_parser.add_argument("--category", help="Document category")
    retrieve_parser.add_argument("--limit", type=int, default=5, help="Maximum number of documents")
    retrieve_parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    # Link command
    link_parser = subparsers.add_parser("link", help="Link code hash to memory hash")
    link_parser.add_argument("code_hash", help="Code repository hash")
    link_parser.add_argument("memory_hash", help="Memory repository hash")
    link_parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    # Lookup command
    lookup_parser = subparsers.add_parser("lookup", help="Look up memory hash for code hash")
    lookup_parser.add_argument("code_hash", help="Code repository hash")
    lookup_parser.add_argument("--historical", action="store_true", help="Return all historical mappings")
    lookup_parser.add_argument("--timestamp", type=int, help="Find mapping closest to this timestamp")
    lookup_parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    # New history command
    history_parser = subparsers.add_parser("history", help="View full history of code-memory mappings")
    history_parser.add_argument("--limit", type=int, default=10, help="Maximum number of entries")
    history_parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    # Add general arguments
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    args = parser.parse_args()
    
    # Configure basic logging
    debug_mode = getattr(args, "debug", False)
    logging_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format='%(message)s'
    )
    
    # Hide DEBUG messages by default when run as a command-line tool
    if not debug_mode:
        logging.getLogger().setLevel(logging.INFO)
    
    if args.command == "store":
        if args.file:
            with open(args.file, "r") as f:
                content = f.read()
        elif args.content:
            content = args.content
        else:
            content = sys.stdin.read()
        
        metadata = {}
        if args.id:
            metadata["id"] = args.id
        if args.code_hash:
            metadata["code_hash"] = args.code_hash
        
        result = store_document(content, args.category, metadata)
        print(f"Document stored: {result['path']}")
    
    elif args.command == "retrieve":
        documents = retrieve_documents(args.category, args.limit)
        
        if not documents:
            print("No documents found")
            return
        
        for i, (path, content) in enumerate(documents):
            print(f"\n--- Document {i+1}: {path} ---")
            print(content[:500])
            if len(content) > 500:
                print("... (truncated)")
    
    elif args.command == "link":
        update_cross_reference(args.code_hash, args.memory_hash)
        print(f"Linked code hash {args.code_hash[:7]} to memory hash {args.memory_hash[:7]}")
    
    elif args.command == "lookup":
        if args.historical:
            mappings = get_memory_for_code_hash(args.code_hash, historical=True)
            if mappings:
                print("\nHistorical mappings for code hash:", args.code_hash)
                for i, mapping in enumerate(mappings):
                    print(f"{i+1}. {mapping['timestamp']} (Unix timestamp): {mapping['memory_hash']}")
        elif args.timestamp:
            memory_hash = get_memory_for_code_hash(args.code_hash, timestamp=args.timestamp)
            if memory_hash:
                print(f"Memory hash: {memory_hash}")
        else:
            memory_hash = get_memory_for_code_hash(args.code_hash)
            if memory_hash:
                print(f"Memory hash: {memory_hash}")
                
    elif args.command == "history":
        # Get repository path
        repo_path = get_repo_path()
        cross_ref_path = Path(repo_path) / "metadata" / "cross-reference.json"
        
        if not cross_ref_path.exists():
            print(f"Cross-reference file not found: {cross_ref_path}")
            return
        
        # Load cross-reference
        with open(cross_ref_path, "r") as f:
            cross_ref = json.load(f)
        
        if "mappings" not in cross_ref:
            print("No history found in cross-reference file")
            return
        
        # Sort by timestamp (newest first)
        mappings = sorted(cross_ref["mappings"], key=lambda m: m["timestamp"], reverse=True)
        
        # Limit results
        mappings = mappings[:args.limit]
        
        print("\nCross-reference history:")
        for i, mapping in enumerate(mappings):
            timestamp = mapping["timestamp"]
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            print(f"{i+1}. {time_str}: Code {mapping['code_hash'][:7]} -> Memory {mapping['memory_hash'][:7]}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 