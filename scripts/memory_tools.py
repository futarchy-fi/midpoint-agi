#!/usr/bin/env python
"""
Basic tools for working with the memory repository.

This script provides simple functions for storing and retrieving documents.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def get_repo_path():
    """Get the memory repository path from environment variables."""
    load_dotenv()
    return os.getenv("MEMORY_REPO_PATH", "./agent-memory")

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

def store_document(content, category, metadata=None, repo_path=None):
    """Store a document in the memory repository."""
    # Get repository path
    repo_path = repo_path or get_repo_path()
    repo_path = Path(repo_path)
    
    # Default metadata
    metadata = metadata or {}
    
    # Create a filename based on metadata
    filename = f"{metadata.get('id', 'doc')}_{int(time.time())}.md"
    doc_path = repo_path / "documents" / category / filename
    
    # Ensure directory exists
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write content
    with open(doc_path, "w") as f:
        f.write(content)
    
    # Add to git
    subprocess.run(["git", "add", str(doc_path)], cwd=repo_path, check=True)
    
    # Commit
    message = metadata.get("commit_message", f"Add {category} document: {filename}")
    subprocess.run(["git", "commit", "-m", message], cwd=repo_path, check=True)
    
    # Get commit hash
    commit_hash = get_current_hash(repo_path)
    
    # Update cross-reference if code hash is provided
    if "code_hash" in metadata:
        update_cross_reference(metadata["code_hash"], commit_hash, repo_path)
    
    print(f"Stored document at: {doc_path}")
    print(f"Commit hash: {commit_hash}")
    
    return str(doc_path.relative_to(repo_path))

def retrieve_documents(category=None, limit=10, repo_path=None):
    """Retrieve documents from the memory repository."""
    # Get repository path
    repo_path = repo_path or get_repo_path()
    repo_path = Path(repo_path)
    
    # Set search path
    if category:
        search_path = repo_path / "documents" / category
    else:
        search_path = repo_path / "documents"
    
    if not search_path.exists():
        print(f"Path does not exist: {search_path}")
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

def update_cross_reference(code_hash, memory_hash, repo_path=None):
    """Update the cross-reference between code and memory repositories."""
    # Get repository path
    repo_path = repo_path or get_repo_path()
    repo_path = Path(repo_path)
    
    cross_ref_path = repo_path / "metadata" / "cross-reference.json"
    
    # Ensure directory exists
    cross_ref_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing cross-reference
    if cross_ref_path.exists():
        with open(cross_ref_path, "r") as f:
            cross_ref = json.load(f)
    else:
        cross_ref = {}
    
    # Update with new reference
    cross_ref[code_hash] = memory_hash
    
    # Save updated cross-reference
    with open(cross_ref_path, "w") as f:
        json.dump(cross_ref, f, indent=2)
    
    # Add to git
    subprocess.run(["git", "add", str(cross_ref_path)], cwd=repo_path, check=True)
    
    # Commit
    message = f"Update cross-reference for code hash: {code_hash[:7]}"
    subprocess.run(["git", "commit", "-m", message], cwd=repo_path, check=True)
    
    print(f"Updated cross-reference: {code_hash[:7]} -> {memory_hash[:7]}")

def get_memory_for_code_hash(code_hash, repo_path=None):
    """Get the memory hash corresponding to a code hash."""
    # Get repository path
    repo_path = repo_path or get_repo_path()
    repo_path = Path(repo_path)
    
    cross_ref_path = repo_path / "metadata" / "cross-reference.json"
    
    if not cross_ref_path.exists():
        print(f"Cross-reference file not found: {cross_ref_path}")
        return None
    
    # Load cross-reference
    with open(cross_ref_path, "r") as f:
        cross_ref = json.load(f)
    
    memory_hash = cross_ref.get(code_hash)
    
    if memory_hash:
        print(f"Found memory hash for code hash {code_hash[:7]}: {memory_hash[:7]}")
    else:
        print(f"No memory hash found for code hash: {code_hash[:7]}")
    
    return memory_hash

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
    
    # Retrieve documents command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve documents")
    retrieve_parser.add_argument("--category", help="Document category")
    retrieve_parser.add_argument("--limit", type=int, default=5, help="Maximum number of documents")
    
    # Link command
    link_parser = subparsers.add_parser("link", help="Link code hash to memory hash")
    link_parser.add_argument("code_hash", help="Code repository hash")
    link_parser.add_argument("memory_hash", help="Memory repository hash")
    
    # Lookup command
    lookup_parser = subparsers.add_parser("lookup", help="Look up memory hash for code hash")
    lookup_parser.add_argument("code_hash", help="Code repository hash")
    
    args = parser.parse_args()
    
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
        
        store_document(content, args.category, metadata)
    
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
    
    elif args.command == "lookup":
        get_memory_for_code_hash(args.code_hash)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 