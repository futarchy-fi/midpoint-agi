"""
CLI client for testing git operations.
"""

import asyncio
import argparse
from pathlib import Path
from agents.tools import (
    check_repo_state,
    create_branch,
    revert_to_hash,
    create_commit,
    get_current_hash,
    get_current_branch,
    checkout_branch
)
from agents.repo_context import RepoConfig

async def print_repo_state(repo_path: str):
    """Print current repository state."""
    state = await check_repo_state(repo_path)
    print("\nRepository State:")
    print(f"Clean: {state['is_clean']}")
    print(f"Uncommitted changes: {state['has_uncommitted']}")
    print(f"Untracked files: {state['has_untracked']}")
    print(f"Merge conflicts: {state['has_merge_conflicts']}")
    print(f"Rebase conflicts: {state['has_rebase_conflicts']}")

async def main():
    parser = argparse.ArgumentParser(description="Test git operations on a repository")
    parser.add_argument("repo_path", help="Path to the git repository")
    parser.add_argument("--branch", help="Base name for new branch")
    parser.add_argument("--hash", help="Git hash to revert to")
    parser.add_argument("--commit", help="Commit message")
    parser.add_argument("--checkout", help="Branch to checkout")
    parser.add_argument("--state", action="store_true", help="Check repository state")
    
    args = parser.parse_args()
    
    # Validate repo path
    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        return
        
    try:
        if args.state:
            await print_repo_state(str(repo_path))
            
        if args.branch:
            branch_name = await create_branch(str(repo_path), args.branch)
            print(f"\nCreated new branch: {branch_name}")
            
        if args.hash:
            await revert_to_hash(str(repo_path), args.hash)
            print(f"\nReverted to hash: {args.hash}")
            
        if args.commit:
            commit_hash = await create_commit(str(repo_path), args.commit)
            print(f"\nCreated commit: {commit_hash}")
            
        if args.checkout:
            await checkout_branch(str(repo_path), args.checkout)
            print(f"\nChecked out branch: {args.checkout}")
            
        # Always show current state
        current_branch = await get_current_branch(str(repo_path))
        current_hash = await get_current_hash(str(repo_path))
        print(f"\nCurrent branch: {current_branch}")
        print(f"Current hash: {current_hash}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 