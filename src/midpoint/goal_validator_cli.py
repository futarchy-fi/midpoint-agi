"""
Command-line interface for goal validation.

This module provides a standalone command-line interface for validating goals
using the GoalValidator agent.
"""

import argparse
import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime

# Import directly from the source modules
from midpoint.agents.tools.git_tools import get_current_hash, get_current_branch
from midpoint.agents.goal_validator import GoalValidator
from midpoint.agents.models import Goal, ExecutionResult, ValidationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger('goal_validator_cli')


def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path(".goal")
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: .goal")
    return goal_path


async def get_repository_state(goal_id: str, debug: bool = False) -> Dict[str, Any]:
    """
    Gather repository state information for validation.
    
    Args:
        goal_id: ID of the goal being validated
        debug: Whether to show debug output
        
    Returns:
        dict: Repository state information
    """
    # Use current directory as repo path
    repo_path = os.getcwd()
    
    repo_state = {
        "current_hash": await get_current_hash(repo_path),
        "current_branch": await get_current_branch(repo_path),
        "timestamp": datetime.datetime.now().isoformat(),
        "goal_id": goal_id
    }
    
    # Get goal data if it exists
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if goal_file.exists():
        try:
            with open(goal_file, 'r') as f:
                goal_data = json.load(f)
                
            # Add initial hash if available
            if "initial_hash" in goal_data:
                repo_state["initial_hash"] = goal_data["initial_hash"]
        except Exception as e:
            if debug:
                logger.debug(f"Error loading goal data: {e}")
    
    return repo_state


async def validate_goal_cli(goal_id: str, 
                         repository_path: Optional[str] = None, 
                         model: str = "gpt-4o-mini",
                         debug: bool = False, 
                         quiet: bool = False,
                         output_json: bool = False) -> bool:
    """
    Validate a goal using the GoalValidator agent.
    
    Args:
        goal_id: The ID of the goal to validate
        repository_path: Optional repository path (defaults to current directory)
        model: Model to use for validation
        debug: Whether to show debug output
        quiet: Whether to only show warnings and result
        output_json: Whether to output the result as JSON
        
    Returns:
        bool: True if validation was successful, False otherwise
    """
    try:
        # Use the GoalValidator to validate the goal
        validator = GoalValidator(model=model)
        
        # Set logging level based on debug/quiet flags
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        elif quiet:
            logging.getLogger().setLevel(logging.WARNING)
        
        if not quiet:
            print(f"Validating goal {goal_id} using model {model}...")
        
        # Validate the goal
        validation_result = await validator.validate_goal(
            goal_id=goal_id,
            repository_path=repository_path,
            auto=True
        )
        
        # Get repository state
        repo_state = await get_repository_state(goal_id, debug)
        
        # Create validation record
        from midpoint.validation import create_validation_record, save_validation_record
        
        criteria_results = []
        for result in validation_result.criteria_results:
            criteria_results.append({
                "criterion": result.get("criterion", ""),
                "passed": result.get("passed", False),
                "reasoning": result.get("reasoning", ""),
                "evidence": result.get("evidence", [])
            })
        
        validation_record = create_validation_record(
            goal_id=goal_id,
            criteria_results=criteria_results,
            repo_state=repo_state,
            auto=True
        )
        
        # Save validation record
        validation_file = await save_validation_record(goal_id, validation_record)
        
        # Show results
        if output_json:
            # Output the full validation record as JSON
            print(json.dumps(validation_record, indent=2))
        else:
            # Show human-readable results
            print("\nValidation Results:")
            print(f"Overall Score: {validation_result.score:.2%}")
            print(f"Passed: {sum(1 for r in criteria_results if r.get('passed', False))}/{len(criteria_results)} criteria")
            print(f"Success Threshold: {0.8:.2%}")
            
            # Show validation reasoning
            print("\nReasoning:")
            print(validation_result.reasoning)
            
            # Show individual criteria results
            print("\nCriteria Results:")
            for i, result in enumerate(criteria_results, 1):
                print(f"\n{i}. {result['criterion']}")
                print(f"   {'✅ Passed' if result['passed'] else '❌ Failed'}")
                print(f"   Reasoning: {result['reasoning']}")
                if result.get('evidence'):
                    print(f"   Evidence: {'; '.join(result['evidence'])}")
            
            print(f"\nValidation record saved to: {validation_file}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating goal {goal_id}: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main entry point for the validator CLI."""
    parser = argparse.ArgumentParser(description="Validate a goal using LLM")
    parser.add_argument("goal_id", help="ID of the goal to validate")
    parser.add_argument("--repo", "-r", dest="repository_path", help="Path to the repository (default: current directory)")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="Model to use for validation (default: gpt-4o-mini)")
    parser.add_argument("--debug", "-d", action="store_true", help="Show debug output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show warnings and result")
    parser.add_argument("--json", "-j", dest="output_json", action="store_true", help="Output result as JSON")
    
    args = parser.parse_args()
    
    try:
        success = asyncio.run(validate_goal_cli(
            goal_id=args.goal_id,
            repository_path=args.repository_path,
            model=args.model,
            debug=args.debug,
            quiet=args.quiet,
            output_json=args.output_json
        ))
        
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nValidation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 