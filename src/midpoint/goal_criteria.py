"""
Helper functions for managing validation criteria for goals.
"""

import json
import logging
from typing import List, Optional
from openai import OpenAI

from midpoint.agents.config import get_openai_api_key, get_agent_config

logger = logging.getLogger(__name__)


def generate_validation_criteria(goal_description: str) -> List[str]:
    """
    Generate validation criteria for a goal using LLM.
    
    Args:
        goal_description: The description of the goal
        
    Returns:
        List of validation criteria strings
    """
    try:
        # Get API key and config
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        agent_config = get_agent_config("goal_analyzer")  # Use goal_analyzer config as default
        model = agent_config.get("model", "gpt-4o-mini")
        max_tokens = agent_config.get("max_tokens", 2000)
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Create system prompt
        system_prompt = """You are an expert at creating clear, measurable validation criteria for software development goals.

Your task is to analyze a goal description and generate a list of validation criteria that can be used to verify the goal has been successfully completed.

Guidelines:
- Criteria should be specific and measurable
- Criteria should be testable/verifiable
- Focus on outcomes, not implementation details
- Include 3-5 criteria (more for complex goals, fewer for simple ones)
- Each criterion should be a single, clear statement

Return ONLY a JSON array of strings, like: ["criterion 1", "criterion 2", "criterion 3"]
Do not include any explanatory text before or after the JSON."""
        
        # Create user prompt
        user_prompt = f"""Goal Description: {goal_description}

Generate validation criteria for this goal. Return a JSON array of criteria strings."""
        
        # Call LLM
        # Use max_completion_tokens (standardized parameter for newer model families)
        # Note: Some models don't support custom temperature, so we omit it
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=max_tokens
        )
        
        # Extract response
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON (might be wrapped in markdown)
        try:
            # Try direct JSON parse
            criteria = json.loads(content)
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
            if json_match:
                criteria = json.loads(json_match.group(1))
            else:
                # Try to find array in the text
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    criteria = json.loads(json_match.group(0))
                else:
                    raise ValueError(f"Could not parse JSON from LLM response: {content[:200]}")
        
        # Validate it's a list of strings
        if not isinstance(criteria, list):
            raise ValueError(f"Expected list, got {type(criteria)}")
        
        criteria = [str(c) for c in criteria if c]  # Convert to strings and filter empty
        
        if not criteria:
            raise ValueError("LLM returned empty criteria list")
        
        logger.info(f"Generated {len(criteria)} validation criteria for goal")
        return criteria
        
    except Exception as e:
        logger.error(f"Failed to generate validation criteria: {e}")
        raise


def prompt_for_validation_criteria(goal_description: str) -> List[str]:
    """
    Prompt user for validation criteria with option to auto-generate.
    
    Args:
        goal_description: The description of the goal
        
    Returns:
        List of validation criteria strings
    """
    print(f"\nGoal: {goal_description}")
    print("\nValidation Criteria:")
    print("Enter validation criteria (one per line, empty line to finish).")
    print("Or type 'auto' to have AI generate criteria, or 'skip' to leave empty.")
    print("-" * 60)
    
    criteria = []
    while True:
        try:
            line = input("Criterion (or 'auto'/'skip'): ").strip()
            
            if not line:
                # Empty line - finish input
                break
            elif line.lower() == 'auto':
                # Generate automatically
                print("\nGenerating validation criteria using AI...")
                try:
                    criteria = generate_validation_criteria(goal_description)
                    print(f"\nGenerated {len(criteria)} criteria:")
                    for i, criterion in enumerate(criteria, 1):
                        print(f"  {i}. {criterion}")
                    print()
                    return criteria
                except Exception as e:
                    print(f"Error generating criteria: {e}")
                    print("Please enter criteria manually or type 'skip' to continue without criteria.")
                    continue
            elif line.lower() == 'skip':
                # Skip criteria
                return []
            else:
                # Add criterion
                criteria.append(line)
        except (EOFError, KeyboardInterrupt):
            # User interrupted
            print("\n")
            if criteria:
                return criteria
            return []
    
    return criteria

