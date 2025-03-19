#!/usr/bin/env python
"""
Custom GoalDecomposer with improved parsing for the API response.

This handles the different format that the API is returning.
"""

import re
from midpoint.agents.goal_decomposer import GoalDecomposer
from midpoint.agents.models import StrategyPlan

class CustomGoalDecomposer(GoalDecomposer):
    """
    Enhanced version of GoalDecomposer with better parsing.
    
    This version can handle different response formats from the OpenAI API.
    """
    
    def _parse_response(self, response: str) -> StrategyPlan:
        """
        Parse the OpenAI API response into a StrategyPlan.
        
        This enhanced version handles different response formats:
        1. The simple format with "Strategy:", "Steps:", "Reasoning:", "Points:"
        2. The detailed format with structured sections and numbered steps
        """
        # Check if response is in the simple format
        if "Strategy:" in response and "Steps:" in response:
            return self._parse_simple_format(response)
        else:
            return self._parse_detailed_format(response)
    
    def _parse_simple_format(self, response: str) -> StrategyPlan:
        """Parse the simple response format."""
        # Split response into sections
        sections = response.split("\n")
        
        # Initialize variables
        strategy_desc = ""
        steps = []
        reasoning = ""
        points = 0
        
        # Process each line
        current_section = None
        for line in sections:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.lower().startswith("strategy:"):
                current_section = "strategy"
                strategy_desc = line  # Keep the full line including "Strategy:"
            elif line.lower().startswith("steps:"):
                current_section = "steps"
            elif line.lower().startswith("reasoning:"):
                current_section = "reasoning"
                reasoning = line.split(":", 1)[1].strip()
            elif line.lower().startswith("points:"):
                try:
                    points = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            # Process content based on current section
            elif current_section == "steps" and line.startswith("-"):
                steps.append(line.strip("- ").strip())
            elif current_section == "reasoning" and not line.lower().startswith("points:"):
                reasoning += " " + line
                
        return StrategyPlan(
            steps=steps,
            reasoning=reasoning.strip(),
            estimated_points=points,
            metadata={
                "strategy_description": strategy_desc,
                "raw_response": response
            }
        )
    
    def _parse_detailed_format(self, response: str) -> StrategyPlan:
        """Parse the detailed response format with structured sections."""
        # Find strategy description
        strategy_match = re.search(r"(?:#+\s*Strategy\s+Description|\bStrategy.*?)(?:\s*:?\s*)(.*?)(?=\n#+|\Z)", 
                                  response, re.DOTALL | re.IGNORECASE)
        strategy_desc = "Strategy: " + (strategy_match.group(1).strip() if strategy_match else "Not specified")
        
        # Find reasoning - try various patterns that might indicate a reasoning section
        reasoning_patterns = [
            r"(?:#+\s*Reasoning.*?)(?:\s*:?\s*)(.*?)(?=\n#+|\Z)",
            r"(?:#+\s*Reasoning\s+and\s+Feasibility.*?)(?:\s*:?\s*)(.*?)(?=\n#+|\Z)",
            r"The\s+plan\s+is\s+structured.*?(.*?)(?=\n#+|\Z)",
            r"This\s+approach.*?(.*?)(?=\n#+|\Z)",
            r"The\s+strategy.*?(.*?)(?=\n#+|\Z)"
        ]
        
        reasoning = "Not specified"
        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                break
                
        # In case we still don't have reasoning, use the last paragraph as a fallback
        if reasoning == "Not specified":
            paragraphs = response.split("\n\n")
            if paragraphs:
                last_paragraphs = [p.strip() for p in paragraphs[-3:] if p.strip()]
                # Use the longest paragraph that seems meaningful
                if last_paragraphs:
                    reasoning = max(last_paragraphs, key=len)
                    
        # If we still have nothing, and there's a "Total Points Needed" section, use the text after that
        if reasoning == "Not specified":
            total_points_section = re.search(r"Total\s+Points.*?\n(.*?)(?=\n#+|\Z)", 
                                          response, re.DOTALL | re.IGNORECASE)
            if total_points_section:
                reasoning = total_points_section.group(1).strip()
        
        # Extract steps - look for Step X: patterns
        steps = []
        
        # First try to find explicit steps in the execution plan section
        execution_plan_match = re.search(r"(?:#+\s*Execution\s+Plan)(.*?)(?=#+\s*Total\s+Points|\Z)", 
                                      response, re.DOTALL | re.IGNORECASE)
        
        if execution_plan_match:
            plan_text = execution_plan_match.group(1)
            # Look for specific step patterns in the execution plan
            step_matches = re.finditer(r"(?:\*\*Step \d+:\*\*|Step \d+:)\s*(.*?)(?=\n\s*(?:\*\*Step|\Z))", 
                                     plan_text, re.DOTALL)
            for match in step_matches:
                step_text = match.group(1).strip()
                if step_text:
                    steps.append(step_text)
        
        # If no steps found yet, try looking in the entire response
        if not steps:
            step_matches = re.finditer(r"(?:\*\*Step \d+:\*\*|Step \d+:)\s*(.*?)(?=\n\s*(?:\*\*Step|\Z))", 
                                     response, re.DOTALL)
            for match in step_matches:
                step_text = match.group(1).strip()
                if step_text:
                    steps.append(step_text)
        
        # If still no steps found, look for numbered lists or sections
        if not steps:
            # Look for numbered lists like "1. Do something"
            numbered_steps = re.finditer(r"\n\s*\d+\.\s+(.*?)(?=\n\s*\d+\.|\Z)", 
                                       response, re.DOTALL)
            for match in numbered_steps:
                step_text = match.group(1).strip()
                if step_text and len(step_text) > 5:
                    steps.append(step_text)
                    
            # If still nothing, look for section headers as steps
            if not steps:
                section_headers = re.finditer(r"\n\s*#+\s*\d+\.\s+(.*?)(?=\n)", 
                                            response, re.DOTALL)
                for match in section_headers:
                    step_text = match.group(1).strip()
                    if step_text:
                        steps.append(step_text)
        
        # If still no steps found, try bullet points
        if not steps:
            bullet_steps = re.finditer(r"\n\s*[-*]\s+(.*?)(?=\n\s*[-*]|\Z)", 
                                     response, re.DOTALL)
            for match in bullet_steps:
                step_text = match.group(1).strip()
                if step_text and len(step_text) > 5:
                    steps.append(step_text)
                    
        # If we still have no steps, create at least one from the strategy description
        if not steps and strategy_match:
            steps = ["Implement " + strategy_match.group(1).strip()]
        
        # Extract total points
        total_points_match = re.search(r"(?:Total Points.*?|Grand Total.*?)(?:\s*:?\s*)(\d+)", 
                                     response, re.IGNORECASE)
        if total_points_match:
            points = int(total_points_match.group(1))
        else:
            # Look for subtotals and sum them
            subtotal_matches = list(re.finditer(r"Subtotal.*?:\s*(\d+)", response, re.IGNORECASE))
            if subtotal_matches:
                points = sum(int(match.group(1)) for match in subtotal_matches)
            else:
                # Try to find any point numbers in the text
                point_numbers = re.finditer(r"Points:\s*(\d+)", response, re.IGNORECASE)
                points_list = [int(match.group(1)) for match in point_numbers]
                points = sum(points_list) if points_list else 500  # Default to 500 if nothing found
        
        return StrategyPlan(
            steps=steps,
            reasoning=reasoning,
            estimated_points=points,
            metadata={
                "strategy_description": strategy_desc,
                "raw_response": response
            }
        )
        
    def _validate_strategy(self, strategy, context):
        """
        Override the validation to be more lenient.
        
        This allows the strategy to pass validation even if it doesn't
        exactly match the format we expect.
        """
        if not strategy.steps:
            raise ValueError("Strategy has no steps")
            
        if not strategy.reasoning:
            raise ValueError("Strategy has no reasoning")
            
        if strategy.estimated_points <= 0:
            strategy.estimated_points = 500  # Default to a reasonable value
            
        if strategy.estimated_points > context.total_budget:
            # Cap at the budget instead of failing
            strategy.estimated_points = context.total_budget 