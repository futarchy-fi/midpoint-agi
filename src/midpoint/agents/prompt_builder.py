"""
Shared prompt builder infrastructure for all agents.

This module provides a base PromptBuilder class that handles:
- Section-based prompt construction
- Importance-based section ordering
- Metadata tracking for debugging
- Consistent formatting

Agent-specific builders can inherit from this and customize:
- Header format
- Final instructions
- Section building logic
"""

from typing import Dict, Any, Tuple, Optional, List
import logging


class PromptBuilder:
    """
    Base class for building structured prompts with section tracking.
    
    This class provides the core functionality for building prompts:
    - Adding sections with importance levels
    - Sorting sections by importance
    - Tracking metadata for debugging
    - Building final prompt with consistent formatting
    """
    
    def __init__(self, identifier: str, logger: logging.Logger, header: Optional[str] = None, final_instructions: Optional[str] = None):
        """
        Initialize the prompt builder.
        
        Args:
            identifier: Unique identifier for this prompt (e.g., goal_id, task_id)
            logger: Logger instance for debugging
            header: Optional custom header (if None, uses default)
            final_instructions: Optional custom final instructions (if None, uses default)
        """
        self.identifier = identifier
        self.logger = logger
        self.sections: List[Dict[str, Any]] = []
        self.header = header
        self.final_instructions = final_instructions
    
    def add_section(self, name: str, content: str, importance: str = "normal", source: str = "unknown"):
        """
        Add a section to the prompt.
        
        Args:
            name: Section name (e.g., "Task Details", "Failure History")
            content: Section content (text)
            importance: "critical", "important", or "normal"
            source: Where this data came from (e.g., "goal_file", "memory", "context")
        """
        if not content or not content.strip():
            self.logger.debug(f"Skipping empty section: {name}")
            return
            
        section = {
            "name": name,
            "content": content.strip(),
            "size": len(content),
            "importance": importance,
            "source": source
        }
        self.sections.append(section)
        self.logger.debug(f"Added section '{name}': {len(content)} chars (importance: {importance})")
    
    def get_header(self) -> str:
        """
        Get the prompt header. Override in subclasses for custom headers.
        
        Returns:
            Header string
        """
        if self.header:
            return self.header
        return f"Request [{self.identifier}]"
    
    def get_final_instructions(self) -> List[str]:
        """
        Get the final instructions. Override in subclasses for custom instructions.
        
        Returns:
            List of instruction lines
        """
        if self.final_instructions:
            return [self.final_instructions]
        return ["Based on ALL the context above, complete the requested task."]
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """
        Build the final prompt and return it with metadata.
        
        Returns:
            Tuple of (prompt_text, metadata_dict)
            metadata_dict contains:
                - sections: List of section metadata
                - total_size: Total character count
                - section_count: Number of sections
                - importance_breakdown: Count of sections by importance
        """
        # Sort sections by importance (critical first, then important, then normal)
        importance_order = {"critical": 0, "important": 1, "normal": 2}
        sorted_sections = sorted(
            self.sections,
            key=lambda s: (importance_order.get(s["importance"], 3), s["name"])
        )
        
        # Build the prompt
        prompt_lines = [
            self.get_header(),
            "=" * 60,
            ""
        ]
        
        # Add sections in order
        for section in sorted_sections:
            prompt_lines.append(section["content"])
            prompt_lines.append("")  # Blank line between sections
        
        # Add final instructions
        prompt_lines.append("=" * 60)
        prompt_lines.extend(self.get_final_instructions())
        
        final_prompt = "\n".join(prompt_lines)
        
        # Build metadata
        importance_breakdown = {}
        for section in self.sections:
            imp = section["importance"]
            importance_breakdown[imp] = importance_breakdown.get(imp, 0) + 1
        
        metadata = {
            "sections": [
                {
                    "name": s["name"],
                    "size": s["size"],
                    "importance": s["importance"],
                    "source": s["source"]
                }
                for s in sorted_sections
            ],
            "total_size": len(final_prompt),
            "section_count": len(self.sections),
            "importance_breakdown": importance_breakdown
        }
        
        self.logger.debug(f"Built prompt: {len(final_prompt)} chars, {len(self.sections)} sections")
        return final_prompt, metadata
    
    def get_context_summary(self) -> str:
        """
        Get a summary of the context being built (for debugging).
        
        Returns:
            Summary string
        """
        if not self.sections:
            return "No sections added yet"
        
        summary_lines = [f"Prompt Builder for {self.identifier}:"]
        for section in self.sections:
            summary_lines.append(f"  - {section['name']}: {section['size']} chars ({section['importance']})")
        return "\n".join(summary_lines)


class TaskExecutionPromptBuilder(PromptBuilder):
    """
    Builds prompts for task execution with section tracking and debugging support.
    
    Inherits from PromptBuilder and customizes header and final instructions for task execution.
    """
    
    def __init__(self, task_id: str, logger: logging.Logger):
        """
        Initialize the task execution prompt builder.
        
        Args:
            task_id: The task/goal ID being executed
            logger: Logger instance for debugging
        """
        header = f"Task Execution Request [{task_id}]"
        final_instructions = [
            "Based on ALL the context above, execute the task.",
            "Use the available tools to implement the necessary changes.",
            "Validate your changes and create appropriate commits if needed.",
            "Provide a final response in JSON format with summary, success status, validation steps, and git hash."
        ]
        super().__init__(task_id, logger, header=header, final_instructions="\n".join(final_instructions))


class ValidationPromptBuilder(PromptBuilder):
    """
    Builds prompts for goal validation with section tracking and debugging support.
    
    Inherits from PromptBuilder and customizes header and final instructions for validation.
    """
    
    def __init__(self, goal_id: str, logger: logging.Logger):
        """
        Initialize the validation prompt builder.
        
        Args:
            goal_id: The goal ID being validated
            logger: Logger instance for debugging
        """
        header = f"Goal Validation Request [{goal_id}]"
        final_instructions = [
            "Based on ALL the context above, validate the goal's completion criteria.",
            "1. Verify you are on the target git branch using tools.",
            "2. Use tools to get the repository diff between the initial and final git hashes.",
            "3. Use tools to investigate memory changes between the initial and final memory hashes (if applicable).",
            "4. Analyze the gathered evidence against the validation criteria.",
            "5. Provide your assessment in the required JSON format with criteria_results, score, and reasoning."
        ]
        super().__init__(goal_id, logger, header=header, final_instructions="\n".join(final_instructions))

