"""
Simple tests for the goal_cli decompose command integration.
"""

import os
import json
import unittest
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import tempfile
import shutil
import subprocess
import pytest

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import to get the module available for patching
import midpoint.goal_cli


class TestGoalDecomposeSimple(unittest.TestCase):
    """Simple tests for the goal decompose command."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a .goal directory
        self.goal_dir = Path(self.temp_dir) / ".goal"
        self.goal_dir.mkdir(exist_ok=True)
        
        # Create a test goal file
        self.goal_id = "G1"
        self.goal_file = self.goal_dir / f"{self.goal_id}.json"
        self.goal_content = {
            "goal_id": self.goal_id,
            "description": "Test goal",
            "branch_name": "goal-G1",
            "timestamp": "20250324_000000"
        }
        
        with open(self.goal_file, 'w') as f:
            json.dump(self.goal_content, f)
        
        # Save the original directory
        self.original_dir = os.getcwd()
        
        # Change to the temporary directory
        os.chdir(self.temp_dir)
        
        # Save original logging configuration
        self.original_log_level = logging.getLogger().level
        
        # Configure a custom log handler for testing
        self.log_capture = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)
    
    def tearDown(self):
        """Clean up the test environment."""
        # Return to the original directory
        os.chdir(self.original_dir)
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Restore original logging configuration
        logging.getLogger().removeHandler(self.log_handler)
        logging.getLogger().setLevel(self.original_log_level)
    
    @pytest.mark.asyncio
    @patch('midpoint.goal_cli.agent_decompose_goal')
    @patch('midpoint.goal_cli.get_current_branch')
    @patch('subprocess.run')
    async def test_decompose_command_success(self, mock_run, mock_get_branch, mock_decompose_goal, capsys):
        """Test successful goal decomposition."""
        # Setup mock result
        mock_result = {
            "success": True,
            "next_state": "Implement feature X",
            "validation_criteria": ["Test criteria"],
            "requires_further_decomposition": True,
            "git_hash": "abc123",
            "goal_file": "G1.json",
            "is_task": False,
            "memory_hash": None,
            "reasoning": "Test reasoning for implementing feature X",
            "relevant_context": {}
        }
        mock_decompose_goal.return_value = mock_result
        mock_get_branch.return_value = "main"
        mock_run.return_value = type('Result', (), {'stdout': '', 'returncode': 0})
        
        # Reset log capture
        self.log_capture.truncate(0)
        self.log_capture.seek(0)
        
        # Run the command
        result = await midpoint.goal_cli.decompose_existing_goal(
            self.goal_id,
            debug=False,
            quiet=False,
            bypass_validation=True
        )
        
        # Verify success
        assert result is True
        
        # Verify output
        captured = capsys.readouterr()
        assert "Goal G1 successfully decomposed into subgoals" in captured.out
        assert "Next step: Implement feature X" in captured.out
        assert "Test criteria" in captured.out
        
        # Verify the mock was called with the right parameters
        mock_decompose_goal.assert_called_once()
        args, kwargs = mock_decompose_goal.call_args
        self.assertEqual(kwargs["goal"], "Test goal")
        self.assertEqual(kwargs["parent_goal"], self.goal_id)
        self.assertTrue(kwargs["bypass_validation"])
    
    @pytest.mark.asyncio
    @patch('midpoint.goal_cli.agent_decompose_goal')
    @patch('midpoint.goal_cli.get_current_branch')
    @patch('subprocess.run')
    async def test_decompose_command_failure(self, mock_run, mock_get_branch, mock_decompose_goal, capsys):
        """Test failed goal decomposition."""
        # Setup mock result for failure
        mock_result = {
            "success": False,
            "error": "Decomposition failed"
        }
        mock_decompose_goal.return_value = mock_result
        mock_get_branch.return_value = "main"
        mock_run.return_value = type('Result', (), {'stdout': '', 'returncode': 0})
        
        # Reset log capture
        self.log_capture.truncate(0)
        self.log_capture.seek(0)
        
        # Run the command
        result = await midpoint.goal_cli.decompose_existing_goal(
            self.goal_id,
            debug=False,
            quiet=False,
            bypass_validation=True
        )
        
        # Verify failure
        assert result is False
        
        # Verify error output
        captured = capsys.readouterr()
        assert "Failed to decompose goal" in captured.err
        assert "Decomposition failed" in captured.err
    
    def test_decompose_nonexistent_goal(self):
        """Test decomposing a nonexistent goal."""
        # Reset log capture
        self.log_capture.truncate(0)
        self.log_capture.seek(0)
        
        # Run the command for a nonexistent goal
        import asyncio
        async def run_command():
            return await midpoint.goal_cli.decompose_existing_goal(
                "NonExistentGoal", 
                debug=False, 
                quiet=False, 
                bypass_validation=True
            )
        
        result = asyncio.run(run_command())
        
        # Verify failure
        self.assertFalse(result)
        
        # Verify error output was logged
        log_output = self.log_capture.getvalue()
        self.assertIn("Goal NonExistentGoal not found", log_output)


# Make sure to import io at the top with other imports
import io

if __name__ == "__main__":
    unittest.main() 