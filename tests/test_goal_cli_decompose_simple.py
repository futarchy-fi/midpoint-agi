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
            "description": "Test goal description",
            "parent_goal": "",
            "timestamp": "20250101_000000"
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
    
    @patch('midpoint.goal_cli.agent_decompose_goal')
    def test_decompose_command_success(self, mock_decompose_goal):
        """Test successful goal decomposition."""
        # Setup mock result
        mock_result = {
            "success": True,
            "next_step": "Implement feature X",
            "validation_criteria": ["Test criteria"],
            "requires_further_decomposition": True,
            "git_hash": "abc123",
            "goal_file": "G1-S1.json"
        }
        mock_decompose_goal.return_value = mock_result
        
        # Set up arguments for the command
        sys.argv = ["goal_cli.py", "decompose", self.goal_id, "--bypass-validation"]
        
        # Capture stdout
        import io
        stdout_buffer = io.StringIO()
        orig_stdout = sys.stdout
        sys.stdout = stdout_buffer
        
        try:
            # Run the command through a small script that mimics calling the function directly
            import asyncio
            async def run_command():
                return await midpoint.goal_cli.decompose_existing_goal(
                    self.goal_id, 
                    debug=False, 
                    quiet=False, 
                    bypass_validation=True
                )
            
            result = asyncio.run(run_command())
            
            # Verify success
            self.assertTrue(result)
            
            # Verify output
            output = stdout_buffer.getvalue()
            self.assertIn("Goal G1 successfully decomposed into subgoals", output)
            self.assertIn("Next step: Implement feature X", output)
            self.assertIn("Validation criteria:", output)
            self.assertIn("- Test criteria", output)
            
            # Verify the mock was called with the right parameters
            mock_decompose_goal.assert_called_once()
            args, kwargs = mock_decompose_goal.call_args
            self.assertEqual(kwargs["goal"], "Test goal description")
            self.assertEqual(kwargs["parent_goal"], self.goal_id)
            self.assertTrue(kwargs["bypass_validation"])
        finally:
            # Restore stdout
            sys.stdout = orig_stdout
    
    @patch('midpoint.goal_cli.agent_decompose_goal')
    def test_decompose_command_failure(self, mock_decompose_goal):
        """Test failed goal decomposition."""
        # Setup mock result for failure
        mock_result = {
            "success": False,
            "error": "Decomposition failed"
        }
        mock_decompose_goal.return_value = mock_result
        
        # Reset log capture
        self.log_capture.truncate(0)
        self.log_capture.seek(0)
        
        # Run the command through a small script
        import asyncio
        async def run_command():
            return await midpoint.goal_cli.decompose_existing_goal(
                self.goal_id, 
                debug=False, 
                quiet=False, 
                bypass_validation=True
            )
        
        result = asyncio.run(run_command())
        
        # Verify failure
        self.assertFalse(result)
        
        # Verify error output was logged
        log_output = self.log_capture.getvalue()
        self.assertIn("Failed to decompose goal", log_output)
    
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