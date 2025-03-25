"""
Tests for the 'goal decompose' command in goal_cli.py.
"""

import os
import json
import asyncio
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import tempfile
import shutil

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from midpoint.goal_cli import decompose_existing_goal


class TestGoalCliDecompose(unittest.TestCase):
    """Tests for the 'goal decompose' command in goal_cli.py."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary goal directory
        self.temp_dir = tempfile.mkdtemp()
        self.goal_dir = Path(self.temp_dir) / ".goal"
        self.goal_dir.mkdir(exist_ok=True)
        
        # Create a test goal file
        self.test_goal_id = "G1"
        self.test_goal_file = self.goal_dir / f"{self.test_goal_id}.json"
        self.test_goal_content = {
            "goal_id": self.test_goal_id,
            "description": "Test goal description",
            "parent_goal": "",
            "timestamp": "20250101_000000"
        }
        
        with open(self.test_goal_file, 'w') as f:
            json.dump(self.test_goal_content, f)
        
        # Save current directory to restore it later
        self.orig_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        self.addCleanup(self.cleanup)
    
    def cleanup(self):
        """Clean up after the test."""
        # Restore current directory
        os.chdir(self.orig_cwd)
        
        # Clean up the temp directory
        shutil.rmtree(self.temp_dir)
    
    @patch('midpoint.goal_cli.agent_decompose_goal')
    async def test_decompose_existing_goal_success(self, mock_decompose_goal):
        """Test successful goal decomposition."""
        # Setup mock response from decompose_goal
        mock_result = {
            "success": True,
            "next_step": "Implement feature X",
            "validation_criteria": ["Code passes tests", "Feature works as expected"],
            "requires_further_decomposition": True,
            "git_hash": "abcdef123456",
            "memory_hash": None,
            "goal_file": "G1-S1.json"
        }
        mock_decompose_goal.return_value = mock_result
        
        # Capture stdout and stderr
        import io
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        try:
            # Run the decompose_existing_goal function
            result = await decompose_existing_goal(self.test_goal_id, bypass_validation=True)
            
            # Check the result
            self.assertTrue(result)
            
            # Check the output
            stdout = stdout_buffer.getvalue()
            self.assertIn("Goal G1 successfully decomposed into subgoals", stdout)
            self.assertIn("Next step: Implement feature X", stdout)
            self.assertIn("Validation criteria:", stdout)
            self.assertIn("- Code passes tests", stdout)
            self.assertIn("- Feature works as expected", stdout)
            self.assertIn("Requires further decomposition: Yes", stdout)
            self.assertIn("Goal file: G1-S1.json", stdout)
            
            # Verify the mock was called with the right arguments
            mock_decompose_goal.assert_called_once()
            call_args = mock_decompose_goal.call_args[1]
            self.assertEqual(call_args["goal"], "Test goal description")
            self.assertEqual(call_args["parent_goal"], self.test_goal_id)
            self.assertTrue(call_args["bypass_validation"])
        finally:
            # Restore stdout and stderr
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
    
    @patch('midpoint.goal_cli.agent_decompose_goal')
    async def test_decompose_existing_goal_failure(self, mock_decompose_goal):
        """Test failed goal decomposition."""
        # Setup mock response for failure
        mock_result = {
            "success": False,
            "error": "Failed to decompose goal"
        }
        mock_decompose_goal.return_value = mock_result
        
        # Capture stdout and stderr
        import io
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        try:
            # Run the decompose_existing_goal function
            result = await decompose_existing_goal(self.test_goal_id, bypass_validation=True)
            
            # Check the result
            self.assertFalse(result)
            
            # Check the error output
            stderr = stderr_buffer.getvalue()
            self.assertIn("Failed to decompose goal", stderr)
        finally:
            # Restore stdout and stderr
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
    
    async def test_decompose_nonexistent_goal(self):
        """Test decomposing a nonexistent goal."""
        # Capture stdout and stderr
        import io
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        try:
            # Try to decompose a goal that doesn't exist
            result = await decompose_existing_goal("NonExistentGoal", bypass_validation=True)
            
            # Check the result
            self.assertFalse(result)
            
            # Check the error output
            stderr = stderr_buffer.getvalue()
            self.assertIn("Goal NonExistentGoal not found", stderr)
        finally:
            # Restore stdout and stderr
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr


def run_async_test(coro):
    """Helper function to run async tests."""
    return asyncio.run(coro)


# Create test cases that can be run synchronously
class TestGoalCliDecomposeSync(unittest.TestCase):
    """Synchronous wrapper for async tests."""
    
    @patch('midpoint.goal_cli.agent_decompose_goal')
    def test_decompose_existing_goal_success(self, mock_decompose_goal):
        """Run the async test_decompose_existing_goal_success."""
        test = TestGoalCliDecompose()
        test.setUp()
        try:
            run_async_test(test.test_decompose_existing_goal_success(mock_decompose_goal))
        finally:
            test.cleanup()
    
    @patch('midpoint.goal_cli.agent_decompose_goal')
    def test_decompose_existing_goal_failure(self, mock_decompose_goal):
        """Run the async test_decompose_existing_goal_failure."""
        test = TestGoalCliDecompose()
        test.setUp()
        try:
            run_async_test(test.test_decompose_existing_goal_failure(mock_decompose_goal))
        finally:
            test.cleanup()
    
    def test_decompose_nonexistent_goal(self):
        """Run the async test_decompose_nonexistent_goal."""
        test = TestGoalCliDecompose()
        test.setUp()
        try:
            run_async_test(test.test_decompose_nonexistent_goal())
        finally:
            test.cleanup()


if __name__ == "__main__":
    unittest.main() 