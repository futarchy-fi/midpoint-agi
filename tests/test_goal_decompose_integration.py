"""
Integration tests for the 'goal decompose' command.
"""

import os
import json
import unittest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import tempfile
import shutil

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from midpoint.goal_cli import ensure_goal_dir


class TestGoalDecomposeIntegration(unittest.TestCase):
    """Integration tests for the goal decompose command."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create a .goal directory
        self.goal_dir = Path(self.temp_dir) / ".goal"
        self.goal_dir.mkdir(exist_ok=True)
        
        # Create a sample goal file
        self.goal_id = "G1"
        self.goal_file = self.goal_dir / f"{self.goal_id}.json"
        self.goal_content = {
            "goal_id": self.goal_id,
            "description": "Test goal for decomposition",
            "parent_goal": "",
            "timestamp": "20250101_000000"
        }
        
        with open(self.goal_file, 'w') as f:
            json.dump(self.goal_content, f)
        
        # Path to the repository root
        self.repo_root = Path(__file__).parent.parent
        
        # Create a test script that imports goal_cli and runs the command
        self.test_script = Path(self.temp_dir) / "run_decompose.py"
        with open(self.test_script, "w") as f:
            f.write("""
import sys
import asyncio
from pathlib import Path
import os

# Add src to Python path
sys.path.insert(0, 'SRC_DIR')

# Import the goal_cli module
from midpoint.goal_cli import decompose_existing_goal

async def main():
    goal_id = sys.argv[1]
    debug = "--debug" in sys.argv
    quiet = "--quiet" in sys.argv
    
    # Run the decompose_existing_goal function
    await decompose_existing_goal(goal_id, debug, quiet)

if __name__ == "__main__":
    asyncio.run(main())
""".replace("SRC_DIR", str(self.repo_root / "src")))
    
    def tearDown(self):
        """Clean up after the test."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)
    
    @patch('midpoint.agents.goal_decomposer.decompose_goal')
    def test_goal_decompose_command(self, mock_decompose_goal):
        """Test the 'goal decompose' command."""
        # Setup mock for decompose_goal
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
        
        # Set up environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        
        # Run the test script
        process = subprocess.run(
            [sys.executable, str(self.test_script), self.goal_id],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Print output for debugging
        if process.returncode != 0:
            print(f"STDOUT: {process.stdout}")
            print(f"STDERR: {process.stderr}")
        
        # Expected messages in a successful execution
        expected_messages = [
            "Goal G1 successfully decomposed into subgoals",
            "Next step: Implement feature X",
            "Validation criteria:",
            "- Code passes tests",
            "- Feature works as expected",
            "Requires further decomposition: Yes",
            "Goal file:"
        ]
        
        # Check for expected messages
        for message in expected_messages:
            with self.subTest(message=message):
                self.assertIn(message, process.stdout)
    
    @patch('midpoint.agents.goal_decomposer.decompose_goal')
    def test_goal_decompose_with_debug_flag(self, mock_decompose_goal):
        """Test the 'goal decompose' command with the --debug flag."""
        # Setup mock for decompose_goal
        mock_result = {
            "success": True,
            "next_step": "Debug step",
            "validation_criteria": ["Debug successful"],
            "requires_further_decomposition": False,
            "goal_file": "debug.json"
        }
        mock_decompose_goal.return_value = mock_result
        
        # Set up environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        
        # Run the test script with debug flag
        process = subprocess.run(
            [sys.executable, str(self.test_script), self.goal_id, "--debug"],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Check if debug flag was recognized
        self.assertIn("Debug step", process.stdout)
    
    def test_goal_decompose_nonexistent_goal(self):
        """Test the 'goal decompose' command with a nonexistent goal."""
        # Set up environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        
        # Run the test script with a nonexistent goal
        process = subprocess.run(
            [sys.executable, str(self.test_script), "NonExistentGoal"],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Check if the error message is present
        self.assertIn("Goal NonExistentGoal not found", process.stderr)


if __name__ == "__main__":
    unittest.main() 