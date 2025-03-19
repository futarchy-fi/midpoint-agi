import pytest
import os
import json

# Load API key from config.json
with open('config.json', 'r') as f:
    config = json.load(f)
    os.environ["OPENAI_API_KEY"] = config['openai']['api_key']

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_goal_decomposer.py::test_goal_decomposition_basic"]) 