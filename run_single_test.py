import pytest
import asyncio
import os

# Set dummy OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-" + "a" * 48

if __name__ == "__main__":
    asyncio.run(pytest.main(["-v", "tests/test_goal_decomposer.py::test_goal_decomposition_basic"])) 