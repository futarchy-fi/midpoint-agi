import os
from dotenv import load_dotenv
from agents import Agent, Runner

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Set both environment variables for API key and tracing
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_KEY_TRACE"] = api_key

def main():
    # Create a basic agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant that specializes in writing creative content."
    )

    # Run the agent with a task
    result = Runner.run_sync(
        agent,
        "Write a haiku about artificial intelligence and its impact on society."
    )

    # Print the result
    print("\nAgent's Response:")
    print("-" * 50)
    print(result.final_output)
    print("-" * 50)

if __name__ == "__main__":
    main() 