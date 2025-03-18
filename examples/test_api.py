"""
A simple script to test the OpenAI API connection.
"""

import os
from openai import OpenAI
from agents.config import get_openai_api_key

def test_api_key(api_key: str) -> bool:
    """Test a specific OpenAI API key."""
    print(f"\nTesting API key: {api_key[:10]}...{api_key[-4:]}")
        
    try:
        # Initialize client
        client = OpenAI(api_key=api_key)
        
        # Try a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ]
        )
        
        # Print response
        print("\n✅ API test successful!")
        print("Response received:", response.choices[0].message.content)
        return True
        
    except Exception as e:
        print("\n❌ API test failed!")
        print("Error:", str(e))
        if "401" in str(e):
            print("\nTroubleshooting tips:")
            print("1. Check if your API key is valid")
            print("2. If using a project-specific key, make sure it's active")
            print("3. Verify the organization ID if you're using one")
            print("4. Try generating a new API key")
        return False

if __name__ == "__main__":
    print("Testing OpenAI API connection...")
    
    # Get API key from environment
    api_key = get_openai_api_key()
    if not api_key:
        print("Error: No API key found. Please set OPENAI_API_KEY in your environment or .env file.")
        exit(1)
    
    # Test the API key
    success = test_api_key(api_key)
    exit(0 if success else 1) 