"""
Verify the development setup is working correctly.
"""
import sys
import importlib.util
import os
from pathlib import Path

def check_environment():
    """Check if the environment is correctly set up."""
    print("-" * 50)
    print("Midpoint Environment Verification")
    print("-" * 50)
    
    # Python information
    print("\n1. Python Environment:")
    print(f"   Python executable: {sys.executable}")
    print(f"   Python version: {sys.version.split()[0]}")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✅ Running in a virtual environment")
    else:
        print("   ❌ NOT running in a virtual environment")
    
    # Check if midpoint is importable
    print("\n2. Package Status:")
    try:
        import midpoint
        print(f"   ✅ midpoint package found: {midpoint.__file__}")
        print(f"   ✅ midpoint version: {midpoint.__version__}")
        
        # Check submodules
        try:
            import midpoint.agents
            print(f"   ✅ midpoint.agents subpackage found")
        except ImportError as e:
            print(f"   ❌ Cannot import midpoint.agents: {e}")
            
        try:
            from midpoint.agents import models, goal_decomposer, tools, config
            print(f"   ✅ All agent modules imported successfully")
        except ImportError as e:
            print(f"   ❌ Cannot import agent modules: {e}")
            
    except ImportError as e:
        print(f"   ❌ Cannot import midpoint package: {e}")
    
    # Check for conflicting packages
    print("\n3. Potential Conflicts:")
    try:
        spec = importlib.util.find_spec("agents")
        if spec:
            print(f"   ⚠️ Found potentially conflicting 'agents' package at: {spec.origin}")
        else:
            print("   ✅ No conflicting 'agents' package found")
    except (ImportError, AttributeError):
        print("   ✅ No conflicting 'agents' package found")
    
    # Check OpenAI API key
    print("\n4. OpenAI API Key:")
    if "OPENAI_API_KEY" in os.environ:
        key = os.environ["OPENAI_API_KEY"]
        masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
        print(f"   ✅ OPENAI_API_KEY is set: {masked_key}")
    else:
        print("   ❌ OPENAI_API_KEY is not set in environment")
        
        # Check .env file
        env_path = Path(".env")
        if env_path.exists():
            content = env_path.read_text()
            if "OPENAI_API_KEY=" in content:
                print("   ⚠️ OPENAI_API_KEY found in .env file but not loaded in environment")
                print("      Try: source .env (or use python-dotenv in your code)")
            else:
                print("   ❌ OPENAI_API_KEY not found in .env file")
        else:
            print("   ❌ No .env file found")
    
    # Check test repository
    print("\n5. Test Repository:")
    repo_path = os.getenv("MIDPOINT_TEST_REPO", os.path.expanduser("~/midpoint-test-repo"))
    repo_path = Path(repo_path)
    if repo_path.exists():
        if (repo_path / ".git").exists():
            print(f"   ✅ Test repository found at: {repo_path}")
        else:
            print(f"   ⚠️ Directory exists but not a git repository: {repo_path}")
    else:
        print(f"   ❌ Test repository not found at: {repo_path}")
        print("      Try running: python examples/setup_test_repo.py")
    
    print("\n" + "-" * 50)
    print("Verification complete!")
    print("-" * 50)

if __name__ == "__main__":
    check_environment() 