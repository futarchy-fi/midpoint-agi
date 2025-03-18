#!/usr/bin/env python
"""Setup script for midpoint development environment."""
import os
import subprocess
import sys

def main():
    """Set up the development environment."""
    # Create and activate virtual environment
    if not os.path.exists(".venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    
    # Get the path to activate script
    if sys.platform == "win32":
        activate_script = ".venv\\Scripts\\activate"
        pip_cmd = [".venv\\Scripts\\pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
        pip_install_cmd = [".venv\\Scripts\\pip", "install", "-e", ".[dev]"]
        python_exe = ".venv\\Scripts\\python"
    else:
        activate_script = ".venv/bin/activate"
        pip_cmd = [".venv/bin/pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
        pip_install_cmd = [".venv/bin/pip", "install", "-e", ".[dev]"]
        python_exe = ".venv/bin/python"
    
    # Upgrade pip and other build tools
    print("Upgrading pip, setuptools, and wheel...")
    subprocess.run(pip_cmd, check=True)
    
    # Install dependencies
    print("Installing development dependencies...")
    subprocess.run(pip_install_cmd, check=True)
    
    print("\nSetup complete! To activate the environment:")
    print(f"source {activate_script}" if sys.platform != "win32" else activate_script)
    
    # Ask about creating test repo
    setup_test_repo = input("\nCreate test repository now? (y/n): ")
    if setup_test_repo.lower() == "y":
        subprocess.run([python_exe, "examples/setup_test_repo.py"])
        
    # Ask about setting up OpenAI API key
    setup_api_key = input("\nSet up OpenAI API key now? (y/n): ")
    if setup_api_key.lower() == "y":
        api_key = input("Enter your OpenAI API key: ")
        
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                content = f.read()
            
            if "OPENAI_API_KEY=" in content:
                content = "\n".join([
                    line if not line.startswith("OPENAI_API_KEY=") else f"OPENAI_API_KEY={api_key}"
                    for line in content.split("\n")
                ])
            else:
                content += f"\nOPENAI_API_KEY={api_key}\n"
        else:
            content = f"OPENAI_API_KEY={api_key}\n"
            
        with open(env_path, "w") as f:
            f.write(content)
            
        print("API key saved to .env file.")
    
    print("\nDevelopment environment setup complete!")
    print("\nNext steps:")
    print("1. Activate your virtual environment")
    print("2. Try running the test script: python examples/test_goal_decomposer.py")

if __name__ == "__main__":
    main() 