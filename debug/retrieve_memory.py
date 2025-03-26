from midpoint.agents.tools.memory_tools import retrieve_recent_memory
import os
from datetime import datetime

# Get the memory hash from the most recent commit
memory_hash = "e3b373d"  # Using the same hash for comparison
memory_repo_path = os.path.expanduser("~/.midpoint/memory")

# Retrieve the memory
total_chars, memory_documents = retrieve_recent_memory(
    memory_hash=memory_hash,
    char_limit=10000,  # Using the same limit as in goal_decomposer
    repo_path=memory_repo_path
)

# Create output directory if it doesn't exist
os.makedirs("memory_output", exist_ok=True)

# Save the retrieved memory to a file
output_file = "memory_output/retrieved_memory_fixed.md"
with open(output_file, "w") as f:
    f.write("# Retrieved Memory (Fixed Version)\n\n")
    f.write(f"Memory Hash: {memory_hash}\n")
    f.write(f"Total Characters: {total_chars}\n")
    f.write(f"Number of Documents: {len(memory_documents)}\n")
    f.write(f"Retrieved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    for i, (path, content, timestamp) in enumerate(memory_documents, 1):
        f.write(f"## Document {i}: {path}\n")
        f.write(f"Timestamp: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("```\n")
        f.write(content)
        f.write("\n```\n\n")

print(f"Memory has been saved to {output_file}") 