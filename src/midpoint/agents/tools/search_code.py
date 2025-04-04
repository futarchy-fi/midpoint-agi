import os
import re
import glob
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from midpoint.agents.tools.base import Tool
from midpoint.agents.tools.registry import ToolRegistry

# ... existing code ... 