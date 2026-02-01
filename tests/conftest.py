"""
Pytest configuration
Adds project root to sys.path so imports work
"""
import sys
from pathlib import Path

# Add project root to path so 'src' imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
