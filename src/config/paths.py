"""
Project path configuration and constants.

This module defines standard paths used throughout the application for locating
resources like prompts, data files, and configuration.
"""

from pathlib import Path

# Project structure paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Navigate to project root
SRC_DIR = PROJECT_ROOT / "src"
PROMPTS_DIR = SRC_DIR / "core" / "prompts"  # Location of agent prompt YAML files
