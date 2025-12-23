"""
OMR Processor - Main Entry Point
Vidya Bharti Pratibha Khoj 2025-26
"""
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.dirname(os.path.abspath(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Set working directory to project root
os.chdir(project_root)

# Now import using absolute imports from src
from src.gui.app import main

if __name__ == '__main__':
    main()
