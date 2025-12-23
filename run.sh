#!/bin/bash
# Run script for OMR Processor (macOS/Linux)
# Runs the application in development mode

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found."
    echo "Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run the application
echo "Starting OMR Processor..."
python src/main.py
