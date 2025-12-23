#!/bin/bash
# Setup script for OMR Processor (macOS/Linux)
# Creates virtual environment and installs dependencies

set -e

echo "========================================"
echo "OMR Processor - Setup Script"
echo "========================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.9 or later."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check minimum version
MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
    echo "Error: Python 3.9 or later is required. Found $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing..."
    rm -rf venv
fi

$PYTHON_CMD -m venv venv
echo "Virtual environment created."

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install additional macOS dependencies if needed
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "Installing macOS-specific dependencies..."
    pip install pyobjc-core pyobjc-framework-Cocoa pyobjc-framework-WebKit
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "To run the application:"
echo "  ./run.sh"
echo ""
echo "To build the application:"
echo "  ./build.sh"
echo "========================================"
