#!/bin/bash
# Build script for OMR Processor (macOS/Linux)
# Creates distributable application

set -e

echo "========================================"
echo "OMR Processor - Build Script"
echo "========================================"

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
echo "Activating virtual environment..."
source venv/bin/activate

# Check if PyInstaller is installed
if ! pip show pyinstaller &> /dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build dist

# Create assets directory if it doesn't exist
mkdir -p assets

# Build application
echo ""
echo "Building application..."
echo "This may take a few minutes..."
echo ""

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS build
    pyinstaller build.spec --clean --noconfirm

    echo ""
    echo "========================================"
    echo "Build complete!"
    echo ""
    echo "Application bundle created at:"
    echo "  dist/OMR Processor.app"
    echo ""
    echo "To run the app:"
    echo "  open 'dist/OMR Processor.app'"
    echo ""
    echo "To create a DMG installer:"
    echo "  ./create-dmg.sh"
    echo "========================================"
else
    # Linux build
    pyinstaller build.spec --clean --noconfirm

    echo ""
    echo "========================================"
    echo "Build complete!"
    echo ""
    echo "Executable created at:"
    echo "  dist/OMR Processor"
    echo "========================================"
fi
