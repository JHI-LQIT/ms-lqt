#!/bin/bash
# Script to run the mock oscilloscope publisher

echo "======================================"
echo "Mock Oscilloscope Publisher"
echo "======================================"
echo

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Starting mock oscilloscope publisher..."
echo "Press Ctrl+C to stop"
echo

# Run with default parameters or pass arguments
python3 -m ms_viewer.mock_publisher "$@"
