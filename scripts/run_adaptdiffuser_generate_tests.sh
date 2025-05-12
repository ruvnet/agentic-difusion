#!/bin/bash
# Script to run tests for AdaptDiffuser generate command

# Set up error handling
set -e

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if ! pip list | grep -q "torch"; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run all tests on CPU (default)
echo "Running all tests on CPU..."
python scripts/test_adaptdiffuser_generate.py --device cpu --cleanup

# Check if CUDA is available and run GPU tests if it is
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA is available, running tests on GPU..."
    python scripts/test_adaptdiffuser_generate.py --device gpu
else
    echo "CUDA is not available, skipping GPU tests."
fi

# Run specific test categories if desired
# Uncomment the ones you want to run

# echo "Running batch size tests only..."
# python scripts/test_adaptdiffuser_generate.py --device cpu --test batch

# echo "Running guidance scale tests only..."
# python scripts/test_adaptdiffuser_generate.py --device cpu --test guidance

# echo "Running different tasks tests only..."
# python scripts/test_adaptdiffuser_generate.py --device cpu --test tasks

# echo "Running output format tests only..."
# python scripts/test_adaptdiffuser_generate.py --device cpu --test output

echo "All tests completed!"