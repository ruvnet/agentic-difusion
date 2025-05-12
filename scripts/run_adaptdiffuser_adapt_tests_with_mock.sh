#!/bin/bash
# Test script for AdaptDiffuser 'adapt' CLI command with various parameters
# This script applies a mock reward model first for full testing

# Set up environment
export PYTHONPATH=.

# Create output directory for test results
TEST_DIR="./test_results/adaptdiffuser_adapt_with_mock"
mkdir -p $TEST_DIR

echo "==============================================="
echo "AdaptDiffuser 'adapt' CLI Parameter Testing Suite (with Mock Reward Model)"
echo "==============================================="
echo

# Apply the mock reward model patch
echo "Installing mock reward model..."
python scripts/mock_adaptdiffuser_reward.py --reward-base 0.65 --reward-noise 0.3
PATCH_STATUS=$?

if [ $PATCH_STATUS -ne 0 ]; then
  echo "Failed to install mock reward model (exit code $PATCH_STATUS)"
  exit 1
fi

echo "Mock reward model installed successfully"
echo

# Function to check if GPU is available
check_gpu() {
  python -c "import torch; print('GPU available: ' + str(torch.cuda.is_available())); exit(0 if torch.cuda.is_available() else 1)"
  return $?
}

# Set the test directory in the Python script via environment variable
export ADAPTDIFFUSER_TEST_OUTPUT_DIR=$TEST_DIR

# Run CPU tests by default
if [ "$1" = "--device" ] && [ "$2" = "gpu" ]; then
  # Check if GPU is available before running GPU tests
  if check_gpu; then
    echo "Running tests on GPU..."
    python scripts/test_adaptdiffuser_adapt_cli.py --device gpu
  else
    echo "GPU requested but not available. Falling back to CPU."
    python scripts/test_adaptdiffuser_adapt_cli.py --device cpu
  fi
elif [ "$1" = "--device" ] && [ "$2" = "both" ]; then
  echo "Running tests on both CPU and GPU (if available)..."
  python scripts/test_adaptdiffuser_adapt_cli.py --device both
elif [ "$1" = "--test" ]; then
  echo "Running specific test: $2..."
  python scripts/test_adaptdiffuser_adapt_cli.py --test $2
else
  echo "Running tests on CPU..."
  python scripts/test_adaptdiffuser_adapt_cli.py --device cpu
fi

# Check for run success
if [ $? -eq 0 ]; then
  echo "Tests completed successfully!"
else
  echo "Error: Tests failed with exit code $?"
  exit 1
fi

echo "Test results saved to: $TEST_DIR"
echo "To view the summary: cat $TEST_DIR/summary_report.json"