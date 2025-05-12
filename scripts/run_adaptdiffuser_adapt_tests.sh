#!/bin/bash
# Test script for AdaptDiffuser 'adapt' CLI command with various parameters

# Set up environment
export PYTHONPATH=.

# Create output directory for test results
TEST_DIR="./test_results/adaptdiffuser_adapt"
mkdir -p $TEST_DIR

echo "==============================================="
echo "AdaptDiffuser 'adapt' CLI Parameter Testing Suite"
echo "==============================================="
echo

# Function to check if GPU is available
check_gpu() {
  python -c "import torch; print('GPU available: ' + str(torch.cuda.is_available())); exit(0 if torch.cuda.is_available() else 1)"
  return $?
}

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