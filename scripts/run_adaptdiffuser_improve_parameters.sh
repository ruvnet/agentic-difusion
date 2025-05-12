#!/bin/bash
# Test script for AdaptDiffuser 'improve' CLI command with various parameters
# Tests different iterations, trajectories, quality thresholds, and error handling

# Set up environment
export PYTHONPATH=.

# Create output directory for test results
OUTPUT_DIR="./test_results/adaptdiffuser_improve"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

echo "==============================================="
echo "AdaptDiffuser 'improve' CLI Parameters Testing Suite"
echo "==============================================="
echo

# Function to check if GPU is available
check_gpu() {
  python -c "import torch; print('GPU available: ' + str(torch.cuda.is_available())); exit(0 if torch.cuda.is_available() else 1)"
  return $?
}

# Parse command line arguments
TEST_SUITE="all"
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case $1 in
    --test)
      TEST_SUITE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--test all|iterations|trajectories|quality|error_handling] [--device cpu|gpu|both]"
      exit 1
      ;;
  esac
done

# Validate test suite argument
if [[ "$TEST_SUITE" != "all" && "$TEST_SUITE" != "iterations" && "$TEST_SUITE" != "trajectories" && "$TEST_SUITE" != "quality" && "$TEST_SUITE" != "error_handling" ]]; then
  echo "Invalid test suite: $TEST_SUITE"
  echo "Valid options: all, iterations, trajectories, quality, error_handling"
  exit 1
fi

# Validate device argument
if [[ "$DEVICE" != "cpu" && "$DEVICE" != "gpu" && "$DEVICE" != "both" ]]; then
  echo "Invalid device: $DEVICE"
  echo "Valid options: cpu, gpu, both"
  exit 1
fi

# Check GPU availability if GPU testing requested
if [[ "$DEVICE" == "gpu" || "$DEVICE" == "both" ]]; then
  echo "Checking for GPU availability..."
  if ! check_gpu; then
    echo "Warning: GPU requested but not available."
    if [[ "$DEVICE" == "gpu" ]]; then
      echo "Falling back to CPU."
      DEVICE="cpu"
    else
      echo "Will run tests on CPU only."
      DEVICE="cpu"
    fi
  else
    echo "GPU is available."
  fi
fi

echo "Running test suite: $TEST_SUITE on device: $DEVICE"
echo

# Run the Python test script with the specified arguments
python scripts/test_adaptdiffuser_improve_parameters.py --test "$TEST_SUITE" --device "$DEVICE"

# Check if the test was successful
if [ $? -eq 0 ]; then
  echo "Tests completed successfully!"
else
  echo "Tests failed with errors."
  exit 1
fi

echo "Test results are available in: $OUTPUT_DIR"
echo "Summary report can be found in the output directory."

# Make the script executable if run directly (useful for first-time execution)
chmod +x scripts/run_adaptdiffuser_improve_parameters.sh

exit 0