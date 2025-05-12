#!/bin/bash
# Test script for AdaptDiffuser CLI with various arguments and options
# Tests both CPU and GPU options

# Set up environment
export PYTHONPATH=.

# Create output directory for test results
RESULT_DIR="./test_results"
mkdir -p $RESULT_DIR

echo "==============================================="
echo "AdaptDiffuser CLI Testing Suite"
echo "==============================================="
echo

# Function to check if GPU is available
check_gpu() {
  python -c "import torch; print('GPU available: ' + str(torch.cuda.is_available())); exit(0 if torch.cuda.is_available() else 1)"
  return $?
}

# Explicitly force CPU tests first
echo "=== Testing on CPU ==="
echo

# Test 1: Generate with CPU
echo "Test 1: Generate trajectories on CPU"
python -m agentic_diffusion --config ./config/test_cpu.yaml adaptdiffuser generate "navigate_maze" \
  --batch-size 2 --guidance-scale 1.5 \
  --output "$RESULT_DIR/cpu_generate_test.json"
echo

# Test 2: Adapt with CPU
echo "Test 2: Adapt to a task on CPU"
python -m agentic_diffusion --config ./config/test_cpu.yaml adaptdiffuser adapt "avoid_obstacles" \
  --iterations 2 --batch-size 4 --learning-rate 1e-4
echo

# Test 3: Self-improve with CPU
echo "Test 3: Self-improve on a task on CPU"
python -m agentic_diffusion --config ./config/test_cpu.yaml adaptdiffuser improve "reach_goal" \
  --iterations 1 --trajectories 10
echo

# Check if GPU is available and run GPU tests
echo "Checking for GPU availability..."
if check_gpu; then
  echo "GPU detected, running GPU tests"
  echo
  
  # Test 4: Generate with GPU
  echo "Test 4: Generate trajectories on GPU"
  python -m agentic_diffusion --config ./config/test_gpu.yaml adaptdiffuser generate "navigate_maze" \
    --batch-size 4 --guidance-scale 2.0 \
    --output "$RESULT_DIR/gpu_generate_test.json"
  echo
  
  # Test 5: Adapt with GPU
  echo "Test 5: Adapt to a task on GPU"
  python -m agentic_diffusion --config ./config/test_gpu.yaml adaptdiffuser adapt "avoid_obstacles" \
    --iterations 3 --batch-size 8 --learning-rate 1e-5
  echo
  
  # Test 6: Self-improve with GPU
  echo "Test 6: Self-improve on a task on GPU"
  python -m agentic_diffusion --config ./config/test_gpu.yaml adaptdiffuser improve "reach_goal" \
    --iterations 2 --trajectories 20
  echo
  
  # Test 7: Mixed test (larger batch size)
  echo "Test 7: Multi-task processing on GPU with larger batch size"
  python -m agentic_diffusion --config ./config/test_gpu.yaml adaptdiffuser generate "solve_complex_maze" \
    --batch-size 16 --guidance-scale 3.0
  echo
else
  echo "No GPU detected, skipping GPU tests"
fi

# Test invalid command (for error handling)
echo "Test: Invalid command (testing error handling)"
python -m agentic_diffusion adaptdiffuser invalid_command "test" || echo "Error handled correctly"
echo

echo "All tests completed!"
echo "Results saved to: $RESULT_DIR"