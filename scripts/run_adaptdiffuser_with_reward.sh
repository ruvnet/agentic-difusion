#!/bin/bash
# Script to run the AdaptDiffuser tests with the custom reward model

set -e

echo "======================================"
echo "Testing AdaptDiffuser with Custom Reward Model"
echo "======================================"

# Create test output directory
mkdir -p test_results/adaptdiffuser_with_reward/logs

# Step 1: Install the package in development mode if needed
echo -e "\n[Step 1] Ensuring package is installed"
pip install -e . >/dev/null 2>&1

# Step 2: Run the test script
echo -e "\n[Step 2] Running AdaptDiffuser tests with reward model"
python scripts/test_adaptdiffuser_with_reward.py --device cpu

# Check if tests were successful
if [ $? -eq 0 ]; then
    echo -e "\n✅ Tests completed successfully!"
else
    echo -e "\n❌ Some tests failed. Check the logs for details."
fi

echo -e "\nTest results are available in: test_results/adaptdiffuser_with_reward/"