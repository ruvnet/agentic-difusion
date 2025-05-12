#!/bin/bash
# Run the AdaptDiffuser CLI parameter testing script

echo "==============================================="
echo "AdaptDiffuser 'adapt' CLI Parameter Testing Suite"
echo "==============================================="

# Check if test argument is provided
if [ -n "$1" ]; then
    echo "Running specific test: $1..."
    python scripts/test_adaptdiffuser_cli_parameters.py --test "$1"
else
    echo "Running all tests..."
    python scripts/test_adaptdiffuser_cli_parameters.py --test all
fi

# Check if the tests completed successfully
if [ $? -eq 0 ]; then
    echo "Tests completed successfully!"
    echo "Test results saved to: ./test_results/adaptdiffuser_cli_parameters"
    echo "To view the summary: cat ./test_results/adaptdiffuser_cli_parameters/summary_report_*.json"
else
    echo "Tests failed!"
    exit 1
fi