# AdaptDiffuser 'improve' CLI Parameter Testing Suite

This testing suite is designed to evaluate the `adaptdiffuser improve` CLI command with various parameters to ensure proper functionality, error handling, and performance across different configurations.

## Overview

The test script evaluates the AdaptDiffuser CLI's self-improvement capabilities by testing:

1. **Different iteration counts** (1, 5, 10, 20)
2. **Different trajectory counts** (5, 10, 20, 50)
3. **Different quality thresholds** (0.5, 0.6, 0.7, 0.8, 0.9)
4. **Error handling** (missing reward models)
5. **Metrics reporting and analysis**

Each test runs the CLI command with different parameters and captures:
- Return codes to verify successful execution
- Metrics output to evaluate self-improvement performance
- Error outputs for failure cases to ensure graceful handling

## Usage

Run the testing script with:

```bash
# Make the script executable first
chmod +x scripts/run_adaptdiffuser_improve_parameters.sh

# Run all tests on CPU
./scripts/run_adaptdiffuser_improve_parameters.sh

# Run specific test suites
./scripts/run_adaptdiffuser_improve_parameters.sh --test iterations
./scripts/run_adaptdiffuser_improve_parameters.sh --test trajectories
./scripts/run_adaptdiffuser_improve_parameters.sh --test quality
./scripts/run_adaptdiffuser_improve_parameters.sh --test error_handling

# Run on GPU if available
./scripts/run_adaptdiffuser_improve_parameters.sh --device gpu

# Run on both CPU and GPU
./scripts/run_adaptdiffuser_improve_parameters.sh --device both
```

## Test Results

Results are saved to the `test_results/adaptdiffuser_improve/` directory:

- Individual test results as JSON files
- Logs in the `logs/` subdirectory
- A summary report with consolidated results from all test suites and recommendations

To view the summary report:

```bash
cat test_results/adaptdiffuser_improve/summary_report_*.json
```

## Expected Behavior

### Iteration Count Tests
- Different numbers of iterations should have varying effects on model improvement
- Higher iteration counts usually result in more improvement, but may have diminishing returns
- The summary report identifies the optimal number of iterations based on metrics

### Trajectory Count Tests
- Different numbers of trajectories should impact model performance and computation time
- Higher trajectory counts generally sample more of the space but with higher computation cost
- The summary report identifies the optimal number of trajectories based on metrics

### Quality Threshold Tests
- Different threshold values should affect the filtering of samples used for improvement
- Higher thresholds are more selective but may filter out useful examples
- The summary report identifies the optimal threshold based on metrics

### Error Handling Tests
- Tests with missing reward models should fail gracefully
- Clear error messages should indicate the missing components
- The test verifies that the system doesn't crash but provides appropriate feedback

## Metrics Reported

The tests capture and analyze several metrics:

1. **Improvement percentage**: How much the model improved through self-improvement
2. **Final reward**: The reward value achieved after improvement
3. **Execution time**: Time taken for each improvement process
4. **Status**: Whether the improvement process succeeded, failed, or was skipped

## Notes on Reward Models

During tests, you may see messages about missing reward models. This is expected when testing the error handling capabilities. The tests verify that the CLI handles these situations gracefully without crashing.

```
WARNING - No reward model available for improvement, returning empty metrics
Improvement metrics:
  status: skipped
  reason: no_reward_model
```

This indicates that the CLI is properly detecting the missing reward model and providing appropriate feedback.

## Understanding Test Output

- `return_code: 0` indicates successful execution
- `return_code: non-zero` indicates an error occurred
- `metrics` section contains improvement metrics reported by the CLI
- Error messages are captured and logged for analysis

## Extending the Tests

To add new test cases:
1. Modify `scripts/test_adaptdiffuser_improve_parameters.py`
2. Add new test functions for additional parameters
3. Update the main function to include these tests
4. Update the shell script if needed for new parameter options