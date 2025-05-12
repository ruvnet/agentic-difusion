# AdaptDiffuser CLI Parameter Testing Suite

This testing suite is designed to evaluate the `adaptdiffuser adapt` CLI command with various parameters to ensure proper functionality, error handling, and performance across different configurations.

## Overview

The test script evaluates the AdaptDiffuser CLI's adaptation capabilities by testing:

1. **Different learning rates** (1e-5, 1e-4, 1e-3, 1e-2)
2. **Different batch sizes** (1, 2, 4, 8)
3. **Different quality thresholds** (0.5, 0.6, 0.7, 0.8)
4. **Error handling** (non-existent examples file)

Each test runs the CLI command with different parameters and captures:
- Return codes to verify successful execution
- Metrics output to evaluate adaptation performance
- Error outputs for failure cases to ensure graceful handling

## Usage

Run the testing script with:

```bash
# Run all tests
./scripts/run_adaptdiffuser_cli_parameters.sh

# Run a specific test suite
./scripts/run_adaptdiffuser_cli_parameters.sh learning_rates
./scripts/run_adaptdiffuser_cli_parameters.sh batch_sizes
./scripts/run_adaptdiffuser_cli_parameters.sh quality_thresholds
./scripts/run_adaptdiffuser_cli_parameters.sh error_handling
```

## Test Results

Results are saved to the `test_results/adaptdiffuser_cli_parameters/` directory:

- Individual test results as JSON files
- Logs in the `logs/` subdirectory
- A summary report with consolidated results from all test suites

To view the summary report:

```bash
cat test_results/adaptdiffuser_cli_parameters/summary_report_*.json
```

## Expected Behavior

- **Learning rates**: The CLI should accept different learning rates without errors
- **Batch sizes**: The CLI should handle different batch sizes appropriately
- **Quality thresholds**: The CLI should properly filter examples based on provided thresholds
- **Error handling**: The CLI should fail gracefully with appropriate error messages when given invalid inputs

## Notes on Reward Models

During tests, you may see messages about missing reward models. This is expected when testing without proper examples or reward models set up. The tests verify that the CLI handles these situations gracefully without crashing.

```
WARNING - No reward model available for adaptation, returning empty metrics
Adaptation metrics:
  status: skipped
  reason: no_reward_model
```

This indicates that the CLI is properly detecting the missing reward model and providing appropriate feedback.

## Understanding Test Output

- `return_code: 0` indicates successful execution
- `return_code: non-zero` indicates an error occurred
- `metrics` section contains any adaptation metrics reported by the CLI
- Error messages are captured and logged for analysis

## Extending the Tests

To add new test cases:
1. Modify `scripts/test_adaptdiffuser_cli_parameters.py`
2. Add new test functions for additional parameters
3. Update the main function to include these tests