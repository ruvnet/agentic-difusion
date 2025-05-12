# AdaptDiffuser 'adapt' CLI Testing

This directory contains scripts for testing the AdaptDiffuser 'adapt' CLI command with various parameters, focusing on learning rates, batch sizes, quality thresholds, error handling, and metrics reporting.

## Overview

The testing framework allows you to test the AdaptDiffuser adaptation process with various configurations to:

1. Measure how different learning rates affect adaptation quality and speed
2. Test performance with different batch sizes
3. Evaluate the effect of different quality thresholds on sample filtering
4. Verify error handling for missing components or invalid inputs
5. Track metrics reporting across different configurations

## Quick Start

Run all tests on CPU (default):

```bash
./scripts/run_adaptdiffuser_adapt_tests.sh
```

Run tests on GPU (if available):

```bash
./scripts/run_adaptdiffuser_adapt_tests.sh --device gpu
```

Run tests on both CPU and GPU:

```bash
./scripts/run_adaptdiffuser_adapt_tests.sh --device both
```

Run a specific test suite only:

```bash
./scripts/run_adaptdiffuser_adapt_tests.sh --test learning_rates
./scripts/run_adaptdiffuser_adapt_tests.sh --test batch_sizes
./scripts/run_adaptdiffuser_adapt_tests.sh --test quality_thresholds
./scripts/run_adaptdiffuser_adapt_tests.sh --test error_handling
```

## Test Components

### Learning Rate Tests

Tests adaptation with different learning rates:
- 1e-5 (very slow learning)
- 1e-4 (default learning rate)
- 1e-3 (fast learning)
- 1e-2 (very fast learning)

This allows you to understand how learning rate affects the adaptation process speed and quality.

### Batch Size Tests

Tests adaptation with different batch sizes:
- 1 (single example per batch)
- 2 (small batch)
- 4 (medium batch)
- 8 (large batch)

This helps evaluate computational efficiency and the effect of batch size on adaptation quality.

### Quality Threshold Tests

Tests adaptation with different quality thresholds for sample filtering:
- 0.3 (low threshold - more samples accepted)
- 0.5 (medium threshold)
- 0.7 (high threshold)
- 0.9 (very high threshold - very selective)

This helps determine how strict filtering affects the adaptation process and results.

### Error Handling Tests

Tests the system's ability to handle error conditions:
- Invalid task names
- Missing configuration files
- Invalid parameters (like negative learning rates)

This ensures the system gracefully handles errors and provides helpful error messages.

## Test Results

All test results are saved in the `test_results/adaptdiffuser_adapt` directory:

- `logs/` directory: Contains detailed logs for each test run
- Individual test results in JSON format
- `summary_report.json`: Overview of all test results

The summary report includes:
- Execution time for each test
- Return codes
- Extracted metrics
- Comparison of results across parameter values

## Analyzing Results

### Learning Rate Analysis

When analyzing learning rate results, look for:
- Faster convergence with higher learning rates
- Potentially lower final quality with very high learning rates
- Stability issues that might appear with high learning rates

### Batch Size Analysis

When analyzing batch size results, look for:
- Trade-off between speed and quality
- Memory usage differences
- Potential batch size limitations for your hardware

### Quality Threshold Analysis

When analyzing quality threshold results, look for:
- Number of samples that pass the threshold at different levels
- Quality difference between samples at different thresholds
- Effect on convergence speed and final results

## Extending the Tests

To add new parameter tests:

1. Add a new test function in `test_adaptdiffuser_adapt_cli.py`
2. Update the argument parser to include your new test
3. Update the `run_adaptdiffuser_adapt_tests.sh` script if needed