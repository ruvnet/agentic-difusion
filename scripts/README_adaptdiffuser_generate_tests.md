# AdaptDiffuser Generate Command Testing

This directory contains scripts for testing the AdaptDiffuser 'generate' CLI command with various options. The scripts test different combinations of batch sizes, guidance scales, and output configurations to ensure the command works correctly and trajectories are properly saved.

## Overview

Three primary scripts are provided:

1. `test_adaptdiffuser_generate.py` - The main test script that runs various tests with different parameters
2. `run_adaptdiffuser_generate_tests.sh` - A bash script that makes it easy to run all tests or specific categories
3. `adaptdiffuser_generate_example.py` - A simple example script demonstrating basic usage of the generate command

## Quick Start

To run a simple example of the generate command:

```bash
./scripts/adaptdiffuser_generate_example.py
```

This will run the generate command with default parameters and display a summary of the output.

## Running All Tests

To run the complete test suite:

```bash
./scripts/run_adaptdiffuser_generate_tests.sh
```

This will:
1. Create a virtual environment if needed and install dependencies
2. Run all tests on CPU
3. If CUDA is available, also run tests on GPU
4. Save all test results to the `test_results/adaptdiffuser_generate` directory

## Running Specific Tests

You can run specific test categories using the `--test` option:

```bash
# Test different batch sizes
python scripts/test_adaptdiffuser_generate.py --device cpu --test batch

# Test different guidance scales
python scripts/test_adaptdiffuser_generate.py --device cpu --test guidance

# Test different tasks
python scripts/test_adaptdiffuser_generate.py --device cpu --test tasks

# Test output format configurations
python scripts/test_adaptdiffuser_generate.py --device cpu --test output
```

## Test Categories

The test script includes several test categories:

1. **Batch Size Tests**
   - Tests batch sizes: 1, 2, 4, 8
   - Verifies that the number of generated trajectories matches the batch size

2. **Guidance Scale Tests**
   - Tests guidance scales: 0.0, 1.0, 3.0, 5.0
   - Verifies that the guidance scale affects the generation and is correctly reported in metadata

3. **Task Tests**
   - Tests different task descriptions including with spaces and special characters
   - Verifies that task information is correctly processed and included in output

4. **Output Format Tests**
   - Tests different output file paths including nested directories
   - Verifies that output files are correctly created and contain valid data

## Output Verification

Each test verifies:
- Command execution success (return code 0)
- Output file creation
- Output file format (valid JSON)
- Metadata correctness (batch size, guidance scale, task)
- Trajectory generation (correct number and format)

## Configuration

The tests use two different configuration files:
- `config/adaptdiffuser_cpu.yaml` - Configuration optimized for CPU execution
- `config/adaptdiffuser_gpu.yaml` - Configuration optimized for GPU execution (if available)

## Troubleshooting

If you encounter issues:

1. **Missing configuration files**: Ensure that `config/adaptdiffuser_cpu.yaml` and `config/adaptdiffuser_gpu.yaml` exist
2. **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
3. **CUDA errors**: If you get CUDA errors when running GPU tests, try using the CPU tests only
4. **Permission errors**: Ensure the scripts are executable with `chmod +x scripts/*.py scripts/*.sh`
5. **Output directory issues**: Make sure the `test_results` directory is writable

## Adding New Tests

To add new test cases:
1. Modify `test_adaptdiffuser_generate.py` to add new test functions
2. Update `run_adaptdiffuser_generate_tests.sh` to include options for running your new tests
3. Run the tests to verify they work as expected