# AdaptDiffuser CLI Testing

This directory contains scripts and configuration files for testing the AdaptDiffuser CLI functionality with various arguments and options, including both CPU and GPU configurations.

## Overview

The testing script `test_adaptdiffuser_cli.sh` runs a series of tests on the AdaptDiffuser CLI tool, exercising different subcommands, options, and device configurations (CPU and GPU).

## Usage

To run all tests:

```bash
./scripts/test_adaptdiffuser_cli.sh
```

This will:
1. Run CPU tests first (generate, adapt, improve)
2. Check if a GPU is available
3. If GPU is available, run GPU tests
4. Test error handling with an invalid command

## Configuration Files

### CPU Configuration (`config/test_cpu.yaml`)

The CPU configuration is optimized for testing on CPU environments with:
- Reduced model size
- Fewer timesteps
- Smaller batch sizes
- DDIM sampling for faster execution

### GPU Configuration (`config/test_gpu.yaml`)

The GPU configuration is optimized for testing on GPU environments with:
- Larger model size
- More timesteps
- Larger batch sizes
- Mixed precision (float16)
- Memory optimization settings

## CLI Command Structure

The AdaptDiffuser CLI follows this command structure:

```
python -m agentic_diffusion [global options] command subcommand [command options]
```

For example:
```
python -m agentic_diffusion --config ./config/test_cpu.yaml adaptdiffuser generate "task_description" --batch-size 4
```

### Global Options

- `--config PATH`: Path to the configuration file

### AdaptDiffuser Subcommands

1. **generate**: Generate trajectories using AdaptDiffuser
   - Arguments:
     - `task`: Task description or identifier
   - Options:
     - `--batch-size, -b`: Number of trajectories to generate
     - `--guidance-scale, -g`: Scale for reward guidance
     - `--output, -o`: Output file to save trajectories

2. **adapt**: Adapt AdaptDiffuser to a specific task
   - Arguments:
     - `task`: Task description or identifier
   - Options:
     - `--examples, -e`: Path to examples file (JSON format)
     - `--iterations, -i`: Number of adaptation iterations
     - `--batch-size, -b`: Batch size for adaptation
     - `--learning-rate, -lr`: Learning rate for adaptation
     - `--quality-threshold, -q`: Quality threshold for filtering samples
     - `--save-checkpoint, -s`: Save adaptation checkpoint

3. **improve**: Self-improve AdaptDiffuser on a task
   - Arguments:
     - `task`: Task description or identifier
   - Options:
     - `--iterations, -i`: Number of improvement iterations
     - `--trajectories, -t`: Trajectories per iteration
     - `--quality-threshold, -q`: Quality threshold for filtering samples

## Device Configuration

### CPU Testing

CPU testing uses simplified models and reduced steps to ensure tests run efficiently:

```yaml
device: cpu
precision: float32
batch_size: 4
# Smaller model size, fewer steps
```

### GPU Testing

GPU testing leverages the full capabilities of the GPU:

```yaml
device: cuda
precision: float16  # Use half-precision
batch_size: 16
# Larger model size, more steps, additional optimizations
```

## Troubleshooting

If you encounter issues:

1. **Config path errors**: Make sure the `--config` option comes before the command
2. **CUDA errors**: Check if CUDA is available using `torch.cuda.is_available()`
3. **Memory errors**: Reduce batch size or model size in config files
4. **Import errors**: Ensure all dependencies are installed

## Adding New Tests

To add new tests:
1. Add the test command to `test_adaptdiffuser_cli.sh`
2. Update config files if necessary
3. Run the script to execute all tests