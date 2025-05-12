#!/usr/bin/env python3
"""
Test script for AdaptDiffuser generate CLI command.

This script tests the AdaptDiffuser generate command with various options:
- Different batch sizes
- Different guidance scales
- Different output configurations

It verifies the output format and checks that trajectories are properly saved.
"""

import subprocess
import os
import time
import sys
import json
from pathlib import Path
import argparse
import shutil
import re

def run_command(cmd, description=None, show_output=True):
    """Run a command and print its output."""
    if description:
        print(f"\n===== {description} =====")
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    print(f"Command completed in {elapsed:.2f}s with return code {result.returncode}")
    
    if show_output:
        if result.stdout:
            print("\nStandard Output:")
            print(result.stdout)
        
        if result.stderr:
            print("\nStandard Error:")
            print(result.stderr)
    
    return result


def verify_json_output(output_file, batch_size, guidance_scale=None):
    """Verify the JSON output file has the correct format and content."""
    try:
        # Check if file exists
        if not os.path.exists(output_file):
            print(f"ERROR: Output file {output_file} does not exist")
            return False
        
        # Load the JSON file
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Check structure and content
        if not isinstance(data, dict):
            print(f"ERROR: Output file {output_file} is not a valid JSON object")
            return False
        
        # Check metadata
        if 'metadata' not in data:
            print(f"ERROR: Output file {output_file} does not contain metadata")
            return False
        
        metadata = data['metadata']
        if metadata.get('batch_size') != batch_size:
            print(f"WARNING: Batch size in metadata ({metadata.get('batch_size')}) does not match requested batch size ({batch_size})")
        
        if guidance_scale is not None and metadata.get('guidance_scale') != guidance_scale:
            print(f"WARNING: Guidance scale in metadata ({metadata.get('guidance_scale')}) does not match requested guidance scale ({guidance_scale})")
        
        # Check if trajectories are present
        if 'trajectories' not in data:
            print(f"ERROR: Output file {output_file} does not contain trajectories")
            return False
        
        trajectories = data['trajectories']
        if not isinstance(trajectories, list):
            print(f"ERROR: Trajectories in {output_file} is not a list")
            return False
        
        if len(trajectories) != batch_size:
            print(f"WARNING: Number of trajectories ({len(trajectories)}) does not match batch size ({batch_size})")
        
        print(f"Successfully verified JSON output in {output_file}")
        return True
    
    except Exception as e:
        print(f"ERROR verifying output file {output_file}: {str(e)}")
        return False


def create_output_dir():
    """Create output directory for test results."""
    output_dir = Path("test_results/adaptdiffuser_generate")
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def test_batch_sizes(config_file, device, output_dir):
    """Test generate command with different batch sizes."""
    batch_sizes = [1, 2, 4, 8]
    task = "sample_navigational_task"
    
    print(f"\n\n===== TESTING DIFFERENT BATCH SIZES ON {device.upper()} =====\n")
    
    for batch_size in batch_sizes:
        output_file = output_dir / f"batch_size_{batch_size}_{device}.json"
        
        cmd = [
            "python", "-m", "agentic_diffusion",
            "--config", config_file,
            "adaptdiffuser", "generate", task,
            "--batch-size", str(batch_size),
            "--output", str(output_file)
        ]
        
        result = run_command(
            cmd,
            f"Testing batch size {batch_size} on {device}"
        )
        
        # Verify output file exists and has correct format
        if result.returncode == 0:
            verify_json_output(output_file, batch_size)


def test_guidance_scales(config_file, device, output_dir):
    """Test generate command with different guidance scales."""
    guidance_scales = [0.0, 1.0, 3.0, 5.0]
    task = "sample_guidance_task"
    batch_size = 2
    
    print(f"\n\n===== TESTING DIFFERENT GUIDANCE SCALES ON {device.upper()} =====\n")
    
    for guidance_scale in guidance_scales:
        output_file = output_dir / f"guidance_scale_{guidance_scale}_{device}.json"
        
        cmd = [
            "python", "-m", "agentic_diffusion",
            "--config", config_file,
            "adaptdiffuser", "generate", task,
            "--batch-size", str(batch_size),
            "--guidance-scale", str(guidance_scale),
            "--output", str(output_file)
        ]
        
        result = run_command(
            cmd,
            f"Testing guidance scale {guidance_scale} on {device}"
        )
        
        # Verify output file exists and has correct format
        if result.returncode == 0:
            verify_json_output(output_file, batch_size, guidance_scale)


def test_different_tasks(config_file, device, output_dir):
    """Test generate command with different tasks."""
    tasks = [
        "navigate_maze",
        "avoid_obstacles",
        "reach_target",
        "custom_task with spaces"
    ]
    batch_size = 2
    guidance_scale = 2.0
    
    print(f"\n\n===== TESTING DIFFERENT TASKS ON {device.upper()} =====\n")
    
    for task in tasks:
        # Create safe filename from task name
        task_filename = re.sub(r'[^\w]', '_', task)
        output_file = output_dir / f"task_{task_filename}_{device}.json"
        
        cmd = [
            "python", "-m", "agentic_diffusion",
            "--config", config_file,
            "adaptdiffuser", "generate", task,
            "--batch-size", str(batch_size),
            "--guidance-scale", str(guidance_scale),
            "--output", str(output_file)
        ]
        
        result = run_command(
            cmd,
            f"Testing task '{task}' on {device}"
        )
        
        # Verify output file exists and has correct format
        if result.returncode == 0:
            verify_json_output(output_file, batch_size, guidance_scale)


def test_output_format(config_file, device, output_dir):
    """Test different output options."""
    task = "sample_output_format_task"
    batch_size = 2
    guidance_scale = 2.0
    
    print(f"\n\n===== TESTING OUTPUT FORMAT ON {device.upper()} =====\n")
    
    # Test with different output paths
    test_cases = [
        ("Default JSON", output_dir / f"output_default_{device}.json"),
        ("Nested directory", output_dir / f"nested/output_{device}.json"),
        ("With timestamp", output_dir / f"output_{time.strftime('%Y%m%d-%H%M%S')}_{device}.json")
    ]
    
    for description, output_file in test_cases:
        # Ensure parent directory exists
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        cmd = [
            "python", "-m", "agentic_diffusion",
            "--config", config_file,
            "adaptdiffuser", "generate", task,
            "--batch-size", str(batch_size),
            "--guidance-scale", str(guidance_scale),
            "--output", str(output_file)
        ]
        
        result = run_command(
            cmd,
            f"Testing {description} on {device}"
        )
        
        # Verify output file exists and has correct format
        if result.returncode == 0:
            verify_json_output(output_file, batch_size, guidance_scale)


def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description="Test the AdaptDiffuser generate CLI command")
    parser.add_argument("--device", choices=["cpu", "gpu", "both"], default="cpu",
                       help="Device to test on (cpu, gpu, or both)")
    parser.add_argument("--test", choices=["batch", "guidance", "tasks", "output", "all"], default="all",
                       help="Which test to run (default: all)")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up test output directory before running tests")
    
    args = parser.parse_args()
    
    # Check if torch is available
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it to run this test.")
        sys.exit(1)
    
    # Create output directory
    output_dir = create_output_dir()
    
    if args.cleanup and output_dir.exists():
        print(f"Cleaning up test output directory: {output_dir}")
        shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define device configurations to test
    devices_to_test = []
    if args.device == "both":
        devices_to_test.append(("cpu", "config/adaptdiffuser_cpu.yaml"))
        if torch.cuda.is_available():
            devices_to_test.append(("gpu", "config/adaptdiffuser_gpu.yaml"))
        else:
            print("\nGPU not available, skipping GPU tests")
    else:
        if args.device == "gpu" and not torch.cuda.is_available():
            print("\nWARNING: GPU requested but not available, falling back to CPU")
            devices_to_test.append(("cpu", "config/adaptdiffuser_cpu.yaml"))
        else:
            config_file = f"config/adaptdiffuser_{args.device}.yaml"
            devices_to_test.append((args.device, config_file))
    
    # Run tests for each device
    for device, config_file in devices_to_test:
        if args.test == "all" or args.test == "batch":
            test_batch_sizes(config_file, device, output_dir)
        
        if args.test == "all" or args.test == "guidance":
            test_guidance_scales(config_file, device, output_dir)
        
        if args.test == "all" or args.test == "tasks":
            test_different_tasks(config_file, device, output_dir)
        
        if args.test == "all" or args.test == "output":
            test_output_format(config_file, device, output_dir)
    
    print("\n\n===== ALL TESTS COMPLETED =====\n")
    print(f"Test results can be found in: {output_dir}")


if __name__ == "__main__":
    main()