#!/usr/bin/env python3
"""
Test script for AdaptDiffuser CLI.

This script runs the AdaptDiffuser CLI with various arguments to test
both CPU and GPU modes.
"""

import subprocess
import os
import time
import sys
import json
from pathlib import Path
import argparse

def run_command(cmd, description=None):
    """Run a command and print its output."""
    if description:
        print(f"\n===== {description} =====")
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    print(f"Command completed in {elapsed:.2f}s with return code {result.returncode}")
    
    if result.stdout:
        print("\nStandard Output:")
        print(result.stdout)
    
    if result.stderr:
        print("\nStandard Error:")
        print(result.stderr)
    
    return result


def create_output_dir():
    """Create output directory for test results."""
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def test_adaptdiffuser_cli(device="cpu", test_name=None):
    """Test the AdaptDiffuser CLI with various arguments."""
    output_dir = create_output_dir()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Set up the device-specific config file
    config_file = f"config/adaptdiffuser_{device}.yaml"
    
    # Test help command
    run_command(
        ["python", "-m", "agentic_diffusion", "--config", config_file, "adaptdiffuser", "--help"],
        f"Testing adaptdiffuser help command on {device}"
    )
    
    # Test generate command
    output_file = output_dir / f"adaptdiffuser_generate_{device}_{timestamp}.json"
    run_command(
        ["python", "-m", "agentic_diffusion", "--config", config_file, "adaptdiffuser", "generate",
         "sample_task", "--batch-size", "2", "--guidance-scale", "1.5",
         "--output", str(output_file)],
        f"Testing adaptdiffuser generate command on {device}"
    )
    
    # Test adapt command
    run_command(
        ["python", "-m", "agentic_diffusion", "--config", config_file, "adaptdiffuser", "adapt",
         "sample_adaptation_task", "--iterations", "2", "--batch-size", "4",
         "--learning-rate", "0.0001", "--quality-threshold", "0.6"],
        f"Testing adaptdiffuser adapt command on {device}"
    )
    
    # Test improve command
    run_command(
        ["python", "-m", "agentic_diffusion", "--config", config_file, "adaptdiffuser", "improve",
         "sample_improvement_task", "--iterations", "2", "--trajectories", "10",
         "--quality-threshold", "0.6"],
        f"Testing adaptdiffuser improve command on {device}"
    )


def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description="Test the AdaptDiffuser CLI")
    parser.add_argument("--device", choices=["cpu", "gpu", "both"], default="cpu",
                       help="Device to test on (cpu, gpu, or both)")
    parser.add_argument("--test-name", help="Optional name for the test")
    
    args = parser.parse_args()
    
    # Check if torch is available
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it to run this test.")
        sys.exit(1)
    
    if args.device == "both":
        # Test both CPU and GPU
        print("\n\n===== TESTING ON CPU =====\n")
        test_adaptdiffuser_cli("cpu", args.test_name)
        
        if not torch.cuda.is_available():
            print("\nGPU not available, skipping GPU tests")
        else:
            print("\n\n===== TESTING ON GPU =====\n")
            test_adaptdiffuser_cli("gpu", args.test_name)
    else:
        # Test just the specified device
        if args.device == "gpu" and not torch.cuda.is_available():
            print("\nWARNING: GPU requested but not available, falling back to CPU")
            test_adaptdiffuser_cli("cpu", args.test_name)
        else:
            test_adaptdiffuser_cli(args.device, args.test_name)


if __name__ == "__main__":
    # Make the main function handle torch imports
    main()