#!/usr/bin/env python3
"""
Test script for AdaptDiffuser 'adapt' CLI command with various parameters.

Tests:
1. Different learning rates
2. Different batch sizes 
3. Different quality thresholds
4. Error handling (non-existent model)

Usage:
    python test_adaptdiffuser_cli_parameters.py [--device cpu|gpu|both] [--test all|learning_rates|batch_sizes|quality_thresholds|error_handling]
"""

import argparse
import json
import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("adaptdiffuser_cli_test")

# Set up test output directories
OUTPUT_DIR = Path("test_results/adaptdiffuser_cli_parameters")
LOG_DIR = OUTPUT_DIR / "logs"


def run_command(command, capture_output=False):
    """Run a shell command and return the result."""
    try:
        print(f"Running command: {' '.join(command)}")
        start_time = datetime.now()
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=False,
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Command completed in {elapsed:.2f}s with return code {result.returncode}")
        
        if result.stderr and len(result.stderr.strip()) > 0:
            print("\nStandard Error:")
            print(result.stderr)
            
        return result
    except Exception as e:
        print(f"Error executing command: {e}")
        return None


def extract_metrics_from_output(output):
    """Extract metrics from command output."""
    metrics = {}
    
    # Check if output is None
    if output is None:
        return {"error": "No output captured"}
    
    # Look for adaptation metrics
    if "Adaptation metrics:" in output:
        try:
            metrics_section = output.split("Adaptation metrics:")[1].split("\n\n")[0]
            for line in metrics_section.strip().split("\n"):
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    key = key.strip().strip("  ")
                    value = value.strip()
                    try:
                        metrics[key] = float(value.replace("(final)", "").strip())
                    except ValueError:
                        metrics[key] = value
        except Exception as e:
            metrics["parsing_error"] = str(e)
    
    # Extract training metrics if present
    reward_match = None
    try:
        for line in output.split("\n"):
            if "Final reward:" in line:
                try:
                    reward_match = line.split("Final reward:")[1].strip()
                    metrics["final_reward"] = float(reward_match)
                except (ValueError, IndexError):
                    pass
                    
            if "Improvement:" in line and "%" in line:
                try:
                    improvement = line.split("Improvement:")[1].strip()
                    metrics["improvement"] = float(improvement.replace("%", ""))
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        metrics["parsing_error"] = str(e)
    
    return metrics


def save_test_results(test_name, command, result, metrics, output_dir):
    """Save test results to a file."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = output_dir / f"{test_name}_{timestamp}.json"
    
    results = {
        "test_name": test_name,
        "command": " ".join(map(str, command)),
        "return_code": result.returncode if result else None,
        "execution_time": None,  # Would need to capture this during execution
        "metrics": metrics,
        "stdout_excerpt": result.stdout[:500] + "..." if result and hasattr(result, "stdout") and result.stdout and len(result.stdout) > 500 else (result.stdout if result and hasattr(result, "stdout") else None),
        "stderr_excerpt": result.stderr[:500] + "..." if result and hasattr(result, "stderr") and result.stderr and len(result.stderr) > 500 else (result.stderr if result and hasattr(result, "stderr") else None),
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Test results saved to {results_file}")
    return results_file


def create_output_dirs():
    """Create output directories."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Created output directories: {OUTPUT_DIR}, {LOG_DIR}")


def test_learning_rates(device="cpu"):
    """Test different learning rates."""
    task = "learning_rate_test_task"
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
    
    print(f"\n{'='*20} TESTING LEARNING RATES {'='*20}")
    
    results = {}
    for lr in learning_rates:
        test_name = f"learning_rate_{lr}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        print(f"Testing learning rate {lr} on {device}, logging to {log_file}")
        
        # Command with proper format for adaptdiffuser adapt
        # Command with proper format for adaptdiffuser adapt
        command = [
            "python", "-m", "agentic_diffusion",
            "adaptdiffuser", "adapt",  # Use adaptdiffuser adapt command
            f"{task}_lr_{lr:.1e}",  # Make each task unique
            "--learning-rate", str(lr)  # Specify learning rate
        ]
        # Run once with output capture for analysis and logging
        result = run_command(command, capture_output=True)
        
        # Write to log file
        with open(log_file, "w") as log:
            if result and hasattr(result, 'stdout'):
                log.write(result.stdout)
            if result and hasattr(result, 'stderr'):
                log.write("\nERROR:\n" + result.stderr)
        if result:
            metrics = {}
            if hasattr(result, 'stdout'):
                metrics = extract_metrics_from_output(result.stdout)
                
            results[str(lr)] = {
                "return_code": result.returncode,
                "metrics": metrics
            }
            
            # Save results
            save_test_results(test_name, command, result, metrics, OUTPUT_DIR)
            
        print(f"\nLearning rate {lr} test completed with return code {result.returncode if result else 'unknown'}")
    
    return results


def test_batch_sizes(device="cpu"):
    """Test different batch sizes."""
    task = "batch_size_test_task"
    batch_sizes = [1, 2, 4, 8]
    
    print(f"\n{'='*20} TESTING BATCH SIZES {'='*20}")
    
    results = {}
    for batch_size in batch_sizes:
        test_name = f"batch_size_{batch_size}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        print(f"Testing batch size {batch_size} on {device}, logging to {log_file}")
        
        # Command with proper format for adaptdiffuser adapt
        command = [
            "python", "-m", "agentic_diffusion",
            "adaptdiffuser", "adapt",  # Use adaptdiffuser adapt command
            f"{task}_batch_{batch_size}",  # Make each task unique
            "--batch-size", str(batch_size)  # Specify batch size
        ]
        
        # Redirect output to log file
        # Run once with output capture for analysis and logging
        result = run_command(command, capture_output=True)
        
        # Write to log file
        with open(log_file, "w") as log:
            if result and hasattr(result, 'stdout'):
                log.write(result.stdout)
            if result and hasattr(result, 'stderr'):
                log.write("\nERROR:\n" + result.stderr)
            metrics = {}
            if hasattr(result, 'stdout'):
                metrics = extract_metrics_from_output(result.stdout)
                
            results[str(batch_size)] = {
                "return_code": result.returncode,
                "metrics": metrics
            }
            
            # Save results
            save_test_results(test_name, command, result, metrics, OUTPUT_DIR)
            
        print(f"\nBatch size {batch_size} test completed with return code {result.returncode if result else 'unknown'}")
    
    return results


def test_quality_thresholds(device="cpu"):
    """Test different quality thresholds."""
    task = "quality_threshold_test_task"
    thresholds = [0.5, 0.6, 0.7, 0.8]
    
    print(f"\n{'='*20} TESTING QUALITY THRESHOLDS {'='*20}")
    
    results = {}
    for threshold in thresholds:
        test_name = f"quality_threshold_{threshold}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        print(f"Testing quality threshold {threshold} on {device}, logging to {log_file}")
        
        # Command with proper format for adaptdiffuser adapt
        command = [
            "python", "-m", "agentic_diffusion",
            "adaptdiffuser", "adapt",  # Use adaptdiffuser adapt command
            f"{task}_threshold_{threshold:.1f}",  # Make each task unique
            "--quality-threshold", str(threshold)  # Specify quality threshold
        ]
        
        # Redirect output to log file
        # Run once with output capture for analysis and logging
        result = run_command(command, capture_output=True)
        
        # Write to log file
        with open(log_file, "w") as log:
            if result and hasattr(result, 'stdout'):
                log.write(result.stdout)
            if result and hasattr(result, 'stderr'):
                log.write("\nERROR:\n" + result.stderr)
            metrics = {}
            if hasattr(result, 'stdout'):
                metrics = extract_metrics_from_output(result.stdout)
                
            results[str(threshold)] = {
                "return_code": result.returncode,
                "metrics": metrics
            }
            
            # Save results
            save_test_results(test_name, command, result, metrics, OUTPUT_DIR)
            
        print(f"\nQuality threshold {threshold} test completed with return code {result.returncode if result else 'unknown'}")
    
    return results


def test_error_handling(device="cpu"):
    """Test error handling (missing reward model)."""
    print(f"\n{'='*20} TESTING ERROR HANDLING {'='*20}")
    
    # Test with a non-existent examples file
    test_name = f"nonexistent_examples_file_{device}"
    log_file = LOG_DIR / f"{test_name}.log"
    
    print(f"Testing error handling (non-existent examples file) on {device}, logging to {log_file}")
    
    # Command with non-existent reward model file
    command = [
        "python", "-m", "agentic_diffusion",
        "adaptdiffuser", "adapt",  # Use adaptdiffuser adapt command
        "error_handling_test",
        "--examples", "nonexistent_examples_file.json"  # This should cause an error
    ]
    
    # Redirect output to log file
    # Run once with output capture for analysis and logging
    result = run_command(command, capture_output=True)
    
    # Write to log file
    with open(log_file, "w") as log:
        if result and hasattr(result, 'stdout'):
            log.write(result.stdout)
        if result and hasattr(result, 'stderr'):
            log.write("\nERROR:\n" + result.stderr)
    if result and hasattr(result, 'stdout'):
        metrics = extract_metrics_from_output(result.stdout)
            
    # Save results
    save_test_results(test_name, command, result, metrics, OUTPUT_DIR)
        
    print(f"\nError handling test completed with return code {result.returncode if result else 'unknown'}")
    
    # Expect the command to fail or show warning
    success = (result.returncode != 0) or ('error' in result.stderr.lower()) if result and hasattr(result, 'stderr') else False
    
    if success:
        print("✅ Error handling test passed - command failed gracefully or showed error.")
    else:
        print("❌ Error handling test might have issues - command did not fail or show errors.")
    
    return {"nonexistent_examples_file": {"return_code": result.returncode if result else None}}


def generate_summary_report(learning_rate_results, batch_size_results, quality_threshold_results, error_handling_results):
    """Generate a summary report of all tests."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_file = OUTPUT_DIR / f"summary_report_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "learning_rate_tests": learning_rate_results,
        "batch_size_tests": batch_size_results,
        "quality_threshold_tests": quality_threshold_results,
        "error_handling_tests": error_handling_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Summary report saved to {summary_file}")
    return summary_file


def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description="Test the AdaptDiffuser adapt CLI command with different parameters")
    parser.add_argument("--device", choices=["cpu", "gpu", "both"], default="cpu",
                       help="Device to test on (cpu, gpu, or both)")
    parser.add_argument("--test", choices=["all", "learning_rates", "batch_sizes", "quality_thresholds", "error_handling"],
                       default="all", help="Which test suite to run")
    
    args = parser.parse_args()
    
    # Check if torch is available
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it to run this test.")
        sys.exit(1)
    
    # Create output directories
    create_output_dirs()
    
    # Run tests based on device
    devices = []
    if args.device == "both":
        devices.append("cpu")
        if torch.cuda.is_available():
            devices.append("gpu")
        else:
            print("WARNING: GPU requested but not available")
    else:
        if args.device == "gpu" and not torch.cuda.is_available():
            print("WARNING: GPU requested but not available, falling back to CPU")
            devices.append("cpu")
        else:
            devices.append(args.device)
    
    for device in devices:
        print(f"\n\n{'='*30} TESTING ON {device.upper()} {'='*30}\n")
        
        learning_rate_results = {}
        batch_size_results = {}
        quality_threshold_results = {}
        error_handling_results = {}
        
        # Run selected tests
        if args.test == "all" or args.test == "learning_rates":
            learning_rate_results = test_learning_rates(device)
            
        if args.test == "all" or args.test == "batch_sizes":
            batch_size_results = test_batch_sizes(device)
            
        if args.test == "all" or args.test == "quality_thresholds":
            quality_threshold_results = test_quality_thresholds(device)
            
        if args.test == "all" or args.test == "error_handling":
            error_handling_results = test_error_handling(device)
            
        # Generate summary report for this device
        if args.test == "all":
            generate_summary_report(
                learning_rate_results, 
                batch_size_results, 
                quality_threshold_results,
                error_handling_results
            )
    
    print("\n\nAll tests completed!")
    print(f"Test results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()