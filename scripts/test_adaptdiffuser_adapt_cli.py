#!/usr/bin/env python3
"""
Test script for AdaptDiffuser 'adapt' CLI command with various parameters.

This script tests the 'adapt' command with different configurations:
- Learning rates
- Batch sizes
- Quality thresholds
- Error handling (missing reward models)
- Metrics reporting

Usage:
    python scripts/test_adaptdiffuser_adapt_cli.py
"""

import subprocess
import os
import time
import sys
import json
from pathlib import Path
import argparse
import importlib
import re
from datetime import datetime

# Configure output directories - can be overridden via environment variable
OUTPUT_DIR = Path(os.environ.get("ADAPTDIFFUSER_TEST_OUTPUT_DIR", "test_results/adaptdiffuser_adapt"))
LOG_DIR = OUTPUT_DIR / "logs"


def run_command(cmd, description=None, capture_output=True):
    """Run a command and print its output."""
    if description:
        print(f"\n{'='*10} {description} {'='*10}")
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - start_time
            
            print(f"Command completed in {elapsed:.2f}s with return code {result.returncode}")
            
            if result.stdout:
                print("\nStandard Output (excerpt):")
                # Print the first and last 10 lines of output
                lines = result.stdout.splitlines()
                if len(lines) > 20:
                    print("\n".join(lines[:10]))
                    print("...")
                    print("\n".join(lines[-10:]))
                else:
                    print(result.stdout)
            
            if result.stderr:
                print("\nStandard Error:")
                print(result.stderr)
                
            return result
        else:
            # Run without capturing output (streams directly to console)
            result = subprocess.run(cmd)
            elapsed = time.time() - start_time
            print(f"Command completed in {elapsed:.2f}s with return code {result.returncode}")
            return result
            
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        if e.output:
            print(f"Output: {e.output}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return e
    except Exception as e:
        print(f"Failed to run command: {e}")
        return None


def extract_metrics_from_output(output):
    """Extract metrics from command output."""
    metrics = {}
    
    # Extract adaptation loss
    loss_match = re.search(r"Adaptation loss: ([0-9.]+)", output)
    if loss_match:
        metrics["adaptation_loss"] = float(loss_match.group(1))
        
    # Extract quality scores
    quality_match = re.search(r"Average quality score: ([0-9.]+)", output)
    if quality_match:
        metrics["avg_quality_score"] = float(quality_match.group(1))
        
    # Extract number of high-quality samples
    samples_match = re.search(r"Generated (\d+) synthetic samples, filtered (\d+) high-quality", output)
    if samples_match:
        metrics["total_samples"] = int(samples_match.group(1))
        metrics["high_quality_samples"] = int(samples_match.group(2))
    
    # Extract training metrics if present
    acc_match = re.search(r"Accuracy: ([0-9.]+)", output)
    if acc_match:
        metrics["accuracy"] = float(acc_match.group(1))
    
    return metrics


def save_test_results(test_name, command, result, metrics, output_dir):
    """Save test results to a file."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = output_dir / f"{test_name}_{timestamp}.json"
    
    results = {
        "test_name": test_name,
        "command": " ".join(command),
        "return_code": result.returncode if result else None,
        "execution_time": None,  # Would need to capture this during execution
        "metrics": metrics,
        "stdout_excerpt": result.stdout[:500] + "..." if result and result.stdout and len(result.stdout) > 500 else (result.stdout if result else None),
        "stderr_excerpt": result.stderr[:500] + "..." if result and result.stderr and len(result.stderr) > 500 else (result.stderr if result else None),
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
    config_file = f"config/adaptdiffuser_{device}.yaml"
    task = "learning_rate_test_task"
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
    
    print(f"\n{'='*20} TESTING LEARNING RATES {'='*20}")
    
    results = {}
    for lr in learning_rates:
        test_name = f"learning_rate_{lr}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        cmd = [
            "python", "-m", "agentic_diffusion", 
            "--config", config_file, 
            "adaptdiffuser", "adapt", task,
            "--iterations", "3",
            "--batch-size", "4", 
            "--learning-rate", str(lr),
            "--quality-threshold", "0.6"
        ]
        
        # Run with output captured to both console and log file
        with open(log_file, 'w') as log:
            print(f"Testing learning rate {lr} on {device}, logging to {log_file}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log.write(result.stdout)
            
            metrics = extract_metrics_from_output(result.stdout)
            results[str(lr)] = {
                "return_code": result.returncode,
                "metrics": metrics
            }
            
            output_summary = f"\nLearning rate {lr} test completed with return code {result.returncode}"
            if metrics:
                output_summary += f"\nMetrics: {json.dumps(metrics, indent=2)}"
            print(output_summary)
            log.write(output_summary)
            
    return results


def test_batch_sizes(device="cpu"):
    """Test different batch sizes."""
    config_file = f"config/adaptdiffuser_{device}.yaml"
    task = "batch_size_test_task"
    batch_sizes = [1, 2, 4, 8]
    
    print(f"\n{'='*20} TESTING BATCH SIZES {'='*20}")
    
    results = {}
    for batch_size in batch_sizes:
        test_name = f"batch_size_{batch_size}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        cmd = [
            "python", "-m", "agentic_diffusion", 
            "--config", config_file, 
            "adaptdiffuser", "adapt", task,
            "--iterations", "3",
            "--batch-size", str(batch_size), 
            "--learning-rate", "1e-4",
            "--quality-threshold", "0.6"
        ]
        
        # Run with output captured to both console and log file
        with open(log_file, 'w') as log:
            print(f"Testing batch size {batch_size} on {device}, logging to {log_file}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log.write(result.stdout)
            
            metrics = extract_metrics_from_output(result.stdout)
            results[str(batch_size)] = {
                "return_code": result.returncode,
                "metrics": metrics
            }
            
            output_summary = f"\nBatch size {batch_size} test completed with return code {result.returncode}"
            if metrics:
                output_summary += f"\nMetrics: {json.dumps(metrics, indent=2)}"
            print(output_summary)
            log.write(output_summary)
            
    return results


def test_quality_thresholds(device="cpu"):
    """Test different quality thresholds."""
    config_file = f"config/adaptdiffuser_{device}.yaml"
    task = "quality_threshold_test_task"
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    print(f"\n{'='*20} TESTING QUALITY THRESHOLDS {'='*20}")
    
    results = {}
    for threshold in thresholds:
        test_name = f"quality_threshold_{threshold}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        cmd = [
            "python", "-m", "agentic_diffusion", 
            "--config", config_file, 
            "adaptdiffuser", "adapt", task,
            "--iterations", "3",
            "--batch-size", "4", 
            "--learning-rate", "1e-4",
            "--quality-threshold", str(threshold)
        ]
        
        # Run with output captured to both console and log file
        with open(log_file, 'w') as log:
            print(f"Testing quality threshold {threshold} on {device}, logging to {log_file}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log.write(result.stdout)
            
            metrics = extract_metrics_from_output(result.stdout)
            results[str(threshold)] = {
                "return_code": result.returncode,
                "metrics": metrics
            }
            
            output_summary = f"\nQuality threshold {threshold} test completed with return code {result.returncode}"
            if metrics:
                output_summary += f"\nMetrics: {json.dumps(metrics, indent=2)}"
            print(output_summary)
            log.write(output_summary)
            
    return results


def test_error_handling(device="cpu"):
    """Test error handling for missing components."""
    config_file = f"config/adaptdiffuser_{device}.yaml"
    
    print(f"\n{'='*20} TESTING ERROR HANDLING {'='*20}")
    
    # Test 1: Invalid task name
    test_name = f"error_invalid_task_{device}"
    log_file = LOG_DIR / f"{test_name}.log"
    
    cmd = [
        "python", "-m", "agentic_diffusion", 
        "--config", config_file, 
        "adaptdiffuser", "adapt", "THIS_TASK_DOES_NOT_EXIST",
        "--iterations", "2",
        "--batch-size", "4"
    ]
    
    with open(log_file, 'w') as log:
        print(f"Testing invalid task on {device}, logging to {log_file}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(result.stdout)
        
        output_summary = f"\nInvalid task test completed with return code {result.returncode}"
        print(output_summary)
        log.write(output_summary)
    
    # Test 2: Invalid configuration file
    test_name = f"error_invalid_config_{device}"
    log_file = LOG_DIR / f"{test_name}.log"
    
    cmd = [
        "python", "-m", "agentic_diffusion", 
        "--config", "config/nonexistent_config.yaml", 
        "adaptdiffuser", "adapt", "sample_task",
        "--iterations", "2",
        "--batch-size", "4"
    ]
    
    with open(log_file, 'w') as log:
        print(f"Testing invalid config on {device}, logging to {log_file}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(result.stdout)
        
        output_summary = f"\nInvalid config test completed with return code {result.returncode}"
        print(output_summary)
        log.write(output_summary)
    
    # Test 3: Invalid learning rate (negative)
    test_name = f"error_invalid_learning_rate_{device}"
    log_file = LOG_DIR / f"{test_name}.log"
    
    cmd = [
        "python", "-m", "agentic_diffusion", 
        "--config", config_file, 
        "adaptdiffuser", "adapt", "sample_task",
        "--iterations", "2",
        "--batch-size", "4",
        "--learning-rate", "-1"
    ]
    
    with open(log_file, 'w') as log:
        print(f"Testing invalid learning rate on {device}, logging to {log_file}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(result.stdout)
        
        output_summary = f"\nInvalid learning rate test completed with return code {result.returncode}"
        print(output_summary)
        log.write(output_summary)


def generate_summary_report(learning_rate_results, batch_size_results, quality_threshold_results):
    """Generate a summary report of all test results."""
    summary_file = OUTPUT_DIR / "summary_report.json"
    
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "learning_rate_tests": learning_rate_results,
        "batch_size_tests": batch_size_results,
        "quality_threshold_tests": quality_threshold_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Summary report saved to {summary_file}")
    return summary_file


def main():
    """Main function to run all tests."""
    # Import the mock reward model to patch AdaptDiffuser
    try:
        print("Importing mock reward model...")
        import scripts.mock_adaptdiffuser_reward
        print("Mock reward model imported successfully")
    except Exception as e:
        print(f"WARNING: Failed to import mock reward model: {e}")
        print("Tests will run without mock reward model")
    
    parser = argparse.ArgumentParser(description="Test the AdaptDiffuser adapt CLI command")
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
        
        # Run selected tests
        if args.test == "all" or args.test == "learning_rates":
            learning_rate_results = test_learning_rates(device)
            
        if args.test == "all" or args.test == "batch_sizes":
            batch_size_results = test_batch_sizes(device)
            
        if args.test == "all" or args.test == "quality_thresholds":
            quality_threshold_results = test_quality_thresholds(device)
            
        if args.test == "all" or args.test == "error_handling":
            test_error_handling(device)
            
        # Generate summary report for this device
        if args.test == "all":
            generate_summary_report(learning_rate_results, batch_size_results, quality_threshold_results)
    
    print("\n\nAll tests completed!")
    print(f"Test results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()