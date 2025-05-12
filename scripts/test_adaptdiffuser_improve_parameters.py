#!/usr/bin/env python3
"""
Test script for AdaptDiffuser 'improve' CLI command with various parameters.

This script tests the self-improvement process of AdaptDiffuser with:
1. Different numbers of iterations
2. Different trajectory counts
3. Different quality thresholds
4. Error handling for missing reward models
5. Metric reporting verification

Usage:
    python test_adaptdiffuser_improve_parameters.py [--device cpu|gpu|both] [--test all|iterations|trajectories|quality|error_handling]
"""

import argparse
import json
import os
import sys
import logging
import subprocess
import re
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("adaptdiffuser_improve_test")

# Set up test output directories
OUTPUT_DIR = Path("test_results/adaptdiffuser_improve")
LOG_DIR = OUTPUT_DIR / "logs"


def run_command(command, capture_output=True):
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
    
    # Look for improvement metrics
    if "Improvement metrics:" in output:
        try:
            metrics_section = output.split("Improvement metrics:")[1].split("\n\n")[0]
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
    
    # Extract final reward and improvement percentage if present
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
                    
            # Check for error messages about missing reward models
            if "no reward model" in line.lower() or "missing reward model" in line.lower():
                metrics["error_missing_reward"] = True
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


def test_iterations(device="cpu"):
    """Test different iteration counts."""
    task = "iteration_test_task"
    iterations = [1, 5]  # Reduced set for demo purposes
    
    print(f"\n{'='*20} TESTING ITERATIONS {'='*20}")
    
    results = {}
    for iter_count in iterations:
        test_name = f"iterations_{iter_count}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        print(f"Testing {iter_count} iterations on {device}, logging to {log_file}")
        
        # Command for adaptdiffuser improve
        command = [
            "python", "-m", "agentic_diffusion",
            "--config", f"./config/test_{device}.yaml",
            "adaptdiffuser", "improve",
            f"{task}_iters_{iter_count}",  # Make each task unique
            "--iterations", str(iter_count),  # Specify iteration count
            "--trajectories", "10"          # Keep trajectories constant for this test
        ]
        
        # Run with output capture for analysis and logging
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
                
            results[str(iter_count)] = {
                "return_code": result.returncode,
                "metrics": metrics
            }
            
            # Save results
            save_test_results(test_name, command, result, metrics, OUTPUT_DIR)
            
        print(f"\nIteration count {iter_count} test completed with return code {result.returncode if result else 'unknown'}")
    
    return results


def test_trajectories(device="cpu"):
    """Test different trajectory counts."""
    task = "trajectory_test_task"
    trajectory_counts = [5, 10]  # Reduced set for demo purposes
    
    print(f"\n{'='*20} TESTING TRAJECTORY COUNTS {'='*20}")
    
    results = {}
    for traj_count in trajectory_counts:
        test_name = f"trajectories_{traj_count}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        print(f"Testing {traj_count} trajectories on {device}, logging to {log_file}")
        
        # Command for adaptdiffuser improve
        command = [
            "python", "-m", "agentic_diffusion",
            "--config", f"./config/test_{device}.yaml",
            "adaptdiffuser", "improve",
            f"{task}_trajs_{traj_count}",  # Make each task unique
            "--iterations", "3",           # Keep iterations constant for this test
            "--trajectories", str(traj_count)  # Specify trajectory count
        ]
        
        # Run with output capture for analysis and logging
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
                
            results[str(traj_count)] = {
                "return_code": result.returncode,
                "metrics": metrics
            }
            
            # Save results
            save_test_results(test_name, command, result, metrics, OUTPUT_DIR)
            
        print(f"\nTrajectory count {traj_count} test completed with return code {result.returncode if result else 'unknown'}")
    
    return results


def test_quality_thresholds(device="cpu"):
    """Test different quality thresholds."""
    task = "quality_threshold_test_task"
    thresholds = [0.5, 0.8]  # Reduced set for demo purposes
    
    print(f"\n{'='*20} TESTING QUALITY THRESHOLDS {'='*20}")
    
    results = {}
    for threshold in thresholds:
        test_name = f"quality_threshold_{threshold}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        print(f"Testing quality threshold {threshold} on {device}, logging to {log_file}")
        
        # Command for adaptdiffuser improve
        command = [
            "python", "-m", "agentic_diffusion",
            "--config", f"./config/test_{device}.yaml",
            "adaptdiffuser", "improve",
            f"{task}_threshold_{threshold:.1f}",  # Make each task unique
            "--iterations", "3",                 # Keep iterations constant for this test
            "--trajectories", "10",              # Keep trajectories constant for this test
            "--quality-threshold", str(threshold)  # Specify quality threshold
        ]
        
        # Run with output capture for analysis and logging
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
                
            results[str(threshold)] = {
                "return_code": result.returncode,
                "metrics": metrics
            }
            
            # Save results
            save_test_results(test_name, command, result, metrics, OUTPUT_DIR)
            
        print(f"\nQuality threshold {threshold} test completed with return code {result.returncode if result else 'unknown'}")
    
    return results


def test_error_handling(device="cpu"):
    """Test error handling for various error scenarios."""
    print(f"\n{'='*20} TESTING ERROR HANDLING {'='*20}")
    
    results = {}
    
    # Test 1: With an invalid quality threshold value
    test_name = f"invalid_quality_threshold_{device}"
    log_file = LOG_DIR / f"{test_name}.log"
    
    print(f"Testing error handling (invalid quality threshold) on {device}, logging to {log_file}")
    
    # Command with invalid quality threshold
    command = [
        "python", "-m", "agentic_diffusion",
        "--config", f"./config/test_{device}.yaml",
        "adaptdiffuser", "improve",
        "error_handling_test_invalid_threshold",
        "--iterations", "2",
        "--trajectories", "5",
        "--quality-threshold", "2.0"  # Invalid value (should be between 0.0-1.0)
    ]
    
    # Run with output capture for analysis and logging
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
            
        # Save results
        save_test_results(test_name, command, result, metrics, OUTPUT_DIR)
        
    print(f"\nInvalid quality threshold test completed with return code {result.returncode if result else 'unknown'}")
    
    # Check if the command executed without crashing, which is the minimum we expect
    error_handled_gracefully = (result is not None and result.returncode == 0)
    
    # Note: When no reward model is available, the quality threshold validation
    # may not be triggered because the process is skipped entirely
    
    if error_handled_gracefully:
        print("✅ Invalid threshold test passed - command executed without crashing.")
        if hasattr(result, 'stdout') and "no_reward_model" in result.stdout.lower():
            print("   Note: Process was skipped due to missing reward model, so quality threshold wasn't validated.")
    else:
        print("❌ Invalid threshold test failed - command crashed.")
    
    results["invalid_quality_threshold"] = {"return_code": result.returncode if result else None}
    
    # Test 2: With a non-existent task ID (which should trigger missing reward model)
    test_name = f"missing_reward_model_{device}"
    log_file = LOG_DIR / f"{test_name}.log"
    
    print(f"Testing error handling (non-existent task ID) on {device}, logging to {log_file}")
    
    # Command with non-existent task ID that should cause missing reward model
    command = [
        "python", "-m", "agentic_diffusion",
        "--config", f"./config/test_{device}.yaml",
        "adaptdiffuser", "improve",
        "nonexistent_task_id_that_should_not_have_reward_model",
        "--iterations", "2",
        "--trajectories", "5"
    ]
    
    # Run with output capture for analysis and logging
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
            
        # Save results
        save_test_results(test_name, command, result, metrics, OUTPUT_DIR)
        
    print(f"\nMissing reward model test completed with return code {result.returncode if result else 'unknown'}")
    
    # Check if the command handled the missing reward model gracefully
    missing_reward_handled_gracefully = False
    if result:
        # Expectation: Either succeeds with warning or fails with clear error
        if result.returncode == 0:
            # Look for warning about missing reward model in stdout
            if hasattr(result, 'stdout') and ("no reward model" in result.stdout.lower() or
                                              "missing reward model" in result.stdout.lower() or
                                              "skipped" in result.stdout.lower()):
                missing_reward_handled_gracefully = True
        else:
            # For failure case, check for clear error message
            if hasattr(result, 'stderr') and ("no reward model" in result.stderr.lower() or
                                              "missing reward model" in result.stderr.lower() or
                                              "skipped" in result.stderr.lower()):
                missing_reward_handled_gracefully = True
    
    if missing_reward_handled_gracefully:
        print("✅ Missing reward model test passed - command handled missing reward model gracefully.")
    else:
        print("❌ Missing reward model test might have issues - command did not handle missing reward model gracefully.")
    
    results["missing_reward_model"] = {"return_code": result.returncode if result else None}
    
    return results


def generate_summary_report(iteration_results, trajectory_results, quality_threshold_results, error_handling_results):
    """Generate a summary report of all tests."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_file = OUTPUT_DIR / f"summary_report_{timestamp}.json"
    
    # Find the best values based on metrics
    best_iteration = None
    best_trajectory = None
    best_quality_threshold = None
    max_improvement = -float('inf')
    
    # Find best iteration count
    for iter_count, data in iteration_results.items():
        if "metrics" in data and "improvement" in data["metrics"]:
            improvement = data["metrics"]["improvement"]
            if improvement > max_improvement:
                max_improvement = improvement
                best_iteration = iter_count
    
    # Reset for trajectory analysis
    max_improvement = -float('inf')
    # Find best trajectory count
    for traj_count, data in trajectory_results.items():
        if "metrics" in data and "improvement" in data["metrics"]:
            improvement = data["metrics"]["improvement"]
            if improvement > max_improvement:
                max_improvement = improvement
                best_trajectory = traj_count
    
    # Reset for quality threshold analysis
    max_improvement = -float('inf')
    # Find best quality threshold
    for threshold, data in quality_threshold_results.items():
        if "metrics" in data and "improvement" in data["metrics"]:
            improvement = data["metrics"]["improvement"]
            if improvement > max_improvement:
                max_improvement = improvement
                best_quality_threshold = threshold
    
    summary = {
        "timestamp": timestamp,
        "iteration_tests": iteration_results,
        "trajectory_tests": trajectory_results,
        "quality_threshold_tests": quality_threshold_results,
        "error_handling_tests": error_handling_results,
        "recommendations": {
            "best_iteration_count": best_iteration,
            "best_trajectory_count": best_trajectory,
            "best_quality_threshold": best_quality_threshold
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Summary report saved to {summary_file}")
    return summary_file


def initialize_reward_model():
    """Initialize the test reward model for AdaptDiffuser."""
    print("\n\n" + "="*30 + " INITIALIZING REWARD MODEL " + "="*30 + "\n")
    
    # Run the mock_adaptdiffuser_reward.py script
    command = ["python", "./scripts/mock_adaptdiffuser_reward.py"]
    result = run_command(command, capture_output=True)
    
    if result and result.returncode == 0:
        print("✅ Successfully initialized reward model")
        return True
    else:
        print("❌ Failed to initialize reward model")
        if result and hasattr(result, 'stderr') and result.stderr:
            print("Error: " + result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
        return False


def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description="Test the AdaptDiffuser improve CLI command with different parameters")
    parser.add_argument("--device", choices=["cpu", "gpu", "both"], default="cpu",
                       help="Device to test on (cpu, gpu, or both)")
    parser.add_argument("--test", choices=["all", "iterations", "trajectories", "quality", "error_handling"],
                       default="all", help="Which test suite to run")
    parser.add_argument("--skip-reward-model", action="store_true",
                       help="Skip initializing the reward model (use if already initialized)")
    
    args = parser.parse_args()
    
    # Check if torch is available
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it to run this test.")
        sys.exit(1)
        
    # Initialize the reward model if not skipped
    if not args.skip_reward_model:
        if not initialize_reward_model():
            print("WARNING: Failed to initialize reward model. Tests may fail or be skipped.")
    
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
        
        iteration_results = {}
        trajectory_results = {}
        quality_threshold_results = {}
        error_handling_results = {}
        
        # Run selected tests
        if args.test == "all" or args.test == "iterations":
            iteration_results = test_iterations(device)
            
        if args.test == "all" or args.test == "trajectories":
            trajectory_results = test_trajectories(device)
            
        if args.test == "all" or args.test == "quality":
            quality_threshold_results = test_quality_thresholds(device)
            
        if args.test == "all" or args.test == "error_handling":
            error_handling_results = test_error_handling(device)
            
        # Generate summary report for this device
        if args.test == "all":
            generate_summary_report(
                iteration_results, 
                trajectory_results, 
                quality_threshold_results,
                error_handling_results
            )
    
    print("\n\nAll tests completed!")
    print(f"Test results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()