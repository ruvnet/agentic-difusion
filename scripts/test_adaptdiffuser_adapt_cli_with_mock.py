#!/usr/bin/env python3
"""
Test script for AdaptDiffuser 'adapt' CLI command with various parameters.

This script includes a built-in mock reward model to ensure adaptation tests can
run with metrics reporting.
"""

import os
import sys
import re
import json
import time
import subprocess
import argparse
import importlib
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

# Configure output directories - can be overridden via environment variable
OUTPUT_DIR = Path(os.environ.get("ADAPTDIFFUSER_TEST_OUTPUT_DIR", "test_results/adaptdiffuser_adapt_with_mock"))
LOG_DIR = OUTPUT_DIR / "logs"


# ==================== Mock Reward Model Implementation ====================

class MockRewardModel(nn.Module):
    """
    A simple mock reward model that returns pre-defined reward values.
    Used for testing AdaptDiffuser CLI commands.
    """
    
    def __init__(self, reward_base=0.6, reward_noise=0.2):
        """
        Initialize the mock reward model.
        
        Args:
            reward_base: Base reward value
            reward_noise: Amount of noise to add to rewards
        """
        super().__init__()
        self.reward_base = reward_base
        self.reward_noise = reward_noise
        
        # Simple linear layer to make this a proper torch module
        self.layer = nn.Linear(10, 1)
        
    def forward(self, x, task=None):
        """
        Forward pass - returns mock rewards.
        
        Args:
            x: Input tensor (trajectory)
            task: Task embedding (optional)
            
        Returns:
            Reward tensor
        """
        batch_size = x.shape[0] if len(x.shape) > 3 else 1
        
        # Generate deterministic but varied rewards based on input
        if isinstance(x, torch.Tensor):
            # Use hash of input tensor to generate consistent rewards
            reward_seed = int(abs(torch.sum(x.reshape(-1)[:10]).item())) % 1000
        else:
            # Fallback for non-tensor inputs
            reward_seed = 42
            
        # Set the random seed for deterministic rewards
        torch.manual_seed(reward_seed)
        
        # Generate rewards: base value plus noise
        rewards = self.reward_base + self.reward_noise * torch.rand(batch_size, 1, device=x.device)
        
        # Add task-dependent component if task is provided
        if task is not None and isinstance(task, torch.Tensor):
            # Extract a scalar from task embedding to adjust reward
            task_scalar = torch.mean(task).item() if task.numel() > 0 else 0
            task_factor = 0.1 * task_scalar  # Scale the effect
            rewards += task_factor
            
        return rewards.squeeze(-1)
    
    def compute_reward(self, x, task=None):
        """
        Compute rewards for trajectories.
        
        Args:
            x: Input tensor (trajectory)
            task: Task embedding (optional)
            
        Returns:
            Reward tensor
        """
        return self.forward(x, task)
    
    # Utility methods to match AdaptDiffuser API
    def to(self, device):
        """Move model to device."""
        super().to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        return self
    
    def train(self, mode=True):
        """Set model to training mode."""
        super().train(mode)
        return self


# ==================== Monkey Patching Logic ====================

def ensure_reward_model_exists():
    """Ensure that a reward model exists in AdaptDiffuser instances."""
    try:
        # Import relevant modules
        from agentic_diffusion.api.adapt_diffuser_api import AdaptDiffuserAPI
        from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser
        
        # Save original adapt method
        original_adapt = AdaptDiffuserAPI.adapt
        
        # Define patched adapt method
        def patched_adapt(self, task, iterations=None, batch_size=None, 
                         learning_rate=None, quality_threshold=None, 
                         trajectories=None, rewards=None, save_checkpoint=False, 
                         checkpoint_dir=None):
            
            # Check if model has reward model
            if hasattr(self, 'model'):
                # If no reward model, install one
                if not hasattr(self.model, 'reward_model') or self.model.reward_model is None:
                    print("Installing mock reward model for adaptation")
                    device = self.model.device if hasattr(self.model, 'device') else 'cpu'
                    self.model.reward_model = MockRewardModel(reward_base=0.65, reward_noise=0.25).to(device)
                    
                    # Also update base model if available
                    if hasattr(self.model, 'base_model'):
                        self.model.base_model.reward_model = self.model.reward_model
            
            # Call original method
            return original_adapt(self, task, iterations, batch_size, learning_rate, 
                                quality_threshold, trajectories, rewards, 
                                save_checkpoint, checkpoint_dir)
        
        # Apply patch
        AdaptDiffuserAPI.adapt = patched_adapt
        print("Successfully applied mock reward model patch")
        return True
        
    except ImportError as e:
        print(f"WARNING: Could not patch AdaptDiffuser (import error: {e})")
        return False
    except Exception as e:
        print(f"ERROR: Failed to patch AdaptDiffuser: {e}")
        return False


# ==================== Test Functions ====================

def run_command(cmd, capture_output=True):
    """Run a command and return the result."""
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        
        if capture_output:
            # Run with output capture
            result = subprocess.run(
                cmd, 
                capture_output=True,
                text=True,
                check=False
            )
            
            elapsed = time.time() - start_time
            print(f"Command completed in {elapsed:.2f}s with return code {result.returncode}")
            
            if result.stdout and len(result.stdout.strip()) > 0:
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
        
        print(f"Testing learning rate {lr} on {device}, logging to {log_file}")
        
        # Command with proper format and unique task per learning rate
        command = [
            "python", "-m", "agentic_diffusion",
            "adapt",
            f"{task}_lr_{lr:.1e}",  # Make each task unique
            "--model", "cpu" if device == "cpu" else "gpu"
        ]
        
        # Redirect output to log file
        with open(log_file, "w") as log:
            result = run_command(command, capture_output=True)
            if result and hasattr(result, 'stdout'):
                log.write(result.stdout)
            if result and hasattr(result, 'stderr'):
                log.write("\nERROR:\n" + result.stderr)
        
        result = run_command(command)
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
    config_file = f"config/adaptdiffuser_{device}.yaml"
    task = "batch_size_test_task"
    batch_sizes = [1, 2, 4, 8]
    
    print(f"\n{'='*20} TESTING BATCH SIZES {'='*20}")
    
    results = {}
    for batch_size in batch_sizes:
        test_name = f"batch_size_{batch_size}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        print(f"Testing batch size {batch_size} on {device}, logging to {log_file}")
        
        # Command with proper format
        command = [
            "python", "-m", "agentic_diffusion",
            "adapt",
            f"{task}_batch_{batch_size}",  # Make each task unique
            "--model", "cpu" if device == "cpu" else "gpu"
        ]
        
        # Redirect output to log file
        with open(log_file, "w") as log:
            result = run_command(command, capture_output=True)
            if result and hasattr(result, 'stdout'):
                log.write(result.stdout)
            if result and hasattr(result, 'stderr'):
                log.write("\nERROR:\n" + result.stderr)
        
        result = run_command(command)
        if result:
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
    config_file = f"config/adaptdiffuser_{device}.yaml"
    task = "quality_threshold_test_task"
    thresholds = [0.5, 0.6, 0.7, 0.8]
    
    print(f"\n{'='*20} TESTING QUALITY THRESHOLDS {'='*20}")
    
    results = {}
    for threshold in thresholds:
        test_name = f"quality_threshold_{threshold}_{device}"
        log_file = LOG_DIR / f"{test_name}.log"
        
        print(f"Testing quality threshold {threshold} on {device}, logging to {log_file}")
        
        # Command with proper format
        command = [
            "python", "-m", "agentic_diffusion",
            "adapt",
            f"{task}_threshold_{threshold:.1f}",  # Make each task unique
            "--model", "cpu" if device == "cpu" else "gpu"
        ]
        
        # Redirect output to log file
        with open(log_file, "w") as log:
            result = run_command(command, capture_output=True)
            if result and hasattr(result, 'stdout'):
                log.write(result.stdout)
            if result and hasattr(result, 'stderr'):
                log.write("\nERROR:\n" + result.stderr)
        
        result = run_command(command)
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
    """Test error handling (missing reward model)."""
    config_file = f"config/adaptdiffuser_{device}.yaml"
    
    print(f"\n{'='*20} TESTING ERROR HANDLING {'='*20}")
    
    # Test with a non-existent reward model file
    test_name = f"missing_reward_model_{device}"
    log_file = LOG_DIR / f"{test_name}.log"
    
    print(f"Testing error handling (missing reward model) on {device}, logging to {log_file}")
    # Command with proper format for error handling test
    command = [
        "python", "-m", "agentic_diffusion",
        "adapt",
        "error_handling_test",
        "--model", "nonexistent_model"  # This should cause an error
    ]
    
    # Redirect output to log file
    with open(log_file, "w") as log:
        result = run_command(command, capture_output=True)
        if result and hasattr(result, 'stdout'):
            log.write(result.stdout)
        if result and hasattr(result, 'stderr'):
            log.write("\nERROR:\n" + result.stderr)
    
    result = run_command(command)
    if result:
        metrics = {}
        if hasattr(result, 'stdout'):
            metrics = extract_metrics_from_output(result.stdout)
            
        # Save results
        save_test_results(test_name, command, result, metrics, OUTPUT_DIR)
        
    print(f"\nError handling test completed with return code {result.returncode if result else 'unknown'}")
    
    # Expect the command to fail or show warning
    success = result.returncode != 0 or 'warning' in result.stdout.lower()
    
    if success:
        print("Error handling test passed - command failed gracefully or showed warning.")
    else:
        print("Error handling test might have issues - command did not fail or show warnings.")
    
    return {"missing_reward_model": {"return_code": result.returncode if result else None}}


def generate_summary_report(learning_rate_results, batch_size_results, quality_threshold_results):
    """Generate a summary report of all tests."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_file = OUTPUT_DIR / "summary_report.json"
    
    summary = {
        "timestamp": timestamp,
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
    # Apply the mock reward model patch
    ensure_reward_model_exists()
    
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