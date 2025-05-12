#!/usr/bin/env python3
"""
Example script for the AdaptDiffuser generate command.

This script demonstrates basic usage of the AdaptDiffuser generate command
with a single task, batch size, and guidance scale configuration.
"""

import subprocess
import os
import json
import time
from pathlib import Path

def main():
    """Run a simple example of the AdaptDiffuser generate command."""
    print("AdaptDiffuser Generate Command Example")
    print("======================================")
    
    # Create output directory if it doesn't exist
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    # Define the output file with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"adaptdiffuser_generate_example_{timestamp}.json"
    
    # Define command parameters
    config_file = "config/adaptdiffuser_cpu.yaml"  # Use CPU config for compatibility
    task = "example_navigation_task"
    batch_size = 2
    guidance_scale = 3.0
    
    # Build the command
    cmd = [
        "python", "-m", "agentic_diffusion",
        "--config", config_file,
        "adaptdiffuser", "generate", task,
        "--batch-size", str(batch_size),
        "--guidance-scale", str(guidance_scale),
        "--output", str(output_file)
    ]
    
    # Print the command being run
    print(f"Running command: {' '.join(cmd)}\n")
    
    # Execute the command
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    # Print the command result
    print(f"Command completed in {elapsed:.2f}s with return code {result.returncode}")
    
    if result.stdout:
        print("\nStandard Output:")
        print(result.stdout)
    
    if result.stderr:
        print("\nStandard Error:")
        print(result.stderr)
    
    # Check if output file was generated
    if os.path.exists(output_file):
        print(f"\nOutput file created: {output_file}")
        
        # Load and display summary of output file
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Print summary of the output
            print("\nOutput Summary:")
            if 'metadata' in data:
                metadata = data['metadata']
                print(f"- Task: {metadata.get('task', 'unknown')}")
                print(f"- Batch Size: {metadata.get('batch_size', 'unknown')}")
                print(f"- Guidance Scale: {metadata.get('guidance_scale', 'unknown')}")
                
                if 'rewards' in metadata:
                    rewards = metadata['rewards']
                    print(f"- Mean Reward: {rewards.get('mean', 'unknown')}")
                    print(f"- Max Reward: {rewards.get('max', 'unknown')}")
                    print(f"- Min Reward: {rewards.get('min', 'unknown')}")
            
            if 'trajectories' in data:
                print(f"- Number of Trajectories: {len(data['trajectories'])}")
                
                # Print a snippet of the first trajectory if available
                if data['trajectories'] and isinstance(data['trajectories'], list) and len(data['trajectories']) > 0:
                    first_traj = data['trajectories'][0]
                    print("\nFirst Trajectory Snippet (shape or first few values):")
                    if isinstance(first_traj, dict):
                        print(f"Trajectory keys: {list(first_traj.keys())}")
                    elif isinstance(first_traj, list):
                        print(f"Trajectory length: {len(first_traj)}")
                        if len(first_traj) > 0:
                            print(f"First few elements: {first_traj[:3]}")
                    else:
                        print(f"Trajectory type: {type(first_traj)}")
        except Exception as e:
            print(f"Error parsing output file: {e}")
    else:
        print(f"\nOutput file was not created: {output_file}")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()