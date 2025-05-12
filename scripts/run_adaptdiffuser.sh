#!/bin/bash
# Run AdaptDiffuser examples

# Set up environment
export PYTHONPATH=.

# Print header
echo "==============================================="
echo "AdaptDiffuser Command Line Interface Examples"
echo "==============================================="
echo

# Example 1: Generate trajectories
echo "=== Example 1: Generate trajectories ==="
python -m agentic_diffusion adaptdiffuser generate "navigate_maze" --batch-size 4 --guidance-scale 2.5
echo

# Example 2: Adapt to a task
echo "=== Example 2: Adapt to a task ==="
python -m agentic_diffusion adaptdiffuser adapt "avoid_obstacles" --iterations 3 --batch-size 8
echo

# Example 3: Self-improve on a task
echo "=== Example 3: Self-improve on a task ==="
python -m agentic_diffusion adaptdiffuser improve "reach_goal" --iterations 2 --trajectories 20
echo

# Example 4: Full workflow
echo "=== Example 4: Full workflow (adapt, improve, generate) ==="
echo "1. Adapting to task 'solve_maze'..."
python -m agentic_diffusion adaptdiffuser adapt "solve_maze" --iterations 2 --batch-size 4
echo
echo "2. Self-improving on task 'solve_maze'..."
python -m agentic_diffusion adaptdiffuser improve "solve_maze" --iterations 2 --trajectories 10
echo
echo "3. Generating final trajectories for 'solve_maze'..."
python -m agentic_diffusion adaptdiffuser generate "solve_maze" --batch-size 2 --guidance-scale 3.5
echo

echo "All examples completed!"