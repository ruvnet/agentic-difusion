"""
Trajectory type definitions for adaptive planning.

This module provides type definitions and utilities for representing
trajectories in the context of adaptive planning.
"""

import torch
from typing import NamedTuple, Any


class TrajectorySegment(NamedTuple):
    """
    Represents a segment of a trajectory for hierarchical planning.
    
    Attributes:
        states: States in the segment
        actions: Actions in the segment
        rewards: Rewards for each state-action pair
        is_valid: Whether the segment is valid
    """
    states: torch.Tensor  # States in the segment
    actions: torch.Tensor  # Actions in the segment
    rewards: torch.Tensor  # Rewards for each state-action pair
    is_valid: bool  # Whether the segment is valid