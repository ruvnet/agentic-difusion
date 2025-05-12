"""
Adaptive planner module for AdaptDiffuser.

This module provides the main planning capabilities based on AdaptDiffuser.
This is a compatibility wrapper that re-exports the enhanced planner implementation.
"""

import warnings

# Import from modular components
from agentic_diffusion.planning.guidance_strategies import (
    GuidanceStrategy,
    ClassifierFreeGuidance,
    ConstraintGuidance,
    MultiObjectiveGuidance,
    ProgressiveGuidance
)
from agentic_diffusion.planning.trajectory_types import TrajectorySegment
from agentic_diffusion.planning.enhanced_adaptive_planner import EnhancedAdaptivePlanner

# For backward compatibility
AdaptivePlanner = EnhancedAdaptivePlanner

# Display a deprecation warning when directly importing from this file
warnings.warn(
    "Importing directly from adaptive_planner.py is deprecated. "
    "Please import from the specific modules instead:\n"
    "- from agentic_diffusion.planning.guidance_strategies import GuidanceStrategy, ...\n"
    "- from agentic_diffusion.planning.trajectory_types import TrajectorySegment\n"
    "- from agentic_diffusion.planning.enhanced_adaptive_planner import EnhancedAdaptivePlanner",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'GuidanceStrategy', 
    'ClassifierFreeGuidance',
    'ConstraintGuidance',
    'MultiObjectiveGuidance',
    'ProgressiveGuidance',
    'TrajectorySegment',
    'EnhancedAdaptivePlanner',
    'AdaptivePlanner'  # For backward compatibility
]
