"""
AdaptDiffuser module for adaptive diffusion models.

This module provides components for building and using AdaptDiffuser,
a framework for adaptive diffusion models that can quickly adapt to
new tasks through efficient gradient-based adaptation and reward-guided
generation.
"""

from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser as BaseAdaptDiffuser
from agentic_diffusion.core.adapt_diffuser.multi_task import MultiTaskAdaptDiffuser
from agentic_diffusion.core.adapt_diffuser.guidance import AdaptDiffuserGuidance
from agentic_diffusion.core.adapt_diffuser.selection import (
    SelectionStrategy,
    TopKSelection,
    TemperatureSelection,
    DiversitySelection,
    HybridSelection
)
from agentic_diffusion.core.adapt_diffuser.utils import (
    encode_task,
    compute_reward_statistics,
    save_adaptation_metrics,
    load_adaptation_metrics,
    calculate_adaptive_guidance_schedule
)

# Import test-specific models
from agentic_diffusion.core.adapt_diffuser.test_models import (
    AdaptDiffuserModel,
    AdaptDiffuserDiscriminator,
    TaskEmbeddingManager,
    TaskRewardModel
)

# Export main AdaptDiffuser class as the default implementation
AdaptDiffuser = BaseAdaptDiffuser

__all__ = [
    'AdaptDiffuser',
    'BaseAdaptDiffuser',
    'MultiTaskAdaptDiffuser',
    'AdaptDiffuserGuidance',
    'SelectionStrategy',
    'TopKSelection',
    'TemperatureSelection',
    'DiversitySelection',
    'HybridSelection',
    'encode_task',
    'compute_reward_statistics',
    'save_adaptation_metrics',
    'load_adaptation_metrics',
    'calculate_adaptive_guidance_schedule',
    'AdaptDiffuserModel',
    'AdaptDiffuserDiscriminator',
    'TaskEmbeddingManager',
    'TaskRewardModel'
]