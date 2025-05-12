"""
Core diffusion model components for the Agentic Diffusion system.

This module contains the core components needed for diffusion models, 
including noise schedulers, diffusion models, and denoising processes.
"""

# Import the extension function and apply it to extend the NoiseScheduler class
from agentic_diffusion.core.noise_schedules_extension import extend_noise_scheduler

# Apply the extensions
extend_noise_scheduler()

# Now import the extended classes to make them available from this module
from agentic_diffusion.core.noise_schedules import (
    NoiseScheduler,
    LinearScheduler,
    CosineScheduler,
    SigmoidScheduler
)

from agentic_diffusion.core.diffusion_model import (
    DiffusionModel,
    DenoisingDiffusionModel,
    LatentDiffusionModel
)

from agentic_diffusion.core.denoising_process import (
    DenoisingDiffusionProcess,
    DDPMSampler,
    DDIMSampler
)