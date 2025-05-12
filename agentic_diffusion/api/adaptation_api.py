"""
Adaptation API for Agentic Diffusion

Usage:
    # Initialize with desired adaptation mechanism and reward model
    api = AdaptationAPI(
        diffusion_model=diffusion_model,
        code_generator=code_generator,
        adaptation_type="hybrid",  # "gradient", "memory", or "hybrid"
        reward_model=reward_model,
        config={
            "adaptation_rate": 0.1,      # For gradient/hybrid
            "memory_size": 100,          # For memory/hybrid
            "similarity_threshold": 0.7, # For memory/hybrid
            "gradient_weight": 0.5,      # For hybrid
            "memory_weight": 0.5         # For hybrid
        }
    )

    # Adapt code with feedback and reference
    adapted_code = api.adapt(
        code="def foo(): pass",
        feedback={"reward": 1.0},
        language="python",
        reference="def foo(): pass"
    )

    # Dynamically switch adaptation strategy
    api.set_adaptation_type("gradient")

    # Save/load adaptation state for stability and reproducibility
    api.save_state("path/to/state.pkl")
    api.load_state("path/to/state.pkl")

Optimization Tips:
    - Tune 'adaptation_rate' for stable gradient updates (lower for stability, higher for faster adaptation).
    - Adjust 'memory_size' and 'similarity_threshold' for memory-based recall precision.
    - Balance 'gradient_weight' and 'memory_weight' in hybrid mode for best results.
    - Use appropriate reward models for your code domain (syntax, relevance, style).

"""
"""
Adaptation API for Agentic Diffusion

Provides a unified interface for integrating gradient-based, memory-based, and hybrid adaptation mechanisms
with the code generation pipeline. Enables dynamic selection of adaptation strategies and reward models.
"""

from typing import Optional, Any, Dict
from agentic_diffusion.adaptation.adaptation_mechanism import AdaptationMechanism
from agentic_diffusion.adaptation.gradient_adaptation import (
    GradientBasedAdaptation,
    RewardModel,
    SelfEvolvingDiffuser,
)
from agentic_diffusion.adaptation.memory_adaptation import MemoryAdaptation
from agentic_diffusion.adaptation.hybrid_adaptation import HybridAdaptation

class AdaptationAPI:
    """
    API for self-evolving adaptation mechanisms in code generation.
    """

    def __init__(
        self,
        diffusion_model: Any = None,
        code_generator: Any = None,
        adaptation_type: str = "hybrid",
        reward_model: Optional[Any] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the AdaptationAPI.

        Args:
            diffusion_model: The core diffusion model (required for gradient/hybrid).
            code_generator: The code generator (required for memory/hybrid).
            adaptation_type: "gradient", "memory", or "hybrid".
            reward_model: RewardModel instance (optional).
            config: Optional configuration dictionary for adaptation parameters.
        """
        self.diffusion_model = diffusion_model
        self.code_generator = code_generator
        self.reward_model = reward_model
        self.config = config or {}

        self.adaptation_type = adaptation_type
        self.adaptation_mechanism = self._create_adaptation_mechanism(adaptation_type)
        self.self_evolver = None
        if self.adaptation_type == "gradient":
            self.self_evolver = SelfEvolvingDiffuser(
                diffusion_model, self.adaptation_mechanism, reward_model=reward_model
            )

    def _create_adaptation_mechanism(self, adaptation_type: str) -> AdaptationMechanism:
        """
        Factory for adaptation mechanisms.
        """
        if adaptation_type == "gradient":
            return GradientBasedAdaptation(
                self.diffusion_model,
                adaptation_rate=self.config.get("adaptation_rate", 0.1),
                memory_capacity=self.config.get("memory_capacity", 5),
            )
        elif adaptation_type == "memory":
            return MemoryAdaptation(
                self.code_generator,
                memory_size=self.config.get("memory_size", 100),
                similarity_threshold=self.config.get("similarity_threshold", 0.7),
            )
        elif adaptation_type == "hybrid":
            return HybridAdaptation(
                self.diffusion_model,
                self.code_generator,
                gradient_weight=self.config.get("gradient_weight", 0.5),
                memory_weight=self.config.get("memory_weight", 0.5),
                adaptation_rate=self.config.get("adaptation_rate", 0.1),
                memory_size=self.config.get("memory_size", 100),
            )
        else:
            raise ValueError(f"Unknown adaptation_type: {adaptation_type}")

    def set_adaptation_type(self, adaptation_type: str):
        """
        Dynamically switch adaptation strategy.
        """
        self.adaptation_type = adaptation_type
        self.adaptation_mechanism = self._create_adaptation_mechanism(adaptation_type)
        if adaptation_type == "gradient":
            self.self_evolver = SelfEvolvingDiffuser(
                self.diffusion_model, self.adaptation_mechanism, reward_model=self.reward_model
            )
        else:
            self.self_evolver = None

    def adapt(
        self,
        code: str,
        feedback: Optional[dict] = None,
        language: str = "python",
        reference: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Adapt code using the selected adaptation mechanism.

        Args:
            code: The code to adapt.
            feedback: Feedback or reward signals.
            language: Programming language.
            reference: Reference code/spec for relevance.
            **kwargs: Additional parameters.

        Returns:
            str: Adapted code.
        """
        if self.adaptation_type == "gradient" and self.self_evolver:
            # Use self-improvement loop for gradient-based adaptation
            return self.self_evolver.self_improve(
                code, language=language, reference=reference, feedback=feedback
            )
        else:
            return self.adaptation_mechanism.adapt(
                code=code, feedback=feedback, language=language, **kwargs
            )

    def save_state(self, path: str) -> bool:
        """
        Save the adaptation mechanism state.
        """
        return self.adaptation_mechanism.save_state(path)

    def load_state(self, path: str) -> bool:
        """
        Load the adaptation mechanism state.
        """
        return self.adaptation_mechanism.load_state(path)