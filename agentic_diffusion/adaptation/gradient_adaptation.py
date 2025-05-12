"""
Gradient-based Adaptation Mechanism and RewardModel for Agentic Diffusion.
"""

import os
import pickle
from typing import Any, Callable, List, Optional

from agentic_diffusion.adaptation.adaptation_mechanism import AdaptationMechanism

class RewardModel:
    """
    Modular reward computation interface for adaptation.
    Supports pluggable reward functions (syntax, style, relevance).
    """
    def __init__(
        self,
        syntax_reward: Optional[Callable[[str, str], float]] = None,
        style_reward: Optional[Callable[[str, str], float]] = None,
        relevance_reward: Optional[Callable[[str, str, Optional[str]], float]] = None,
        weights: Optional[dict] = None,
    ):
        """
        Args:
            syntax_reward: Callable for syntax reward (code, language) -> float
            style_reward: Callable for style reward (code, language) -> float
            relevance_reward: Callable for relevance (code, reference, language) -> float
            weights: Dict of weights for each reward type
        """
        self.syntax_reward = syntax_reward
        self.style_reward = style_reward
        self.relevance_reward = relevance_reward
        self.weights = weights or {"syntax": 0.5, "style": 0.25, "relevance": 0.25}

    def __call__(self, code: str, language: str = "python", reference: Optional[str] = None) -> float:
        syntax = self.syntax_reward(code, language) if self.syntax_reward else 1.0
        style = self.style_reward(code, language) if self.style_reward else 1.0
        relevance = (
            self.relevance_reward(code, reference, language)
            if self.relevance_reward and reference
            else 1.0
        )
        return (
            self.weights["syntax"] * syntax
            + self.weights["style"] * style
            + self.weights["relevance"] * relevance
        )

class GradientBasedAdaptation(AdaptationMechanism):
    """Adaptation mechanism that uses gradients to adapt the diffusion model."""
    def __init__(self, diffusion_model, adaptation_rate=0.1, memory_capacity=5):
        self.diffusion_model = diffusion_model
        self.adaptation_rate = adaptation_rate
        self.memory_capacity = memory_capacity
        self.memory_buffer = []

    def adapt(self, code=None, feedback=None, language=None, **kwargs):
        trajectories = kwargs.get('trajectories', [])
        if code and not trajectories:
            from collections import namedtuple
            Trajectory = namedtuple('Trajectory', ['data', 'reward'])
            reward_value = 1.0
            if feedback and isinstance(feedback, dict) and 'reward' in feedback:
                reward_value = feedback['reward']
            trajectories = [Trajectory(code, reward_value)]
        for traj in trajectories:
            grads = self.diffusion_model.compute_gradients(traj.data, traj.reward)
            scaled_grads = [g * self.adaptation_rate for g in grads]
            self.diffusion_model.apply_gradients(scaled_grads)
            self.store_trajectory(traj)
        if code:
            adapted_code = self.diffusion_model.generate(code, language=language)
            return adapted_code
        return None

    def store_trajectory(self, trajectory):
        self.memory_buffer.append(trajectory)
        self.memory_buffer.sort(key=lambda t: t.reward, reverse=True)
        if len(self.memory_buffer) > self.memory_capacity:
            self.memory_buffer = self.memory_buffer[:self.memory_capacity]

    def save_state(self, path):
        try:
            state = {
                'adaptation_rate': self.adaptation_rate,
                'memory_capacity': self.memory_capacity,
                'memory_buffer': self.memory_buffer
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            return True
        except Exception:
            return False

    def load_state(self, path):
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self.adaptation_rate = state.get('adaptation_rate', self.adaptation_rate)
            self.memory_capacity = state.get('memory_capacity', self.memory_capacity)
            self.memory_buffer = state.get('memory_buffer', [])
            return True
        except Exception:
            return False

class SelfEvolvingDiffuser:
    """
    Integrates a diffusion model and adaptation mechanism for self-improvement cycles.
    """
    def __init__(
        self,
        diffusion_model: Any,
        adaptation_mechanism: AdaptationMechanism,
        reward_model: Optional[RewardModel] = None,
        max_iterations: int = 10,
    ):
        """
        Args:
            diffusion_model: The core diffusion model.
            adaptation_mechanism: Adaptation mechanism instance.
            reward_model: RewardModel instance (optional).
            max_iterations: Maximum self-improvement cycles.
        """
        self.diffusion_model = diffusion_model
        self.adaptation_mechanism = adaptation_mechanism
        self.reward_model = reward_model
        self.max_iterations = max_iterations

    def self_improve(
        self,
        code: str,
        language: str = "python",
        reference: Optional[str] = None,
        feedback: Optional[dict] = None,
    ) -> str:
        """
        Runs self-improvement cycles on the code.

        Args:
            code: Initial code.
            language: Programming language.
            reference: Reference code/spec for relevance.
            feedback: Optional feedback dict.

        Returns:
            str: Improved code after adaptation.
        """
        improved_code = code
        for _ in range(self.max_iterations):
            reward = (
                self.reward_model(improved_code, language, reference)
                if self.reward_model
                else 1.0
            )
            feedback = {"reward": reward}
            improved_code = self.adaptation_mechanism.adapt(
                code=improved_code, feedback=feedback, language=language
            )
            if not improved_code:
                break
        return improved_code