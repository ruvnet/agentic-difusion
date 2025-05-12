"""
Unit tests for planning components with AdaptDiffuser integration.

This module tests the functionality of planning components including state representations,
action spaces, plan validation, and integration with AdaptDiffuser.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from agentic_diffusion.planning.state_representations import (
    StateEncoder,
    LinearStateEncoder,
    AdaptiveStateEncoder,
    StateRewardModel,
    TaskAdaptiveStateRepresentation
)
from agentic_diffusion.planning.action_space import (
    ActionSpace,
    ContinuousActionSpace,
    DiscreteActionSpace,
    HybridActionSpace,
    ActionEncoder
)
from agentic_diffusion.planning.plan_validator import (
    PlanValidator,
    RuleBasedValidator,
    LearningBasedValidator
)
from agentic_diffusion.planning.planning_diffusion import (
    PlanningDiffusionModel,
    TrajectoryModel,
    AdaptivePlanner
)
from agentic_diffusion.core.noise_schedules import LinearNoiseScheduler


class TestStateRepresentations:
    """Tests for state representation components."""
    
    def test_linear_state_encoder(self):
        """Test the linear state encoder."""
        # Create encoder
        state_dim = 4
        latent_dim = 2
        encoder = LinearStateEncoder(state_dim, latent_dim)
        
        # Test encoding
        state = torch.randn(state_dim)
        latent = encoder.encode(state)
        
        # Check shapes
        assert latent.shape == (1, latent_dim)
        
        # Test decoding
        reconstructed = encoder.decode(latent)
        assert reconstructed.shape == (1, state_dim)
        
        # Test batch processing
        batch_size = 5
        states = torch.randn(batch_size, state_dim)
        latents = encoder.encode(states)
        assert latents.shape == (batch_size, latent_dim)
        
        # Test normalization
        encoder.update_normalization_stats(states)
        normalized = encoder.normalize_state(states)
        denormalized = encoder.denormalize_state(normalized)
        assert torch.allclose(states, denormalized, rtol=1e-4)
    
    def test_adaptive_state_encoder(self):
        """Test the adaptive state encoder with task conditioning."""
        # Create encoder
        state_dim = 4
        latent_dim = 2
        task_dim = 3
        encoder = AdaptiveStateEncoder(state_dim, latent_dim, task_dim)
        
        # Test encoding with task
        state = torch.randn(state_dim)
        task = torch.randn(task_dim)
        latent = encoder.encode(state, task)
        
        # Check shapes
        assert latent.shape == (1, latent_dim)
        
        # Test decoding with task
        reconstructed = encoder.decode(latent, task)
        assert reconstructed.shape == (1, state_dim)
        
        # Test batch processing
        batch_size = 5
        states = torch.randn(batch_size, state_dim)
        tasks = torch.randn(batch_size, task_dim)
        latents = encoder.encode(states, tasks)
        assert latents.shape == (batch_size, latent_dim)
    
    def test_state_reward_model(self):
        """Test the state reward model."""
        # Create model
        state_dim = 4
        task_dim = 3
        reward_model = StateRewardModel(state_dim, task_dim)
        
        # Test reward computation
        state = torch.randn(state_dim)
        task = torch.randn(task_dim)
        reward = reward_model.compute_reward(state, task)
        
        # Check shape
        assert reward.shape == (1,)
        
        # Test batch processing
        batch_size = 5
        states = torch.randn(batch_size, state_dim)
        tasks = torch.randn(batch_size, task_dim)
        rewards = reward_model.compute_reward(states, tasks)
        assert rewards.shape == (batch_size,)
        
        # Test gradient computation
        state.requires_grad_(True)
        reward = reward_model.compute_reward(state, task)
        gradient = reward_model.compute_reward_gradient(state, task)
        assert gradient.shape == (1, state_dim)
    
    def test_task_adaptive_state_representation(self):
        """Test the task-adaptive state representation."""
        # Create components
        state_dim = 4
        latent_dim = 2
        task_dim = 3
        encoder = AdaptiveStateEncoder(state_dim, latent_dim, task_dim)
        reward_model = StateRewardModel(state_dim, task_dim)
        
        # Create representation
        representation = TaskAdaptiveStateRepresentation(encoder, reward_model)
        
        # Test encoding
        state = torch.randn(state_dim)
        task = torch.randn(task_dim)
        latent = representation.encode_states(state, task)
        assert latent.shape == (1, latent_dim)
        
        # Test decoding
        reconstructed = representation.decode_latents(latent, task)
        assert reconstructed.shape == (1, state_dim)
        
        # Test reward computation
        reward = representation.compute_rewards(state, task)
        assert reward.shape == (1,)
        
        # Test gradient computation
        gradients = representation.compute_reward_gradients(state, task)
        assert gradients.shape == (1, state_dim)


class TestActionSpace:
    """Tests for action space components."""
    
    def test_continuous_action_space(self):
        """Test the continuous action space."""
        # Create space
        low = [-1.0, -2.0]
        high = [1.0, 2.0]
        action_space = ContinuousActionSpace(low, high)
        
        # Test normalization
        action = np.array([0.0, 0.0])
        normalized = action_space.normalize(action)
        assert normalized.shape == (1, 2)
        assert torch.all((normalized >= -1.0) & (normalized <= 1.0))
        
        # Test denormalization
        denormalized = action_space.denormalize(normalized)
        assert torch.allclose(denormalized, torch.tensor([[0.0, 0.0]]), atol=1e-6)
        
        # Test sampling
        samples = action_space.sample(5)
        assert samples.shape == (5, 2)
        assert torch.all((samples >= torch.tensor(low)) & (samples <= torch.tensor(high)))
    
    def test_discrete_action_space(self):
        """Test the discrete action space."""
        # Create space
        n_actions = 4
        action_space = DiscreteActionSpace(n_actions)
        
        # Test normalization
        action = 2
        normalized = action_space.normalize(action)
        assert normalized.shape == (1, n_actions)
        
        # Expected one-hot encoding with -1s and 1s
        expected = torch.full((1, n_actions), -1.0)
        expected[0, 2] = 1.0
        assert torch.allclose(normalized, expected)
        
        # Test denormalization
        denormalized = action_space.denormalize(normalized)
        assert denormalized.item() == 2
        
        # Test sampling
        samples = action_space.sample(5)
        assert samples.shape == (5,)
        assert torch.all((samples >= 0) & (samples < n_actions))
    
    def test_hybrid_action_space(self):
        """Test the hybrid action space."""
        # Create component spaces
        continuous_space = ContinuousActionSpace([-1.0], [1.0])
        discrete_space = DiscreteActionSpace(3)
        
        # Create hybrid space
        hybrid_space = HybridActionSpace([continuous_space, discrete_space])
        
        # Test normalization
        action = [np.array([0.5]), 1]
        normalized = hybrid_space.normalize(action)
        assert normalized.shape == (1, 4)  # 1 continuous + 3 discrete
        
        # Test denormalization
        denormalized = hybrid_space.denormalize(normalized)
        assert len(denormalized) == 2
        assert torch.allclose(denormalized[0], torch.tensor([[0.5]]), atol=1e-6)
        assert denormalized[1].item() == 1
        
        # Test sampling
        samples = hybrid_space.sample(5)
        assert len(samples) == 2
        assert samples[0].shape == (5, 1)
        assert samples[1].shape == (5,)
    
    def test_action_encoder(self):
        """Test the action encoder."""
        # Create space and encoder
        action_space = ContinuousActionSpace([-1.0, -2.0], [1.0, 2.0])
        max_seq_len = 5
        encoder = ActionEncoder(action_space, max_seq_len)
        
        # Test single action encoding
        action = np.array([0.0, 0.0])
        encoded = encoder.encode_action(action)
        assert encoded.shape == (1, 2)
        
        # Test sequence encoding
        actions = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])]
        encoded_seq = encoder.encode_sequence(actions)
        assert encoded_seq.shape == (1, max_seq_len, 2)
        
        # Test sequence decoding
        decoded_seq = encoder.decode_sequence(encoded_seq)
        assert len(decoded_seq) == max_seq_len


class TestPlanValidator:
    """Tests for plan validation components."""
    
    def test_rule_based_validator(self):
        """Test the rule-based validator."""
        # Create constraint functions
        def constraint1(states, actions, task=None):
            # Example: Check if all states are within bounds
            return torch.any(torch.abs(states) > 1.0, dim=2)
            
        def constraint2(states, actions, task=None):
            # Example: Check if actions are smooth
            action_diffs = actions[:, 1:] - actions[:, :-1]
            return torch.any(torch.abs(action_diffs) > 0.5, dim=(1, 2))
        
        # Create validator
        validator = RuleBasedValidator(
            constraint_functions=[constraint1, constraint2],
            constraint_weights=[1.0, 2.0]
        )
        
        # Create test data
        batch_size = 3
        seq_len = 4
        state_dim = 2
        action_dim = 2
        
        states = torch.rand(batch_size, seq_len, state_dim) * 2 - 1  # [-1, 1]
        actions = torch.rand(batch_size, seq_len, action_dim) * 0.2  # Small actions for smoothness
        
        # Simulate a constraint violation in the first batch item
        states[0, 0, 0] = 1.5  # Outside bounds
        
        # Validate plans
        validity, info = validator.validate(states, actions)
        
        # Check results
        assert validity.shape == (batch_size,)
        assert not validity[0]  # First plan should be invalid
        assert info['violations'].shape == (batch_size, 2)
        assert info['violation_counts'].shape == (batch_size,)
        
        # Test penalty computation
        penalties = validator.compute_violation_penalties(states, actions)
        assert penalties.shape == (batch_size,)
        assert penalties[0] > 0  # First plan should have penalties
    
    def test_learning_based_validator(self):
        """Test the learning-based validator."""
        # Create validator
        state_dim = 2
        action_dim = 2
        validator = LearningBasedValidator(state_dim, action_dim)
        
        # Create test data
        batch_size = 3
        seq_len = 4
        
        states = torch.rand(batch_size, seq_len, state_dim)
        actions = torch.rand(batch_size, seq_len, action_dim)
        
        # Test training step
        labels = torch.tensor([1.0, 0.0, 1.0])
        loss = validator.train_step(states, actions, labels)
        assert isinstance(loss, float)
        
        # Test validation
        validity, info = validator.validate(states, actions)
        assert validity.shape == (batch_size,)
        assert 'scores' in info
        
        # Test penalty computation
        penalties = validator.compute_violation_penalties(states, actions)
        assert penalties.shape == (batch_size,)


@pytest.mark.parametrize("use_task_conditioning", [True, False])
class TestPlanningDiffusionModel:
    """Tests for the planning diffusion model."""
    
    def test_model_initialization(self, use_task_conditioning):
        """Test model initialization."""
        # Create model
        state_dim = 4
        action_dim = 2
        task_dim = 3 if use_task_conditioning else 0
        
        model = PlanningDiffusionModel(
            state_dim=state_dim,
            action_dim=action_dim,
            task_conditioned=use_task_conditioning,
            task_dim=task_dim
        )
        
        # Check attributes
        assert model.state_dim == state_dim
        assert model.action_dim == action_dim
        assert model.task_conditioned == use_task_conditioning
        assert model.task_dim == task_dim
    
    def test_forward_pass(self, use_task_conditioning):
        """Test forward pass through the model."""
        # Create model
        state_dim = 4
        action_dim = 2
        task_dim = 3 if use_task_conditioning else 0
        
        model = PlanningDiffusionModel(
            state_dim=state_dim,
            action_dim=action_dim,
            task_conditioned=use_task_conditioning,
            task_dim=task_dim
        )
        
        # Prepare inputs
        batch_size = 2
        x = torch.randn(batch_size, action_dim)
        timesteps = torch.randint(0, 1000, (batch_size,))
        state = torch.randn(batch_size, state_dim)
        
        # Create task embedding if needed
        task_embed = torch.randn(batch_size, task_dim) if use_task_conditioning else None
        
        # Forward pass
        output = model(x, timesteps, state, task_embed)
        
        # Check output shape
        assert output.shape == (batch_size, action_dim)


@pytest.mark.parametrize("use_task_conditioning", [True, False])
class TestTrajectoryModel:
    """Tests for the trajectory model."""
    
    def test_model_initialization(self, use_task_conditioning):
        """Test model initialization."""
        # Create components
        state_dim = 4
        latent_dim = 2
        action_dim = 2
        task_dim = 3 if use_task_conditioning else 0
        
        # Create encoders and diffusion model
        if use_task_conditioning:
            state_encoder = AdaptiveStateEncoder(state_dim, latent_dim, task_dim)
        else:
            state_encoder = LinearStateEncoder(state_dim, latent_dim)
            
        action_space = ContinuousActionSpace([-1.0] * action_dim, [1.0] * action_dim)
        
        diffusion_model = PlanningDiffusionModel(
            state_dim=latent_dim,
            action_dim=action_dim,
            task_conditioned=use_task_conditioning,
            task_dim=task_dim
        )
        
        noise_scheduler = LinearNoiseScheduler(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        # Create trajectory model
        model = TrajectoryModel(
            state_encoder=state_encoder,
            action_space=action_space,
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            max_trajectory_length=5
        )
        
        # Check attributes
        assert model.state_encoder == state_encoder
        assert model.action_space == action_space
        assert model.diffusion_model == diffusion_model
        assert model.noise_scheduler == noise_scheduler
    
    @patch('torch.randn')
    def test_generate_trajectory(self, mock_randn, use_task_conditioning):
        """Test trajectory generation (mocked)."""
        # Create components
        state_dim = 4
        latent_dim = 2
        action_dim = 2
        task_dim = 3 if use_task_conditioning else 0
        
        # Create encoders and diffusion model
        if use_task_conditioning:
            state_encoder = AdaptiveStateEncoder(state_dim, latent_dim, task_dim)
        else:
            state_encoder = LinearStateEncoder(state_dim, latent_dim)
            
        action_space = ContinuousActionSpace([-1.0] * action_dim, [1.0] * action_dim)
        
        # Create mocked diffusion model
        diffusion_model = MagicMock()
        diffusion_model.task_embedding_model = MagicMock() if use_task_conditioning else None
        
        # Mock noise scheduler
        noise_scheduler = MagicMock()
        noise_scheduler.get_sampling_timesteps.return_value = [10, 5, 0]
        
        # Create trajectory model
        model = TrajectoryModel(
            state_encoder=state_encoder,
            action_space=action_space,
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            max_trajectory_length=5
        )
        
        # Mock random noise generation
        mock_randn.return_value = torch.zeros(1, 5, action_dim)
        
        # Set up mock for diffusion model output
        diffusion_model.return_value = torch.zeros(1, action_dim)
        
        # Set up mock for noise scheduler step
        noise_scheduler.step.return_value = torch.zeros(1, 5, action_dim)
        
        # Generate trajectory
        initial_state = torch.zeros(state_dim)
        task = torch.zeros(task_dim) if use_task_conditioning else None
        
        states, actions = model.generate_trajectory(
            initial_state=initial_state,
            task=task,
            num_steps=3
        )
        
        # Check outputs
        assert len(states) > 0
        assert len(actions) > 0


@pytest.mark.parametrize("use_task_conditioning", [True, False])
class TestAdaptivePlanner:
    """Tests for the adaptive planner."""
    
    def test_planner_initialization(self, use_task_conditioning):
        """Test planner initialization."""
        # Create components
        state_dim = 4
        latent_dim = 2
        action_dim = 2
        task_dim = 3 if use_task_conditioning else 0
        
        # Create encoders
        if use_task_conditioning:
            state_encoder = AdaptiveStateEncoder(state_dim, latent_dim, task_dim)
        else:
            state_encoder = LinearStateEncoder(state_dim, latent_dim)
            
        # Create reward model
        reward_model = StateRewardModel(state_dim, task_dim)
        
        # Create state representation
        state_representation = TaskAdaptiveStateRepresentation(state_encoder, reward_model)
        
        # Create action space
        action_space = ContinuousActionSpace([-1.0] * action_dim, [1.0] * action_dim)
        
        # Create diffusion model
        diffusion_model = PlanningDiffusionModel(
            state_dim=latent_dim,
            action_dim=action_dim,
            task_conditioned=use_task_conditioning,
            task_dim=task_dim
        )
        
        # Create noise scheduler
        noise_scheduler = LinearNoiseScheduler(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        # Create validator
        validator = RuleBasedValidator(
            constraint_functions=[lambda s, a, t: torch.zeros(s.shape[0], dtype=torch.bool)],
            constraint_weights=[1.0]
        )
        
        # Create planner
        planner = AdaptivePlanner(
            state_representation=state_representation,
            action_space=action_space,
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            validator=validator
        )
        
        # Check attributes
        assert planner.state_representation == state_representation
        assert planner.action_space == action_space
        assert planner.diffusion_model == diffusion_model
        assert planner.noise_scheduler == noise_scheduler
        assert planner.validator == validator
    
    @patch('agentic_diffusion.core.adapt_diffuser.base.AdaptDiffuser.generate')
    def test_plan_generation(self, mock_generate, use_task_conditioning):
        """Test plan generation (mocked)."""
        # Create components
        state_dim = 4
        latent_dim = 2
        action_dim = 2
        task_dim = 3 if use_task_conditioning else 0
        
        # Create encoders
        if use_task_conditioning:
            state_encoder = AdaptiveStateEncoder(state_dim, latent_dim, task_dim)
        else:
            state_encoder = LinearStateEncoder(state_dim, latent_dim)
            
        # Create reward model
        reward_model = StateRewardModel(state_dim, task_dim)
        
        # Create state representation
        state_representation = TaskAdaptiveStateRepresentation(state_encoder, reward_model)
        
        # Create action space
        action_space = ContinuousActionSpace([-1.0] * action_dim, [1.0] * action_dim)
        
        # Create diffusion model
        diffusion_model = PlanningDiffusionModel(
            state_dim=latent_dim,
            action_dim=action_dim,
            task_conditioned=use_task_conditioning,
            task_dim=task_dim
        )
        
        # Create noise scheduler
        noise_scheduler = LinearNoiseScheduler(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        # Create planner
        planner = AdaptivePlanner(
            state_representation=state_representation,
            action_space=action_space,
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler
        )
        
        # Mock adapt_diffuser.generate
        mock_generate.return_value = torch.zeros(2, 5, action_dim)
        
        # Test plan generation
        initial_state = torch.zeros(state_dim)
        task = torch.zeros(task_dim) if use_task_conditioning else None
        
        states, actions = planner.plan(
            initial_state=initial_state,
            task=task,
            num_samples=2
        )
        
        # Check outputs
        assert len(states) > 0
        assert len(actions) > 0
        
        # Verify adapt_diffuser.generate was called
        mock_generate.assert_called_once()