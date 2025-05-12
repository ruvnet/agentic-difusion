"""
Unit tests for adaptive planner with AdaptDiffuser integration.

This module tests the AdaptivePlanner class and its integration with AdaptDiffuser
for planning tasks using diffusion models.
"""

import torch
import pytest
from unittest.mock import MagicMock, patch

from agentic_diffusion.planning.adaptive_planner import AdaptivePlanner
from agentic_diffusion.planning.state_representations import (
    TaskAdaptiveStateRepresentation,
    StateEncoder,
    StateRewardModel
)
from agentic_diffusion.planning.action_space import ActionSpace
from agentic_diffusion.planning.planning_diffusion import PlanningDiffusionModel
from agentic_diffusion.planning.plan_validator import PlanValidator
from agentic_diffusion.core.noise_schedules import NoiseScheduler, CosineBetaSchedule
from agentic_diffusion.core.adapt_diffuser.guidance import AdaptDiffuserGuidance
from agentic_diffusion.core.adapt_diffuser.utils import encode_task


class TestAdaptivePlanner:
    """Tests for the AdaptivePlanner class."""
    
    @pytest.fixture
    def mock_components(self):
        """Set up mock components for testing."""
        # Mock state encoder
        state_encoder = MagicMock(spec=StateEncoder)
        state_encoder.state_dim = 4
        state_encoder.latent_dim = 8
        state_encoder.encode.return_value = torch.randn(8)
        state_encoder.decode.return_value = torch.randn(4)
        
        # Mock reward model
        reward_model = MagicMock(spec=StateRewardModel)
        reward_model.state_dim = 8
        reward_model.task_dim = 16
        reward_model.compute_reward.return_value = torch.tensor([0.8])
        reward_model.compute_reward_gradient.return_value = torch.randn(8)
        
        # Mock state representation
        state_representation = MagicMock(spec=TaskAdaptiveStateRepresentation)
        state_representation.state_encoder = state_encoder
        state_representation.reward_model = reward_model
        state_representation.encode_states.return_value = torch.randn(8)
        state_representation.decode_latents.return_value = torch.randn(4)
        state_representation.compute_rewards.return_value = torch.tensor([0.8])
        
        # Mock action space
        action_space = MagicMock(spec=ActionSpace)
        action_space.action_dim = 2
        action_space.normalize.return_value = torch.randn(2)
        action_space.denormalize.return_value = torch.randn(2)
        
        # Mock diffusion model
        diffusion_model = MagicMock(spec=PlanningDiffusionModel)
        diffusion_model.state_dim = 8
        diffusion_model.action_dim = 2
        diffusion_model.forward.return_value = torch.randn(1, 2)
        
        # Mock noise scheduler
        noise_scheduler = MagicMock(spec=NoiseScheduler)
        noise_scheduler.prediction_type = "noise"
        noise_scheduler.get_sampling_timesteps.return_value = [999, 500, 100, 0]
        noise_scheduler.step.return_value = torch.randn(1, 10, 2)
        noise_scheduler.add_noise.return_value = (torch.randn(1, 10, 2), torch.randn(1, 10, 2))
        
        # Mock validator
        validator = MagicMock(spec=PlanValidator)
        validator.validate.return_value = (True, {"scores": torch.tensor([0.9])})
        validator.compute_violation_penalties.return_value = torch.tensor([0.1])
        
        return {
            "state_representation": state_representation,
            "action_space": action_space,
            "diffusion_model": diffusion_model,
            "noise_scheduler": noise_scheduler,
            "validator": validator
        }
    
    @pytest.fixture
    def adaptive_planner(self, mock_components):
        """Create an AdaptivePlanner instance for testing."""
        return AdaptivePlanner(
            state_representation=mock_components["state_representation"],
            action_space=mock_components["action_space"],
            diffusion_model=mock_components["diffusion_model"],
            noise_scheduler=mock_components["noise_scheduler"],
            validator=mock_components["validator"],
            max_trajectory_length=10,
            device="cpu"
        )
    
    def test_initialization(self, adaptive_planner, mock_components):
        """Test that the AdaptivePlanner initializes correctly."""
        assert adaptive_planner.state_representation == mock_components["state_representation"]
        assert adaptive_planner.action_space == mock_components["action_space"]
        assert adaptive_planner.diffusion_model == mock_components["diffusion_model"]
        assert adaptive_planner.noise_scheduler == mock_components["noise_scheduler"]
        assert adaptive_planner.validator == mock_components["validator"]
        assert adaptive_planner.max_trajectory_length == 10
        assert isinstance(adaptive_planner.adapt_diffuser, object)
        assert adaptive_planner.device == "cpu"
    
    def test_plan_generation(self, adaptive_planner, mock_components):
        """Test plan generation with AdaptDiffuser."""
        # Mock the AdaptDiffuser generate method
        adaptive_planner.adapt_diffuser.generate = MagicMock(
            return_value=torch.randn(3, 10, 2)  # 3 samples, 10 steps, 2 action dims
        )
        
        # Mock action decoder
        adaptive_planner.action_encoder.decode_sequence = MagicMock(
            return_value=[torch.randn(2) for _ in range(10)]
        )
        
        # Generate a plan
        initial_state = torch.randn(4)
        task = "navigate to the goal"
        states, actions = adaptive_planner.plan(
            initial_state=initial_state,
            task=task,
            num_samples=3
        )
        
        # Check that the plan is generated
        assert len(states) > 0
        assert len(actions) > 0
        
        # Check that the AdaptDiffuser was called with correct parameters
        adaptive_planner.adapt_diffuser.generate.assert_called_once()
        call_args = adaptive_planner.adapt_diffuser.generate.call_args[1]
        assert call_args["batch_size"] == 3
        assert "task" in call_args
        assert "conditioning" in call_args
        
        # Check that the validator was used to select the best plan
        mock_components["validator"].validate.assert_called_once()
    
    def test_multi_objective_plan(self, adaptive_planner):
        """Test multi-objective planning."""
        # Mock the AdaptDiffuser guidance settings
        adaptive_planner.adapt_diffuser.guidance = MagicMock()
        adaptive_planner.adapt_diffuser.guidance.set_objectives = MagicMock()
        
        # Mock plan method
        with patch.object(adaptive_planner, 'plan') as mock_plan:
            mock_plan.return_value = ([torch.randn(4) for _ in range(5)], 
                                      [torch.randn(2) for _ in range(4)])
            
            # Define test objectives
            objectives = {
                "reward": lambda s, a, t: torch.ones(1),
                "safety": lambda s, a, t: torch.ones(1) * 0.8
            }
            weights = {"reward": 0.7, "safety": 0.3}
            
            # Call multi-objective plan
            initial_state = torch.randn(4)
            states, actions = adaptive_planner.multi_objective_plan(
                initial_state=initial_state,
                objective_dict=objectives,
                weights=weights,
                num_samples=5
            )
            
            # Check that the guidance was set properly
            adaptive_planner.adapt_diffuser.guidance.set_objectives.assert_called_once_with(
                objectives, weights
            )
            
            # Check that plan was called
            mock_plan.assert_called_once()
            assert len(states) > 0
            assert len(actions) > 0
    
    def test_plan_with_dynamics(self, adaptive_planner):
        """Test planning with dynamics model."""
        # Mock dynamics model
        dynamics_model = MagicMock()
        dynamics_model.return_value = torch.randn(4)
        adaptive_planner.dynamics_model = dynamics_model
        
        # Mock plan method to return multiple candidates
        with patch.object(adaptive_planner, 'plan') as mock_plan:
            # Return 3 candidate plans
            mock_plan.return_value = [
                ([torch.randn(4) for _ in range(5)], [torch.randn(2) for _ in range(4)]),
                ([torch.randn(4) for _ in range(5)], [torch.randn(2) for _ in range(4)]),
                ([torch.randn(4) for _ in range(5)], [torch.randn(2) for _ in range(4)])
            ]
            
            # Call plan with dynamics
            initial_state = torch.randn(4)
            states, actions = adaptive_planner.plan_with_dynamics(
                initial_state=initial_state,
                num_samples=3
            )
            
            # Check that dynamics model was used
            assert dynamics_model.call_count > 0
            
            # Check result
            assert len(states) > 0
            assert len(actions) > 0
    
    def test_adapt_to_task(self, adaptive_planner):
        """Test adaptation to a specific task."""
        # Mock adapt_diffuser adapt_to_task
        adaptive_planner.adapt_diffuser.adapt_to_task = MagicMock(
            return_value={"loss_history": [0.5, 0.4, 0.3], "reward_history": [0.6, 0.7, 0.8]}
        )
        
        # Test adaptation
        task = "navigate to the goal"
        metrics = adaptive_planner.adapt_to_task(
            task=task,
            initial_states=[torch.randn(4) for _ in range(5)],
            num_steps=50
        )
        
        # Check that adapt_to_task was called
        adaptive_planner.adapt_diffuser.adapt_to_task.assert_called_once()
        
        # Check that metrics were returned
        assert "loss_history" in metrics
        assert "reward_history" in metrics
    
    def test_adapt_to_task_with_demonstrations(self, adaptive_planner):
        """Test adaptation using demonstration trajectories."""
        # Mock adapt_diffuser methods
        adaptive_planner.adapt_diffuser.generate = MagicMock(
            return_value=torch.randn(4, 10, 2)
        )
        
        # Create demo trajectories
        demo_trajectories = [
            (
                [torch.randn(4) for _ in range(5)],  # states
                [torch.randn(2) for _ in range(4)]   # actions
            ),
            (
                [torch.randn(4) for _ in range(5)],  # states
                [torch.randn(2) for _ in range(4)]   # actions
            )
        ]
        
        # Test adaptation with demos
        task = "navigate to the goal"
        metrics = adaptive_planner.adapt_to_task_with_demonstrations(
            task=task,
            demo_trajectories=demo_trajectories,
            num_steps=50
        )
        
        # Check that metrics were returned
        assert isinstance(metrics, dict)
        
        # Basic metrics should be present
        for key in ["loss_history", "reward_history"]:
            assert key in metrics


class TestGuidanceStrategies:
    """Tests for various guidance strategies used with AdaptivePlanner."""
    
    @pytest.fixture
    def reward_model(self):
        """Create a reward model for testing."""
        model = MagicMock(spec=StateRewardModel)
        model.state_dim = 8
        model.task_dim = 16
        model.compute_reward.return_value = torch.tensor([0.8])
        model.compute_reward_gradient.return_value = torch.randn(8)
        return model
    
    def test_multi_objective_guidance(self, reward_model):
        """Test multi-objective guidance."""
        from agentic_diffusion.planning.guidance_strategies import MultiObjectiveGuidance
        
        # Create the guidance
        guidance = MultiObjectiveGuidance(
            reward_model=reward_model,
            guidance_scale=3.0,
            device="cpu"
        )
        
        # Set objectives
        objectives = {
            "reward": lambda s, a, t: torch.ones(1),
            "safety": lambda s, a, t: torch.ones(1) * 0.8
        }
        weights = {"reward": 0.7, "safety": 0.3}
        
        guidance.set_objectives(objectives, weights)
        
        # Check that objectives were set
        assert guidance.objectives == objectives
        assert guidance.weights == weights
        
        # Test guide method with dummy inputs
        x = torch.randn(2, 10, 2, requires_grad=True)
        t = torch.tensor([500, 500])
        conditioning = {"state": torch.randn(2, 8)}
        task = torch.randn(16)
        
        # Should return a gradient
        gradient = guidance.guide(x, t, conditioning, task)
        assert gradient.shape == x.shape
    
    def test_constraint_guidance(self, reward_model):
        """Test constraint-based guidance."""
        from agentic_diffusion.planning.guidance_strategies import ConstraintGuidance
        
        # Create constraints
        constraints = [
            lambda s, a, t: torch.sigmoid(torch.sum(a)),  # Dummy constraint
            lambda s, a, t: torch.sigmoid(-torch.sum(a))  # Another constraint
        ]
        weights = [0.5, 0.5]
        
        # Create the guidance
        guidance = ConstraintGuidance(
            reward_model=reward_model,
            guidance_scale=3.0,
            device="cpu",
            constraint_functions=constraints,
            constraint_weights=weights
        )
        
        # Test guide method with dummy inputs
        x = torch.randn(2, 10, 2, requires_grad=True)
        t = torch.tensor([500, 500])
        conditioning = {"state": torch.randn(2, 8)}
        task = torch.randn(16)
        
        # Should return a gradient
        gradient = guidance.guide(x, t, conditioning, task)
        assert gradient.shape == x.shape