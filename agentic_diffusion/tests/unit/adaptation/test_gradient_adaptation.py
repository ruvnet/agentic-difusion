import pytest
from unittest.mock import MagicMock, patch

from agentic_diffusion.adaptation.gradient_adaptation import GradientBasedAdaptation

class DummyTrajectory:
    def __init__(self, data, reward):
        self.data = data
        self.reward = reward

@pytest.fixture
def mock_diffusion_model():
    return MagicMock(name="DiffusionModel")

@pytest.fixture
def mock_reward_signal():
    return MagicMock(name="RewardSignal")

@pytest.fixture
def adaptation(mock_diffusion_model):
    return GradientBasedAdaptation(
        diffusion_model=mock_diffusion_model,
        adaptation_rate=0.1,
        memory_capacity=5
    )

def test_adapts_model_using_reward_gradients(adaptation, mock_diffusion_model):
    # Given: a trajectory and a reward signal
    trajectory = DummyTrajectory(data=[1,2,3], reward=0.8)
    mock_diffusion_model.compute_gradients.return_value = [0.1, 0.2, 0.3]
    # When: adaptation is performed
    adaptation.adapt([trajectory])
    # Then: model gradients should be computed and applied
    mock_diffusion_model.compute_gradients.assert_called_with([1,2,3], 0.8)
    mock_diffusion_model.apply_gradients.assert_called()

def test_memory_buffer_stores_high_quality_trajectories(adaptation):
    # Given: several trajectories with varying rewards
    high = DummyTrajectory(data=[1], reward=0.9)
    low = DummyTrajectory(data=[2], reward=0.1)
    medium = DummyTrajectory(data=[3], reward=0.5)
    # When: trajectories are added
    adaptation.store_trajectory(high)
    adaptation.store_trajectory(low)
    adaptation.store_trajectory(medium)
    # Then: buffer should contain all, sorted by reward
    buffer = adaptation.memory_buffer
    rewards = [t.reward for t in buffer]
    assert sorted(rewards, reverse=True) == rewards

def test_memory_buffer_capacity_limit(adaptation):
    # Given: more trajectories than capacity
    for i in range(10):
        adaptation.store_trajectory(DummyTrajectory(data=[i], reward=i/10))
    # Then: buffer should not exceed capacity
    assert len(adaptation.memory_buffer) <= adaptation.memory_capacity

def test_adaptation_rate_control(adaptation, mock_diffusion_model):
    # Given: a trajectory and a specific adaptation rate
    trajectory = DummyTrajectory(data=[1,2,3], reward=0.5)
    mock_diffusion_model.compute_gradients.return_value = [0.2, 0.2, 0.2]
    # When: adaptation is performed
    adaptation.adapt([trajectory])
    # Then: gradients should be scaled by adaptation_rate
    applied_grads = mock_diffusion_model.apply_gradients.call_args[0][0]
    assert all(abs(g - 0.02) < 1e-6 for g in applied_grads)  # 0.2 * 0.1