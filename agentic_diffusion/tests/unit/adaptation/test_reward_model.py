import pytest
from unittest.mock import MagicMock

from agentic_diffusion.adaptation.gradient_adaptation import RewardModel

@pytest.fixture
def mock_reward_fn():
    return MagicMock(name="reward_fn")

@pytest.fixture
def reward_model(mock_reward_fn):
    return RewardModel(reward_fn=mock_reward_fn)

def test_calls_pluggable_reward_function(reward_model, mock_reward_fn):
    # Given: input data
    data = {"output": "foo", "target": "bar"}
    mock_reward_fn.return_value = 0.7
    # When: reward is computed
    reward = reward_model.compute_reward(data)
    # Then: the pluggable function is called and result returned
    mock_reward_fn.assert_called_with(data)
    assert reward == 0.7

def test_gradient_computation_for_rewards(reward_model, mock_reward_fn):
    # Given: input data and a reward function with a gradient
    data = {"output": "foo", "target": "bar"}
    # Simulate a reward function with a gradient attribute
    mock_reward_fn.gradient = MagicMock(return_value=0.42)
    # When: reward gradient is computed
    grad = reward_model.compute_gradient(data)
    # Then: the gradient method is called and result returned
    mock_reward_fn.gradient.assert_called_with(data)
    assert grad == 0.42

def test_supports_multiple_reward_functions():
    # Given: two different reward functions
    def reward_fn1(data): return 1.0
    def reward_fn2(data): return 2.0
    model1 = RewardModel(reward_fn=reward_fn1)
    model2 = RewardModel(reward_fn=reward_fn2)
    # When: rewards are computed
    r1 = model1.compute_reward({"output": "a"})
    r2 = model2.compute_reward({"output": "b"})
    # Then: correct function is used for each model
    assert r1 == 1.0
    assert r2 == 2.0