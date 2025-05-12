import pytest
from unittest.mock import MagicMock

from agentic_diffusion.adaptation.gradient_adaptation import SelfEvolvingDiffuser

@pytest.fixture
def mock_diffusion_model():
    return MagicMock(name="DiffusionModel")

@pytest.fixture
def mock_adaptation():
    return MagicMock(name="AdaptationMechanism")

@pytest.fixture
def self_evolving_diffuser(mock_diffusion_model, mock_adaptation):
    return SelfEvolvingDiffuser(
        diffusion_model=mock_diffusion_model,
        adaptation_mechanism=mock_adaptation
    )

def test_integration_of_diffusion_and_adaptation(self_evolving_diffuser, mock_diffusion_model, mock_adaptation):
    # Given: input data and a reward
    input_data = {"input": [1,2,3]}
    reward = 0.8
    mock_diffusion_model.generate.return_value = "output"
    # When: run_one_cycle is called
    self_evolving_diffuser.run_one_cycle(input_data, reward)
    # Then: diffusion model generates output and adaptation is invoked
    mock_diffusion_model.generate.assert_called_with(input_data)
    mock_adaptation.adapt.assert_called()

def test_self_improvement_over_multiple_cycles(self_evolving_diffuser, mock_diffusion_model, mock_adaptation):
    # Given: input data and rewards for multiple cycles
    input_data = {"input": [1,2,3]}
    rewards = [0.5, 0.7, 0.9]
    # When: run_multiple_cycles is called
    self_evolving_diffuser.run_multiple_cycles(input_data, rewards)
    # Then: adaptation is called for each cycle
    assert mock_adaptation.adapt.call_count == len(rewards)
    assert mock_diffusion_model.generate.call_count == len(rewards)