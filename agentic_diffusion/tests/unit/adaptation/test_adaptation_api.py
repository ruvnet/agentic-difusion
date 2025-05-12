import pytest
from unittest.mock import MagicMock
from agentic_diffusion.api.adaptation_api import AdaptationAPI
from agentic_diffusion.adaptation.gradient_adaptation import RewardModel
from agentic_diffusion.code_generation.rewards.syntax_reward import SyntaxReward
from agentic_diffusion.code_generation.rewards.relevance_reward import RelevanceReward

@pytest.fixture
def mock_diffusion_model():
    model = MagicMock(name="DiffusionModel")
    model.compute_gradients.return_value = [1.0]
    model.apply_gradients.return_value = None
    model.generate.return_value = "adapted_code"
    return model

@pytest.fixture
def mock_code_generator():
    gen = MagicMock(name="CodeGenerator")
    gen.generate_code.return_value = "memory_adapted_code"
    return gen

@pytest.fixture
def reward_model():
    return RewardModel(
        syntax_reward=SyntaxReward(),
        relevance_reward=RelevanceReward(),
        weights={"syntax": 0.7, "style": 0.0, "relevance": 0.3}
    )

@pytest.mark.parametrize("adaptation_type", ["gradient", "memory", "hybrid"])
def test_adaptation_api_adapt(adaptation_type, mock_diffusion_model, mock_code_generator, reward_model):
    api = AdaptationAPI(
        diffusion_model=mock_diffusion_model,
        code_generator=mock_code_generator,
        adaptation_type=adaptation_type,
        reward_model=reward_model,
        config={"adaptation_rate": 0.1, "memory_size": 10}
    )
    code = "def foo(): pass"
    feedback = {"reward": 1.0}
    reference = "def foo(): pass"
    adapted = api.adapt(code, feedback=feedback, language="python", reference=reference)
    assert isinstance(adapted, str)
    assert adapted in ["adapted_code", "memory_adapted_code", code]

def test_adaptation_api_switch_type(mock_diffusion_model, mock_code_generator, reward_model):
    api = AdaptationAPI(
        diffusion_model=mock_diffusion_model,
        code_generator=mock_code_generator,
        adaptation_type="gradient",
        reward_model=reward_model,
    )
    code = "def bar(): pass"
    out1 = api.adapt(code)
    api.set_adaptation_type("memory")
    out2 = api.adapt(code)
    api.set_adaptation_type("hybrid")
    out3 = api.adapt(code)
    assert out1 in ["adapted_code", code]
    assert out2 in ["memory_adapted_code", code]
    assert out3 in ["adapted_code", "memory_adapted_code", code]

def test_adaptation_api_save_and_load_state(tmp_path, mock_diffusion_model, mock_code_generator, reward_model):
    api = AdaptationAPI(
        diffusion_model=mock_diffusion_model,
        code_generator=mock_code_generator,
        adaptation_type="hybrid",
        reward_model=reward_model,
    )
    code = "def baz(): pass"
    api.adapt(code)
    save_path = tmp_path / "adaptation_state.pkl"
    assert api.save_state(str(save_path))
    api2 = AdaptationAPI(
        diffusion_model=mock_diffusion_model,
        code_generator=mock_code_generator,
        adaptation_type="hybrid",
        reward_model=reward_model,
    )
    assert api2.load_state(str(save_path))