import pytest
from agentic_diffusion.api.code_generation_api import create_code_generation_api

@pytest.fixture
def mock_diffusion_model():
    class MockModel:
        def sample(self, specification, lang, partial_code=None, batch_size=1, precision="float32", device=None):
            # Simulate batch generation
            return [f"def foo_{i}(): pass" for i in range(batch_size)]
        def generate(self, specification, lang, partial_code=None, batch_size=1, precision="float32", device=None):
            return f"def foo(): pass"
    return MockModel()

@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_code_generation_performance(batch_size, precision, mock_diffusion_model):
    config = {
        "batch_size": batch_size,
        "precision": precision,
        "device": "cpu"
    }
    api = create_code_generation_api(mock_diffusion_model, config)
    code = api.generate_code("Write a function", language="python")
    # Check code output type
    assert isinstance(code, str)
    # Check profiling info is present
    assert hasattr(api, "last_profile")
    profile = api.last_profile
    assert "elapsed_time_sec" in profile
    assert "memory_peak_bytes" in profile
    # Performance regression: elapsed time should be reasonable for mock
    assert profile["elapsed_time_sec"] < 1.0
    # Memory usage should be reasonable for mock
    assert profile["memory_peak_bytes"] < 10**7