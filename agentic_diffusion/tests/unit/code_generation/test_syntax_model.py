import pytest
from unittest.mock import MagicMock

from agentic_diffusion.code_generation.syntax_model import SyntaxModel

@pytest.fixture
def python_code():
    return "def foo(x):\n    return x + 1"

@pytest.fixture
def invalid_python_code():
    return "def foo(x):\nreturn x + 1"

@pytest.fixture
def js_code():
    return "function foo(x) { return x + 1; }"

@pytest.fixture
def invalid_js_code():
    return "function foo(x) { return x + ; }"

@pytest.fixture
def syntax_model():
    # Use real SyntaxModel with default parsers for most tests
    return SyntaxModel()

def test_valid_python_syntax(syntax_model, python_code):
    assert syntax_model.validate_syntax(python_code, language="python") is True

def test_invalid_python_syntax(invalid_python_code):
    mock_parser = MagicMock()
    mock_parser.validate.return_value = False
    model = SyntaxModel(python_parser_cls=lambda: mock_parser)
    assert model.validate_syntax(invalid_python_code, language="python") is False

def test_valid_js_syntax(syntax_model, js_code):
    assert syntax_model.validate_syntax(js_code, language="javascript") is True

def test_invalid_js_syntax(invalid_js_code):
    mock_parser = MagicMock()
    mock_parser.validate.return_value = False
    model = SyntaxModel(javascript_parser_cls=lambda: mock_parser)
    assert model.validate_syntax(invalid_js_code, language="javascript") is False

def test_syntax_aware_sampling_returns_only_valid(syntax_model):
    samples = ["def foo(): pass", "def foo(: pass", "def bar(): return 1"]
    with pytest.MonkeyPatch.context() as m:
        m.setattr(syntax_model, "validate_syntax", lambda s, language="python": s != "def foo(: pass")
        valid = syntax_model.syntax_aware_sample(samples, language="python")
        assert valid == ["def foo(): pass", "def bar(): return 1"]

def test_language_specific_constraints_python(syntax_model):
    code = "def foo(x):\n    return x + 1"
    # Assume constraint: no global variables allowed
    syntax_model.check_constraints = MagicMock(return_value=True)
    assert syntax_model.enforce_constraints(code, language="python") is True
    syntax_model.check_constraints.assert_called_with(code, "python")

def test_language_specific_constraints_js(syntax_model):
    code = "function foo(x) { return x + 1; }"
    syntax_model.check_constraints = MagicMock(return_value=False)
    assert syntax_model.enforce_constraints(code, language="javascript") is False
    syntax_model.check_constraints.assert_called_with(code, "javascript")