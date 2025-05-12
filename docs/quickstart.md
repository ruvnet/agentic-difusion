# Quickstart Guide: Agentic Diffusion

Get started with Agentic Diffusion in minutes. This guide covers installation, basic usage, and pointers to further documentation.

---

## 1. Installation

**Requirements:**  
- Python 3.8+  
- Git

**Recommended:**  
Use a virtual environment for isolation.

```bash
git clone https://github.com/agentic-diffusion/agentic_diffusion.git
cd agentic_diffusion
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python install.py --dev
```

---

## 2. Configuration

- Default config: `config/development.yaml`
- To use a custom config, set the environment variable:
  ```bash
  export AGENTIC_DIFFUSION_CONFIG=config/my_experiment.yaml
  ```

---

## 3. Generate Code

```python
from agentic_diffusion.api.code_generation_api import CodeGenerationAPI

api = CodeGenerationAPI()
code, meta = api.generate_code(
    prompt="Write a function to check for palindromes",
    language="python"
)
print(code)
```

---

## 4. Adapt a Model

```python
from agentic_diffusion.api.adaptation_api import AdaptationAPI

api = AdaptationAPI()
task_id = api.define_task(
    task_description="Generate code for binary search",
    examples=[{"prompt": "Binary search", "code": "def binary_search(arr, x): ..."}]
)
adaptation_id = api.adapt_model(model_id="code-model-v1", task_id=task_id)
```

---

## 5. Evaluate Generated Code

```python
quality = api.evaluate_code(
    code="def foo(): pass",
    reward_type="quality"
)
print("Quality score:", quality)
```

---

## 6. Run Tests

```bash
pytest
```

---

## 7. Next Steps

- [Usage Examples](usage_examples.md)
- [System Overview](agentic_diffusion_overview.md)
- [Extensibility](extensibility.md)
- [FAQ](FAQ.md)

---

**Tip:** All public classes and functions include docstrings. Please maintain and improve docstrings when extending the system.