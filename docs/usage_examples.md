# Agentic Diffusion Usage Examples

This guide provides practical examples for using the Agentic Diffusion APIs for code generation, adaptation, and evaluation.

---

## 1. Code Generation API

Generate code in a specified language with syntax guidance and custom parameters.

```python
from agentic_diffusion.api.code_generation_api import CodeGenerationAPI

# Initialize the API
api = CodeGenerationAPI()

# Generate Python code for a Fibonacci function
code, metadata = api.generate_code(
    prompt="Create a function to calculate the Fibonacci sequence",
    language="python",
    params={
        "max_length": 200,
        "syntax_guidance": True
    }
)

print("Generated code:")
print(code)
print("Metadata:", metadata)
```

---

## 2. Adaptation API

Define a new task and adapt a model to it.

```python
from agentic_diffusion.api.adaptation_api import AdaptationAPI

# Initialize the API
api = AdaptationAPI()

# Define a custom code generation task
task_id = api.define_task(
    task_description="Generate efficient sorting algorithms",
    examples=[
        {"prompt": "Sort an array", "code": "def quicksort(arr): ..."}
    ]
)

# Adapt a model to the new task
adaptation_id = api.adapt_model(
    model_id="code-model-v1",
    task_id=task_id
)

# Monitor adaptation status
status = api.get_adaptation_status(adaptation_id)
print(f"Adaptation status: {status}")
```

---

## 3. Evaluation API

Evaluate generated code using built-in reward functions.

```python
from agentic_diffusion.api.code_generation_api import CodeGenerationAPI

api = CodeGenerationAPI()

# Generate code
code, _ = api.generate_code(
    prompt="Write a function to reverse a string",
    language="python"
)

# Evaluate code quality and relevance
quality_score = api.evaluate_code(
    code=code,
    reward_type="quality"
)
relevance_score = api.evaluate_code(
    code=code,
    reward_type="relevance",
    reference_solution="def reverse_string(s): return s[::-1]"
)

print("Quality score:", quality_score)
print("Relevance score:", relevance_score)
```

---

## 4. Advanced: Custom Reward Function

Add a custom reward function for evaluation.

```python
from agentic_diffusion.code_generation.rewards import BaseReward

class CustomLengthReward(BaseReward):
    """Reward based on code length (shorter is better)."""
    def compute(self, code: str, **kwargs) -> float:
        return max(0, 1.0 - len(code) / 1000)

# Register and use the custom reward
api = CodeGenerationAPI()
api.register_reward("custom_length", CustomLengthReward())

score = api.evaluate_code(
    code="def foo(): pass",
    reward_type="custom_length"
)
print("Custom length reward score:", score)
```

---

## 5. End-to-End Pipeline Example

```python
from agentic_diffusion.api.code_generation_api import CodeGenerationAPI
from agentic_diffusion.api.adaptation_api import AdaptationAPI

# 1. Define a new code generation task
adapt_api = AdaptationAPI()
task_id = adapt_api.define_task(
    task_description="Generate code for matrix multiplication",
    examples=[{"prompt": "Multiply two matrices", "code": "def matmul(A, B): ..."}]
)

# 2. Adapt the model to the task
adaptation_id = adapt_api.adapt_model(
    model_id="code-model-v1",
    task_id=task_id
)

# 3. Generate code for the new task
gen_api = CodeGenerationAPI()
code, _ = gen_api.generate_code(
    prompt="Multiply two matrices",
    language="python"
)

# 4. Evaluate the generated code
quality = gen_api.evaluate_code(code=code, reward_type="quality")
print("Generated code:", code)
print("Quality score:", quality)
```

---

## See Also

- [Quickstart Guide](quickstart.md)
- [API Reference](../agentic_diffusion/api/)
- [Extensibility](extensibility.md)