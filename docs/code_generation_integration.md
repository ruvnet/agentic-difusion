# Agentic Diffusion Code Generation Pipeline: Integration & Usage

## Overview

The Agentic Diffusion code generation pipeline unifies diffusion-based generation, adaptation mechanisms, recursive refinement, and reward-driven improvement into a modular, extensible system. It supports multi-language code generation and exposes a clean API for integration, research, and production use.

---

## Architecture

```
+-------------------+      +-------------------+      +-------------------+
|   DiffusionModel  | ---> |   CodeGenerator   | ---> | CodeAdaptationModel|
+-------------------+      +-------------------+      +-------------------+
        |                        |                           |
        v                        v                           v
  [Noise Schedules]      [Syntax Model]                [Adaptation Mechanisms]
        |                        |                           |
        +------------------------+---------------------------+
                                 |
                                 v
                        +-------------------+
                        |  CodeGenerationAPI|
                        +-------------------+
                                 |
                                 v
                        [API Endpoints: generate, adapt, improve, refine, evaluate, save/load]
```

- **DiffusionModel**: Core generative model for code synthesis.
- **CodeGenerator**: Orchestrates tokenization, syntax modeling, and diffusion-based code generation.
- **CodeAdaptationModel**: Handles adaptation, improvement, and refinement using gradient, memory, or hybrid mechanisms and reward models.
- **Adaptation Mechanisms**: Modular strategies for agentic adaptation (gradient-based, memory-based, hybrid).
- **Reward Models**: Evaluate code for syntax, quality, and relevance.
- **CodeGenerationAPI**: Unified interface for all pipeline operations.

---

## API Usage

### Instantiation

```python
from agentic_diffusion.api.code_generation_api import create_code_generation_api

diffusion_model = ...  # Your trained or mock diffusion model
config = {
    "default_language": "python",
    "adaptation_type": "hybrid",  # or "gradient", "memory"
    "gradient_weight": 0.5,
    "memory_weight": 0.5
}
api = create_code_generation_api(diffusion_model, config)
```

### Code Generation

```python
spec = "Write a function to sum two numbers."
code = api.generate_code(specification=spec, language="python")
```

### Code Adaptation

```python
feedback = {"improve": "Add type hints"}
adapted = api.adapt_code(code=code, language="python", feedback=feedback)
```

### Code Improvement

```python
improved = api.improve_code(code=code, feedback={"fix": "Add docstring"}, language="python")
```

### Recursive Refinement

```python
refined = api.refine_code(code=code, language="python", iterations=3)
```

### Code Evaluation

```python
metrics = api.evaluate_code(code=code, language="python")
# metrics: {"syntax": ..., "quality": ..., "relevance": ..., "overall": ...}
```

### Multi-Language Support

```python
for lang in ["python", "javascript", "java", "go"]:
    code = api.generate_code(specification=spec, language=lang)
```

### State Persistence

```python
api.save_state("path/to/save_dir")
api.load_state("path/to/save_dir")
```

---
---
## Performance Optimization & Profiling

The Agentic Diffusion pipeline supports advanced performance tuning and profiling for efficient code generation on both CPU and GPU.

### Configurable Performance Options

You can optimize the pipeline by passing the following options in the API config:

```python
config = {
    "batch_size": 4,           # Number of samples to generate in parallel
    "precision": "float16",    # Choose "float32" or "float16" for memory/speed tradeoff
    "device": "cuda"           # "cpu" or "cuda" (GPU)
    # ... other options ...
}
api = create_code_generation_api(diffusion_model, config)
```

These options are propagated through the pipeline and used by the diffusion model for optimal performance.

### Profiling and Benchmarking

All main API methods (`generate_code`, `adapt_code`, `improve_code`, `refine_code`) automatically record performance metrics:

- `elapsed_time_sec`: Wall-clock time for the operation
- `memory_current_bytes`: Current memory usage at end of operation
- `memory_peak_bytes`: Peak memory usage during operation

After any API call, access the latest profile via:

```python
profile = api.last_profile
print("Time:", profile["elapsed_time_sec"], "sec")
print("Peak memory:", profile["memory_peak_bytes"], "bytes")
```

### Performance Regression Testing

A dedicated performance regression test is provided in [`agentic_diffusion/tests/system/performance/test_code_generation_performance.py`](../agentic_diffusion/tests/system/performance/test_code_generation_performance.py). This test:

- Runs the pipeline with various batch sizes and precisions
- Asserts that profiling metrics are present and within reasonable bounds
- Guards against performance regressions

Run all performance/system tests with:

```
pytest -xvs agentic_diffusion/tests/system/performance/
```

---

## Testing

- Integration and system tests are provided in:
  - [`agentic_diffusion/tests/integration/code_generation/test_code_generation_pipeline.py`](../agentic_diffusion/tests/integration/code_generation/test_code_generation_pipeline.py)
  - [`agentic_diffusion/tests/system/end_to_end/test_code_generation_pipeline_api.py`](../agentic_diffusion/tests/system/end_to_end/test_code_generation_pipeline_api.py)
- Run all tests with:
  ```
  pytest -xvs agentic_diffusion/tests/system/end_to_end/ agentic_diffusion/tests/integration/code_generation/test_code_generation_pipeline.py
  ```

---

## Extending the Pipeline

- Add new adaptation mechanisms by subclassing and registering with the API.
- Add new reward models for custom evaluation.
- Extend syntax and tokenizer modules for additional languages.

---

## References

- See [`docs/architecture.md`](architecture.md) and [`docs/detailed_architecture.md`](detailed_architecture.md) for deeper technical details.
- For research background, see [`research/phase2.md`](../research/phase2.md).

---

**Contact:**  
For questions or contributions, open an issue or pull request on the repository.