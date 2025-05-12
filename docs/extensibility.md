# Extending Agentic Diffusion

Agentic Diffusion is designed for extensibility. This guide explains how to add new adaptation strategies, reward functions, and language support.

---

## 1. Adding a New Adaptation Strategy

1. **Subclass `AdaptationMechanism`**  
   File: [`adaptation/adaptation_mechanism.py`](../agentic_diffusion/adaptation/adaptation_mechanism.py)

```python
from agentic_diffusion.adaptation.adaptation_mechanism import AdaptationMechanism

class MyCustomAdaptation(AdaptationMechanism):
    def adapt(self, model, task, **kwargs):
        # Implement custom adaptation logic
        ...
```

2. **Register the new strategy**  
   Update your config or API usage to use your new class.

---

## 2. Adding a New Reward Function

1. **Subclass `BaseReward`**  
   File: [`code_generation/rewards/`](../agentic_diffusion/code_generation/rewards/)

```python
from agentic_diffusion.code_generation.rewards import BaseReward

class MyCustomReward(BaseReward):
    def compute(self, code: str, **kwargs) -> float:
        # Return a reward score
        ...
```

2. **Register the reward**  
   ```python
   api.register_reward("my_custom", MyCustomReward())
   ```

---

## 3. Adding a New Syntax Parser (Language Support)

1. **Implement a new parser**  
   Directory: [`code_generation/syntax_parsers/`](../agentic_diffusion/code_generation/syntax_parsers/)

```python
from agentic_diffusion.code_generation.syntax_parsers.base_parser import BaseParser

class RustParser(BaseParser):
    def parse(self, code: str):
        # Parse Rust code into AST or tokens
        ...
```

2. **Register the parser**  
   Update the code generation config or API to recognize the new language.

---

## 4. Plugging Into the Pipeline

- **APIs**: All new strategies and rewards can be used via the high-level APIs (`api/`).
- **Config**: Reference your new classes in `config/development.yaml` or custom config files.

---

## 5. Testing Your Extensions

- Add unit tests in `agentic_diffusion/tests/unit/adaptation/` or `agentic_diffusion/tests/unit/code_generation/`.
- Use the provided test fixtures for rapid development.

---

## See Also

- [API Reference](../agentic_diffusion/api/)
- [Usage Examples](usage_examples.md)
- [System Overview](agentic_diffusion_overview.md)