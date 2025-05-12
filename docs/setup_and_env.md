# Setup and Environment Configuration

This guide details environment setup, configuration files, and best practices for running Agentic Diffusion.

---

## 1. Python Environment

- **Python Version**: 3.8 or higher is required.
- **Recommended**: Use a virtual environment or conda environment for isolation.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

---

## 2. Installation

- **With UV (Recommended)**:  
  See instructions in [README.md](../README.md#installation).
- **Manual**:  
  Install dependencies from `requirements.txt` and the package in editable mode.

---

## 3. Configuration Files

- **Location**: `config/` directory
- **Default**: `config/development.yaml`
- **Purpose**: Store model, training, and runtime parameters.

### Example: `config/development.yaml`
```yaml
diffusion:
  steps: 1000
  noise_schedule: cosine
code_generation:
  max_length: 256
  syntax_guidance: true
adaptation:
  strategy: hybrid
logging:
  level: INFO
```

**Tip:** Copy and modify this file for custom experiments.

---

## 4. Environment Variables

Some features can be configured via environment variables:

- `AGENTIC_DIFFUSION_CONFIG`: Path to a custom config file.
- `AGENTIC_DIFFUSION_LOG_LEVEL`: Override logging level (e.g., DEBUG, INFO).
- `CUDA_VISIBLE_DEVICES`: Set GPU usage for training/inference.

Example:
```bash
export AGENTIC_DIFFUSION_CONFIG=config/my_experiment.yaml
export AGENTIC_DIFFUSION_LOG_LEVEL=DEBUG
```

---

## 5. Reproducibility

- Use fixed random seeds in your config for reproducible results.
- Document Python, CUDA, and dependency versions for experiments.

---

## 6. Troubleshooting

- **Missing dependencies**: Re-run `python install.py --dev`.
- **CUDA errors**: Check GPU drivers and CUDA toolkit compatibility.
- **Config errors**: Validate YAML syntax and required fields.

---

## 7. Further Reading

- [Quickstart Guide](quickstart.md)
- [System Overview](agentic_diffusion_overview.md)
- [Architecture](architecture.md)