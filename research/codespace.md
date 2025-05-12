To implement AdaptDiffuser in both local and GitHub Codespaces environments while leveraging machine learning capabilities, follow this optimized guide incorporating GitHub's machine learning documentation:

## Local CPU/GPU Setup

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Liang-ZX/AdaptDiffuser.git
cd AdaptDiffuser
conda env create -f environment.yml
conda activate diffuser_kuka
pip install -e .
```

### 2. Configure Hardware Acceleration
For GPU support:
```bash
# Verify CUDA availability
nvidia-smi
```

For CPU-only mode:
```bash
export CUDA_VISIBLE_DEVICES=""
```

## GitHub Codespaces Configuration

### 1. Dev Container Setup
Create `.devcontainer/devcontainer.json`:
```json
{
  "name": "AdaptDiffuser",
  "image": "nvcr.io/nvidia/pytorch:22.07-py3",
  "features": {
    "ghcr.io/devcontainers/features/nvidia-cuda:1": {
      "installCudnn": true
    }
  },
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python", "ms-toolsai.jupyter"]
    }
  },
  "postCreateCommand": "pip install -e ."
}
```

### 2. Machine Learning Workflow Integration
Leverage built-in Jupyter support:
1. Open notebook interface with `Ctrl+Shift+P` > **Jupyter: Create Blank Notebook**
2. Use preinstalled ML stack:
```python
import torch
print(torch.cuda.is_available())  # Verify GPU access [1]
```

### 3. CUDA Optimization
Ensure proper GPU utilization in Codespaces:
```bash
# Rebuild container after config changes
gh codespace rebuild --full
```

## Cross-Environment Execution

### Training Command (CPU/GPU Compatible)
```bash
python scripts/kuka.py --suffix experimental_run \
  --device cuda  # or 'cpu' for non-GPU environments
```

### Key Performance Features
| Feature              | CPU Mode       | GPU Acceleration |
|----------------------|----------------|------------------|
| Batch Processing     | Limited        | Full Parallelism |
| Mixed Precision      | Disabled       | Enabled          |
| Memory Management    | System RAM     | VRAM Optimized   |

For full NVIDIA CUDA configuration details, refer to GitHub's machine learning documentation[1].

This implementation maintains compatibility across environments while utilizing GitHub's native ML toolchain. The configuration automatically detects available hardware resources, falling back to CPU execution when GPUs are unavailable[1].

Citations:
[1] https://docs.github.com/en/enterprise-cloud@latest/codespaces/developing-in-a-codespace/getting-started-with-github-codespaces-for-machine-learning

 