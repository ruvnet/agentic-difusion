# Agentic Diffusion FAQ

---

### **Q: What is Agentic Diffusion?**
A: It is a modular, diffusion-based framework for code generation and agentic planning, supporting self-adaptation and extensibility.

---

### **Q: How do I install the framework?**
A: See [Quickstart Guide](quickstart.md) or [README.md](../README.md#installation) for step-by-step instructions.

---

### **Q: Which Python versions are supported?**
A: Python 3.8 and above.

---

### **Q: How do I generate code for a specific language?**
A: Use the `CodeGenerationAPI` and specify the `language` parameter. See [Usage Examples](usage_examples.md).

---

### **Q: How do I add a new adaptation or reward strategy?**
A: Subclass `AdaptationMechanism` or `BaseReward` and register your class. See [Extensibility](extensibility.md).

---

### **Q: Where do I configure model and training parameters?**
A: In YAML files under the `config/` directory. See [Setup and Environment](setup_and_env.md).

---

### **Q: How do I run tests?**
A: Run `pytest` in the project root. See [Quickstart Guide](quickstart.md).

---

### **Q: How do I contribute?**
A: Fork the repo, create a feature branch, submit a pull request. See [README.md](../README.md#contributing).

---

### **Q: Is GPU required?**
A: Not required, but recommended for training large models. Set `CUDA_VISIBLE_DEVICES` to select GPUs.

---

### **Q: Where can I find more detailed architecture info?**
A: See [System Overview](agentic_diffusion_overview.md) and [Architecture](architecture.md).

---

### **Q: Who maintains Agentic Diffusion?**
A: The open-source community, with inspiration from the AdaptDiffuser research.

---

For more questions, open an issue or see the [documentation index](agentic_diffusion_overview.md).