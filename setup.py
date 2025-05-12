from setuptools import setup, find_packages

setup(
    name="agentic_diffusion",
    version="0.1.0",
    description="Advanced diffusion-based generative framework for code generation and agentic planning",
    author="Agentic Diffusion Team",
    author_email="team@agenticdiffusion.ai",
    url="https://github.com/agentic-diffusion/agentic_diffusion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.5.0",
        "pandas>=1.4.0",
        "transformers>=4.20.0",
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "pylint>=2.12.0",
        "black>=23.0.0",
        "isort>=5.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=23.0.0",
            "isort>=5.10.0",
            "pylint>=2.12.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
)