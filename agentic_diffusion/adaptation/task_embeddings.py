"""
Task embedding module for AdaptDiffuser.

This module provides functionality for embedding task descriptions
and instructions for the diffusion model to adapt to.
"""

import logging
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SimpleTaskEmbeddingModel:
    """
    A simple task embedding model that creates deterministic embeddings
    based on the string content of the task description.
    
    This model is intended for testing purposes only.
    """
    
    def __init__(self, config=None):
        """
        Initialize a simple task embedding model.
        
        Args:
            config: Configuration dictionary, optional
        """
        self.config = config or {}
        self.embedding_dim = self.config.get("embedding_dim", 128)
        self.seed = self.config.get("seed", 42)
        self.device = getattr(config, "device", "cpu")
        logger.info(f"Initialized SimpleTaskEmbeddingModel with embedding_dim={self.embedding_dim}")
    
    def to(self, device):
        """
        Move the model to the specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            The model instance
        """
        self.device = device
        return self
    
    def encode(self, task_description):
        """
        Encode a task description into a deterministic embedding.
        
        Args:
            task_description: Task description string
            
        Returns:
            torch.Tensor: Task embedding
        """
        # Use a simple deterministic encoding based on string content
        # For testing purposes, this is sufficient
        if isinstance(task_description, str):
            # Set seed based on task description to get deterministic but different embeddings
            # for different task descriptions
            task_seed = self.seed + sum(ord(c) for c in task_description)
            random.seed(task_seed)
            np.random.seed(task_seed)
            
            # Generate a random vector based on the seed
            embedding = np.random.normal(0, 1, self.embedding_dim)
            
            # Normalize to unit length
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm
                
            # Convert to torch tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            logger.debug(f"Generated embedding for task: {task_description}")
            return embedding_tensor
        else:
            # Return a zero embedding for non-string inputs
            logger.warning(f"Task description is not a string: {type(task_description)}")
            return torch.zeros(self.embedding_dim, dtype=torch.float32)


class MockTaskEmbeddingModel:
    """
    A mock task embedding model for testing that returns fixed embeddings.
    """
    
    def __init__(self, config=None):
        """
        Initialize a mock task embedding model.
        
        Args:
            config: Configuration dictionary, optional
        """
        self.config = config or {}
        self.embedding_dim = self.config.get("embedding_dim", 128)
        self.device = getattr(config, "device", "cpu") if config else "cpu"
        logger.info(f"Initialized MockTaskEmbeddingModel with embedding_dim={self.embedding_dim}")
    
    def to(self, device):
        """
        Move the model to the specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            The model instance
        """
        self.device = device
        return self
    
    def encode(self, task_description):
        """
        Encode a task description into a fixed embedding.
        
        Args:
            task_description: Task description string
            
        Returns:
            torch.Tensor: Fixed task embedding
        """
        # Always return the same embedding for any task
        logger.debug(f"Generating mock embedding for task: {task_description}")
        embedding = np.ones(self.embedding_dim) * 0.1
        # Add a small random component to make it look more realistic
        embedding += np.random.normal(0, 0.01, self.embedding_dim)
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        return torch.tensor(embedding, dtype=torch.float32)


def create_task_embedding_model(config=None):
    """
    Create a task embedding model based on configuration.
    
    Args:
        config: Configuration dictionary, defaults to empty dict
        
    Returns:
        A task embedding model
    """
    config = config or {}
    model_type = config.get("type", "simple")
    
    if model_type == "simple":
        return SimpleTaskEmbeddingModel(config)
    elif model_type == "mock":
        return MockTaskEmbeddingModel(config)
    else:
        logger.warning(f"Unknown task embedding model type: {model_type}, using simple model")
        return SimpleTaskEmbeddingModel(config)


def encode_task(task_description, model=None):
    """
    Encode a task description into an embedding.
    
    Args:
        task_description: Task description string
        model: Task embedding model, optional
        
    Returns:
        torch.Tensor: Task embedding
    """
    if model is None:
        # Default to simple model
        model = SimpleTaskEmbeddingModel()
    
    return model.encode(task_description)