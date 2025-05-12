"""
API for task embedding models in AdaptDiffuser.

This module provides a unified interface for loading and using task embedding models.
"""

import logging
import importlib
import inspect
from typing import Dict, Any, Optional

from agentic_diffusion.adaptation.task_embeddings import create_task_embedding_model, SimpleTaskEmbeddingModel, MockTaskEmbeddingModel

logger = logging.getLogger(__name__)


class TaskEmbeddingModel:
    """
    Wrapper class for task embedding models that provides a unified interface.
    This class delegates to the actual implementation classes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, embedding_dim: int = 128):
        """
        Initialize a task embedding model.
        
        Args:
            config: Configuration dictionary
            embedding_dim: Embedding dimension (used if no config is provided)
        """
        self.config = config or {"embedding_dim": embedding_dim}
        self.embedding_dim = self.config.get("embedding_dim", embedding_dim)
        
        # Load the underlying model
        self.model = self._create_model()
        logger.info(f"Initialized TaskEmbeddingModel with embedding_dim={self.embedding_dim}")
    
    def _create_model(self):
        """Create the underlying model."""
        return create_task_embedding_model(self.config)
    
    def encode(self, task_description):
        """
        Encode a task description into an embedding.
        
        Args:
            task_description: Task description string
            
        Returns:
            Task embedding
        """
        return self.model.encode(task_description)
    
    def to(self, device):
        """
        Move the model to the specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            The model instance
        """
        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        return self


def load_task_embedding_model(config: Dict[str, Any]) -> TaskEmbeddingModel:
    """
    Factory function to load a task embedding model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Task embedding model instance
    """
    if not config:
        logger.warning("No configuration provided for task embedding model, using default")
        return TaskEmbeddingModel()
    
    # Check for paths
    paths_config = config.get("adaptdiffuser_paths", {})
    task_embedding_path = paths_config.get("task_embedding_model")
    
    # Get task embedding config
    task_config = config.get("adaptdiffuser", {}).get("task_embedding", {})
    
    if not task_config:
        logger.warning("No task embedding configuration found, using default")
        return TaskEmbeddingModel()
    
    # Custom import using path
    if task_embedding_path:
        try:
            module_path, _, class_name = task_embedding_path.rpartition(".")
            if not class_name:
                # If no class specified, use the module's create function
                module = importlib.import_module(module_path)
                if hasattr(module, "create_task_embedding_model"):
                    model = module.create_task_embedding_model(task_config)
                    return TaskEmbeddingModel(config=task_config, embedding_dim=model.embedding_dim)
                else:
                    logger.warning(f"Module {module_path} does not have create_task_embedding_model function")
            else:
                # Try to import the specific class
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
                if inspect.ismodule(model_class):
                    # If it's a module, look for a creation function
                    if hasattr(model_class, "create_task_embedding_model"):
                        return model_class.create_task_embedding_model(task_config)
                    else:
                        logger.warning(f"Module {class_name} does not have create_task_embedding_model function")
                        return TaskEmbeddingModel(config=task_config)
                else:
                    # If it's a class, instantiate it
                    return model_class(task_config)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to load task embedding model from {task_embedding_path}: {e}")
    
    # Default to built-in model
    model_type = task_config.get("type", "simple")
    if model_type == "simple":
        return TaskEmbeddingModel(config=task_config)
    elif model_type == "mock":
        model = MockTaskEmbeddingModel(task_config)
        return TaskEmbeddingModel(config=task_config, embedding_dim=model.embedding_dim)
    else:
        logger.warning(f"Unknown model type: {model_type}, using simple task embedding model")
        return TaskEmbeddingModel(config=task_config)