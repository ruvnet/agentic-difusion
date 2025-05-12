"""
Multi-task extension for AdaptDiffuser models.

This module provides extended functionality for handling multiple tasks
simultaneously, including task interpolation and transfer learning capabilities.
"""

import torch
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Union, Any, Tuple, Set

from agentic_diffusion.core.diffusion_model import DiffusionModel
from agentic_diffusion.core.noise_schedules import NoiseScheduler
from agentic_diffusion.core.common_types import RewardModelProtocol, TaskEmbeddingModelProtocol
from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser
from agentic_diffusion.core.adapt_diffuser.utils import encode_task

# Use protocol types
RewardModel = RewardModelProtocol
TaskEmbeddingModel = TaskEmbeddingModelProtocol

# Configure logging
logger = logging.getLogger(__name__)


class MultiTaskAdaptDiffuser(AdaptDiffuser):
    """
    Extended AdaptDiffuser with enhanced multi-task adaptation capabilities.
    
    This class adds support for handling multiple tasks simultaneously and
    implements task interpolation and transfer learning.
    """
    
    def __init__(
        self,
        base_model: DiffusionModel,
        noise_scheduler: NoiseScheduler,
        img_size: int,
        channels: int,
        reward_model: Optional[RewardModel] = None,
        task_embedding_model: Optional[TaskEmbeddingModel] = None,
        task_vector_dim: int = 768,
        guidance_scale: float = 3.0,
        guidance_min_step_percent: float = 0.1,
        guidance_max_step_percent: float = 0.9,
        classifier_scale: float = 0.0,
        buffer_capacity: int = 10000,
        learning_rate: float = 1e-5,
        device: str = None,
        checkpoint_dir: str = './checkpoints',
        use_ddim: bool = True,
        inference_steps: int = 50,
        memory_length: int = 5
    ):
        """
        Initialize the MultiTaskAdaptDiffuser model.
        
        Args:
            base_model: Pretrained diffusion model to adapt
            noise_scheduler: Noise schedule for diffusion process
            img_size: Size of images/latents to generate
            channels: Number of channels in images/latents
            reward_model: Model for computing task-specific rewards
            task_embedding_model: Model for encoding task descriptions
            task_vector_dim: Dimension of task vectors
            guidance_scale: Scale factor for reward gradient guidance
            guidance_min_step_percent: Percentage of steps to start guidance
            guidance_max_step_percent: Percentage of steps to end guidance
            classifier_scale: Scale factor for classifier-free guidance
            buffer_capacity: Maximum capacity of trajectory buffer
            learning_rate: Learning rate for model adaptation
            device: Device to use for computation
            checkpoint_dir: Directory for saving model checkpoints
            use_ddim: Whether to use DDIM (faster) or DDPM (higher quality) sampling
            inference_steps: Number of steps for inference
            memory_length: Number of previous tasks to keep in memory
        """
        super().__init__(
            base_model=base_model,
            noise_scheduler=noise_scheduler,
            img_size=img_size,
            channels=channels,
            reward_model=reward_model,
            task_embedding_model=task_embedding_model,
            guidance_scale=guidance_scale,
            guidance_min_step_percent=guidance_min_step_percent,
            guidance_max_step_percent=guidance_max_step_percent,
            classifier_scale=classifier_scale,
            buffer_capacity=buffer_capacity,
            learning_rate=learning_rate,
            device=device,
            checkpoint_dir=checkpoint_dir,
            use_ddim=use_ddim,
            inference_steps=inference_steps
        )
        
        # Additional configuration
        self.task_vector_dim = task_vector_dim
        self.memory_length = memory_length
        
        # Storage for task-specific information
        self.task_vectors = {}  # Map task_id -> task embedding
        self.task_adapters = {}  # Map task_id -> task-specific adapter weights
        self.task_metrics = {}  # Map task_id -> adaptation metrics
        
        # Task memory for continual learning
        self.task_memory = []  # List of recent tasks (ordered by recency)
        
    def register_task(
        self,
        task: Union[str, torch.Tensor],
        task_vector: Optional[torch.Tensor] = None
    ) -> str:
        """
        Register a new task for adaptation.
        
        Args:
            task: Task description or identifier
            task_vector: Optional pre-computed task vector
            
        Returns:
            Task ID
        """
        # Generate task ID
        task_id = None
        if isinstance(task, str):
            task_id = task
        elif isinstance(task, torch.Tensor):
            task_id = f"task_{torch.sum(task).item():.4f}"
            
        # Encode task if needed
        if task_vector is None and self.task_embedding_model is not None:
            task_vector = self.encode_task(task)
            
        # Store task vector
        if task_vector is not None:
            self.task_vectors[task_id] = task_vector.detach().to(self.device)
            
        # Add to task memory (maintaining limited size)
        if task_id in self.task_memory:
            # Move to front if already exists
            self.task_memory.remove(task_id)
        
        self.task_memory.insert(0, task_id)
        
        # Trim memory to maximum length
        if len(self.task_memory) > self.memory_length:
            self.task_memory = self.task_memory[:self.memory_length]
            
        logger.info(f"Registered task: {task_id}")
        return task_id
        
    def get_task_vector(
        self,
        task: Union[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get the embedding vector for a task.
        
        Args:
            task: Task description or identifier
            
        Returns:
            Task embedding tensor
            
        Raises:
            ValueError: If task is not registered and cannot be encoded
        """
        # Handle direct tensor input
        if isinstance(task, torch.Tensor):
            return task.to(self.device)
            
        # Look up registered task
        if task in self.task_vectors:
            return self.task_vectors[task]
            
        # Try to encode using task embedding model
        if self.task_embedding_model is not None:
            task_vector = self.encode_task(task)
            # Register for future use
            self.register_task(task, task_vector)
            return task_vector
            
        raise ValueError(f"Task '{task}' is not registered and cannot be encoded")
        
    def interpolate_tasks(
        self,
        task1: Union[str, torch.Tensor],
        task2: Union[str, torch.Tensor],
        interpolation_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Interpolate between two tasks to generate a blended task.
        
        Args:
            task1: First task
            task2: Second task
            interpolation_weight: Weight for task2 (0.0 = task1, 1.0 = task2)
            
        Returns:
            Interpolated task embedding
            
        Raises:
            ValueError: If tasks cannot be encoded
        """
        # Get task vectors
        task1_vector = self.get_task_vector(task1)
        task2_vector = self.get_task_vector(task2)
        
        # Interpolate between task vectors
        interpolated = (1 - interpolation_weight) * task1_vector + interpolation_weight * task2_vector
        
        return interpolated
        
    def generate_with_task_interpolation(
        self,
        task1: Union[str, torch.Tensor],
        task2: Union[str, torch.Tensor],
        interpolation_weight: float = 0.5,
        batch_size: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples by interpolating between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            interpolation_weight: Weight for task2 (0.0 = task1, 1.0 = task2)
            batch_size: Number of samples to generate
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated samples
        """
        # Create interpolated task embedding
        interpolated_task = self.interpolate_tasks(
            task1=task1,
            task2=task2,
            interpolation_weight=interpolation_weight
        )
        
        # Generate using interpolated task
        samples = self.generate(
            batch_size=batch_size,
            task=interpolated_task,
            **kwargs
        )
        
        return samples
    
    def adapt_to_multiple_tasks(
        self,
        tasks: List[Union[str, torch.Tensor]],
        steps_per_task: int = 50,
        adaptation_kwargs: Optional[Dict[str, Any]] = None,
        sequential: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Adapt the model to multiple tasks.
        
        Args:
            tasks: List of tasks to adapt to
            steps_per_task: Number of adaptation steps per task
            adaptation_kwargs: Arguments for task adaptation
            sequential: Whether to adapt to tasks sequentially or jointly
            **kwargs: Additional arguments for adapt_to_task
            
        Returns:
            Dictionary mapping task IDs to adaptation metrics
        """
        adaptation_kwargs = adaptation_kwargs or {}
        task_metrics = {}
        
        if sequential:
            # Sequential adaptation (continual learning)
            for task in tasks:
                task_id = task if isinstance(task, str) else f"task_{torch.sum(task).item():.4f}"
                logger.info(f"Adapting to task: {task_id}")
                
                # Adapt to this task
                metrics = self.adapt_to_task(
                    task=task,
                    num_steps=steps_per_task,
                    **adaptation_kwargs,
                    **kwargs
                )
                
                # Store metrics
                task_metrics[task_id] = metrics
                self.task_metrics[task_id] = metrics
                
                # Register task
                self.register_task(task)
        else:
            # Joint adaptation (multi-task learning)
            combined_task_vectors = []
            for task in tasks:
                task_vector = self.get_task_vector(task)
                combined_task_vectors.append(task_vector)
                
            # Create combined task vector (average)
            combined_vector = torch.stack(combined_task_vectors).mean(dim=0)
            
            # Adapt to combined task
            metrics = self.adapt_to_task(
                task=combined_vector,
                num_steps=steps_per_task * len(tasks),
                **adaptation_kwargs,
                **kwargs
            )
            
            # Store metrics for all tasks
            for task in tasks:
                task_id = task if isinstance(task, str) else f"task_{torch.sum(task).item():.4f}"
                task_metrics[task_id] = metrics
                self.task_metrics[task_id] = metrics
                
                # Register task
                self.register_task(task)
                
        return task_metrics
    
    def compute_task_transfer_matrix(
        self,
        tasks: List[Union[str, torch.Tensor]],
        batch_size: int = 4,
        num_samples: int = 20
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute transfer matrix between tasks.
        
        Args:
            tasks: List of tasks to analyze
            batch_size: Batch size for generation
            num_samples: Number of samples per task
            
        Returns:
            Transfer matrix and list of task IDs
        """
        if self.reward_model is None:
            raise ValueError("Reward model is required for transfer matrix computation")
            
        # Register all tasks and collect IDs
        task_ids = []
        for task in tasks:
            task_id = self.register_task(task)
            task_ids.append(task_id)
            
        # Initialize transfer matrix
        num_tasks = len(task_ids)
        transfer_matrix = np.zeros((num_tasks, num_tasks))
        
        # Generate samples for each source task and evaluate on all target tasks
        for i, source_task_id in enumerate(task_ids):
            source_task = self.get_task_vector(source_task_id)
            
            # Generate samples using source task
            total_samples = []
            for j in range(0, num_samples, batch_size):
                batch = min(batch_size, num_samples - j)
                samples = self.generate(
                    batch_size=batch,
                    task=source_task
                )
                total_samples.append(samples)
                
            # Combine sample batches
            combined_samples = torch.cat(total_samples, dim=0)
            
            # Evaluate samples on all target tasks
            for j, target_task_id in enumerate(task_ids):
                target_task = self.get_task_vector(target_task_id)
                
                # Compute rewards using target task
                rewards = self.compute_reward(combined_samples, target_task)
                avg_reward = rewards.mean().item()
                
                # Fill transfer matrix
                transfer_matrix[i, j] = avg_reward
                
        return transfer_matrix, task_ids
    
    def save(
        self,
        path: str,
        save_task_vectors: bool = True,
        save_metrics: bool = True
    ) -> bool:
        """
        Save the multi-task model to disk.
        
        Args:
            path: Path to save the model
            save_task_vectors: Whether to save task vectors
            save_metrics: Whether to save task metrics
            
        Returns:
            Success flag
        """
        # Use base class save method first
        success = super().save(path)
        
        if not success:
            return False
            
        try:
            # Load the saved state dict to add multi-task info
            state_dict = torch.load(path, map_location=self.device)
            
            # Add multi-task specific information
            if save_task_vectors and self.task_vectors:
                task_vectors_dict = {k: v.cpu() for k, v in self.task_vectors.items()}
                state_dict['task_vectors'] = task_vectors_dict
                
            if self.task_adapters:
                state_dict['task_adapters'] = self.task_adapters
                
            if save_metrics and self.task_metrics:
                state_dict['task_metrics'] = self.task_metrics
                
            state_dict['task_memory'] = self.task_memory
            state_dict['task_vector_dim'] = self.task_vector_dim
            state_dict['memory_length'] = self.memory_length
            
            # Save updated state dict
            torch.save(state_dict, path)
            logger.info(f"Saved MultiTaskAdaptDiffuser model to {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save multi-task model: {e}")
            return False
    
    @classmethod
    def load(
        cls,
        path: str,
        base_model: DiffusionModel,
        noise_scheduler: NoiseScheduler,
        reward_model: Optional[RewardModel] = None,
        task_embedding_model: Optional[TaskEmbeddingModel] = None,
        device: str = None
    ) -> 'MultiTaskAdaptDiffuser':
        """
        Load a multi-task model from disk.
        
        Args:
            path: Path to load the model from
            base_model: Model architecture to load weights into
            noise_scheduler: Noise scheduler for diffusion process
            reward_model: Model for computing task-specific rewards
            task_embedding_model: Model for encoding task descriptions
            device: Device to load the model onto
            
        Returns:
            Loaded MultiTaskAdaptDiffuser model
            
        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If loading fails
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        try:
            # Determine device
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load state dictionary
            state_dict = torch.load(path, map_location=device)
            
            # Load model weights
            base_model.load_state_dict(state_dict['base_model'])
            
            # Extract configuration
            guidance_scale = state_dict.get('guidance_scale', 3.0)
            guidance_min_step_percent = state_dict.get('guidance_min_step_percent', 0.1)
            guidance_max_step_percent = state_dict.get('guidance_max_step_percent', 0.9)
            classifier_scale = state_dict.get('classifier_scale', 0.0)
            img_size = state_dict.get('img_size', 64)
            channels = state_dict.get('channels', 4)
            use_ddim = state_dict.get('use_ddim', True)
            inference_steps = state_dict.get('inference_steps', 50)
            
            # Extract multi-task configuration
            task_vector_dim = state_dict.get('task_vector_dim', 768)
            memory_length = state_dict.get('memory_length', 5)
            
            # Create model instance
            model = cls(
                base_model=base_model,
                noise_scheduler=noise_scheduler,
                img_size=img_size,
                channels=channels,
                reward_model=reward_model,
                task_embedding_model=task_embedding_model,
                task_vector_dim=task_vector_dim,
                guidance_scale=guidance_scale,
                guidance_min_step_percent=guidance_min_step_percent,
                guidance_max_step_percent=guidance_max_step_percent,
                classifier_scale=classifier_scale,
                device=device,
                use_ddim=use_ddim,
                inference_steps=inference_steps,
                memory_length=memory_length
            )
            
            # Load trajectory buffer if available
            buffer_path = state_dict.get('buffer_path')
            if buffer_path and os.path.exists(buffer_path):
                model.trajectory_buffer.load_state(buffer_path)
                logger.info(f"Loaded trajectory buffer from {buffer_path}")
                
            # Load task vectors if available
            if 'task_vectors' in state_dict:
                model.task_vectors = {k: v.to(device) for k, v in state_dict['task_vectors'].items()}
                
            # Load task adapters if available
            if 'task_adapters' in state_dict:
                model.task_adapters = state_dict['task_adapters']
                
            # Load task metrics if available
            if 'task_metrics' in state_dict:
                model.task_metrics = state_dict['task_metrics']
                
            # Load task memory if available
            if 'task_memory' in state_dict:
                model.task_memory = state_dict['task_memory']
            
            logger.info(f"Loaded MultiTaskAdaptDiffuser model from {path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Error loading model: {e}")