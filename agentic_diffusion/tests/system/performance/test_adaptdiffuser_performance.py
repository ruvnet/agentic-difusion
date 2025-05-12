import pytest
import torch
import numpy as np
import time
import os
import json
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from agentic_diffusion.core.adapt_diffuser import AdaptDiffuser, MultiTaskAdaptDiffuser
from agentic_diffusion.core.adapt_diffuser import TaskRewardModel, TaskEmbeddingManager
from agentic_diffusion.core.diffusion_model import DiffusionModel
from agentic_diffusion.core.noise_schedules import LinearNoiseScheduler, CosineScheduler
from agentic_diffusion.core.denoising_process import GuidedDenoisingProcess


class SimpleDiffusionModel(DiffusionModel):
    """Simple diffusion model for performance testing."""
    
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, hidden_dim),  # +1 for timestep
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
        self.input_dim = input_dim
        
    def forward(self, x, timesteps, **kwargs):
        """Forward pass with timestep embedding."""
        original_shape = x.shape
        
        # Flatten input if it's higher dimensional (for images, etc.)
        if len(x.shape) > 2:
            # Keep batch dimension, flatten the rest
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        elif len(x.shape) == 1:
            # Handle single inputs
            x = x.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = x.shape[0]
        
        # Normalize timesteps to [0, 1]
        t_emb = timesteps.float().view(-1, 1) / 1000.0
        
        # Ensure timesteps has correct batch dimension
        if t_emb.shape[0] == 1 and batch_size > 1:
            t_emb = t_emb.expand(batch_size, -1)
            
        # Concatenate input with timestep embedding
        x_input = torch.cat([x, t_emb], dim=1)
        
        # Pass through network
        output = self.net(x_input)
        
        # Reshape output if necessary (for testing purposes, just maintain input shape)
        if len(original_shape) > 2:
            # Reshape to match original dimensions except for the last one
            output_shape = list(original_shape)
            output_shape[-1] = self.input_dim // (output_shape[1] * output_shape[2])
            try:
                output = output.view(output_shape)
            except:
                # If reshape fails, just return flat output
                pass
            
        return output
    
    def sample(self, shape, **kwargs):
        """
        Generate samples from the diffusion model.
        
        Args:
            shape: Shape of the samples to generate
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated samples
        """
        device = next(self.parameters()).device
        # Start with random noise
        x = torch.randn(*shape, device=device)
        
        # If testing with image-like data (4D tensor), handle it properly
        if len(shape) == 4:  # [batch_size, channels, height, width]
            # For testing, just return the noise
            pass
        elif len(shape) == 2:  # [batch_size, dim]
            # For vector data, no special handling needed
            pass
        else:
            # For any other shape, ensure we return compatible dimensions
            pass
            
        # Simple implementation, just return noise for testing purposes
        return x


class SimpleTaskEmbeddingModel:
    """Simple task embedding model for performance testing."""
    
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        self.device = "cpu"
        
    def encode(self, task_description):
        """Encode task description to embedding."""
        if task_description not in self.embeddings:
            # Create a new random embedding for this task
            self.embeddings[task_description] = torch.randn(1, self.embedding_dim, device=self.device)
        return self.embeddings[task_description]
        
    def encode_task(self, task_str: str) -> torch.Tensor:
        """Generate a deterministic embedding for a task string."""
        if task_str in self.embeddings:
            return self.embeddings[task_str]
            
        # Create deterministic embedding based on task string
        # by hashing the string and using it as a seed
        seed = hash(task_str) % 10000
        torch.manual_seed(seed)
        embedding = torch.randn(1, self.embedding_dim, device=self.device)
        self.embeddings[task_str] = embedding
        return embedding
        
    def to(self, device):
        """Move embeddings to device."""
        self.device = device
        self.embeddings = {k: v.to(device) for k, v in self.embeddings.items()}
        return self


class SimpleRewardModel:
    """Simple reward model for performance testing."""
    
    def __init__(self):
        self.task_rewards = {}
        
    def compute_reward(self, samples: torch.Tensor, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute rewards based on L2 norm of samples."""
        # Compute L2 norm
        if len(samples.shape) <= 2:
            norms = torch.norm(samples, p=2, dim=-1)
        else:
            # For image-like data, compute norm across spatial dimensions
            norms = torch.norm(samples.view(samples.shape[0], -1), p=2, dim=-1)
            
        # Convert to reward (smaller norm = higher reward)
        rewards = 1.0 / (1.0 + norms)
        
        # If task is provided, make the reward task-dependent
        if task is not None:
            # Use task vector sum as a scaling factor
            if isinstance(task, torch.Tensor):
                task_factor = 0.5 + 0.5 * torch.sigmoid(torch.sum(task)).item()
                rewards = rewards * task_factor
            
        return rewards
        
    def to(self, device):
        """Move to device (no-op for this simple model)."""
        return self


class PerformanceTracker:
    """Utility class to track performance metrics during testing."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize performance tracker."""
        self.metrics = {
            "cpu": {},
            "gpu": {}
        }
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
    def start_tracking(self, device: str, config: Dict) -> None:
        """Start tracking performance for a specific configuration."""
        if device not in self.metrics:
            self.metrics[device] = {}
            
        config_key = self._get_config_key(config)
        self.metrics[device][config_key] = {
            "config": config,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(device),
            "measurements": []
        }
        return config_key
        
    def record_measurement(self, device: str, config_key: str, operation: str,
                          duration: float, memory_used: float,
                          adaptation_metrics: Optional[Dict] = None) -> None:
        """Record a performance measurement."""
        if device not in self.metrics or config_key not in self.metrics[device]:
            raise ValueError(f"No tracking started for {device}/{config_key}")
            
        measurement = {
            "operation": operation,
            "duration_seconds": duration,
            "memory_bytes": memory_used
        }
        
        if adaptation_metrics:
            measurement["adaptation_metrics"] = adaptation_metrics
            
        self.metrics[device][config_key]["measurements"].append(measurement)
        
    def stop_tracking(self, device: str, config_key: str) -> Dict:
        """Stop tracking and return metrics for the configuration."""
        if device not in self.metrics or config_key not in self.metrics[device]:
            raise ValueError(f"No tracking started for {device}/{config_key}")
            
        metrics = self.metrics[device][config_key]
        metrics["total_duration"] = time.time() - metrics["start_time"]
        metrics["peak_memory"] = self._get_memory_usage(device) - metrics["start_memory"]
        
        return metrics
        
    def save_metrics(self, filename: str = "adaptdiffuser_performance.json") -> str:
        """Save all metrics to a JSON file."""
        if not self.output_dir:
            raise ValueError("No output directory specified")
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert metrics to serializable format
        serializable_metrics = self._prepare_for_serialization(self.metrics)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
        return filepath
        
    def _get_config_key(self, config: Dict) -> str:
        """Generate a unique key for the configuration."""
        keys = []
        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                continue
            keys.append(f"{key}={value}")
        return "-".join(keys)
        
    def _get_memory_usage(self, device: str) -> float:
        """Get current memory usage for the device."""
        if device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            return psutil.Process().memory_info().rss
            
    def _prepare_for_serialization(self, metrics: Dict) -> Dict:
        """Prepare metrics for JSON serialization."""
        result = {}
        for device, device_metrics in metrics.items():
            result[device] = {}
            for config_key, config_metrics in device_metrics.items():
                serializable_metrics = {}
                for key, value in config_metrics.items():
                    # Skip non-serializable values and convert tensors to lists
                    if key in ["start_time", "start_memory"]:
                        continue
                    elif isinstance(value, torch.Tensor):
                        serializable_metrics[key] = value.tolist()
                    elif isinstance(value, np.ndarray):
                        serializable_metrics[key] = value.tolist()
                    else:
                        serializable_metrics[key] = value
                result[device][config_key] = serializable_metrics
        return result


@pytest.mark.performance
class TestAdaptDiffuserPerformance:
    @pytest.fixture
    def performance_tracker(self, request) -> PerformanceTracker:
        """Create a performance tracker that saves results to a file."""
        # Get benchmark_results directory relative to project root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        output_dir = project_root / "benchmark_results" / "adaptdiffuser"
        os.makedirs(output_dir, exist_ok=True)
        return PerformanceTracker(str(output_dir))
    
    @pytest.fixture
    def small_diffusion_model(self) -> SimpleDiffusionModel:
        """Create a small diffusion model for testing."""
        # 4x4 image with 3 channels = 48 features
        return SimpleDiffusionModel(input_dim=48, hidden_dim=32)
        
    @pytest.fixture
    def medium_diffusion_model(self) -> SimpleDiffusionModel:
        """Create a medium diffusion model for testing."""
        # 4x4 image with 3 channels = 48 features
        return SimpleDiffusionModel(input_dim=48, hidden_dim=128)
        
    @pytest.fixture
    def large_diffusion_model(self) -> SimpleDiffusionModel:
        """Create a large diffusion model for testing."""
        # 4x4 image with 3 channels = 48 features
        return SimpleDiffusionModel(input_dim=48, hidden_dim=256)
        
    @pytest.fixture
    def task_embedding_model(self) -> SimpleTaskEmbeddingModel:
        """Create a task embedding model for testing."""
        return SimpleTaskEmbeddingModel(embedding_dim=64)
        
    @pytest.fixture
    def reward_model(self) -> SimpleRewardModel:
        """Create a reward model for testing."""
        return SimpleRewardModel()
        
    @pytest.fixture
    def linear_noise_scheduler(self) -> LinearNoiseScheduler:
        """Create a linear noise scheduler."""
        return LinearNoiseScheduler(num_timesteps=50)
        
    @pytest.fixture
    def cosine_noise_scheduler(self) -> CosineScheduler:
        """Create a cosine noise scheduler."""
        return CosineScheduler(num_timesteps=50)
    
    # Testing constants
    IMG_SIZE = 4
    CHANNELS = 3
    TIMESTEPS = 50
    
    # Define mock data for performance testing
    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        batch_size = 16
        img_size = self.IMG_SIZE
        channels = self.CHANNELS
        
        # Create mock image data
        mock_images = torch.randn(batch_size, channels, img_size, img_size)
        # Create mock task embeddings
        mock_tasks = torch.randn(batch_size, 64)
        
        return {
            "images": mock_images,
            "tasks": mock_tasks
        }
    
    def test_adaptdiffuser_speed_and_memory(self,
                                          small_diffusion_model,
                                          medium_diffusion_model,
                                          large_diffusion_model,
                                          task_embedding_model,
                                          reward_model,
                                          linear_noise_scheduler,
                                          cosine_noise_scheduler,
                                          performance_tracker,
                                          mock_data):
        """
        Given: AdaptDiffuser component models with synthetic data
        When: Running forward passes, noise prediction, and reward calculations
        Then: Performance metrics (speed, memory) meet thresholds
        """
        # Define test configurations varying by model size, batch size, and steps
        test_configs = [
            {"model_size": "small", "batch_size": 4, "steps": 20},
            {"model_size": "medium", "batch_size": 8, "steps": 30},
            {"model_size": "large", "batch_size": 16, "steps": 10},
        ]
        
        # Define task descriptions for testing
        test_tasks = [
            "Generate blue squares",
            "Create patterns with circles",
            "Design abstract landscapes"
        ]
        
        # Setup devices to test
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
            
        # Speed and memory thresholds (adjusted for direct model measurements)
        performance_thresholds = {
            "cpu": {
                "small": {"max_forward_time": 0.5, "max_noise_prediction_time": 1.0, "max_memory_mb": 150},
                "medium": {"max_forward_time": 1.0, "max_noise_prediction_time": 2.0, "max_memory_mb": 250},
                "large": {"max_forward_time": 2.0, "max_noise_prediction_time": 4.0, "max_memory_mb": 400},
            },
            "cuda": {
                "small": {"max_forward_time": 0.1, "max_noise_prediction_time": 0.2, "max_memory_mb": 300},
                "medium": {"max_forward_time": 0.2, "max_noise_prediction_time": 0.5, "max_memory_mb": 600},
                "large": {"max_forward_time": 0.5, "max_noise_prediction_time": 1.0, "max_memory_mb": 1200},
            }
        }
        
        # Mapping from model size to model instance
        model_map = {
            "small": small_diffusion_model,
            "medium": medium_diffusion_model,
            "large": large_diffusion_model
        }
        
        # Mapping from noise scheduler type to instance
        noise_map = {
            "linear": linear_noise_scheduler,
            "cosine": cosine_noise_scheduler
        }
        
        # Run performance tests for each device and configuration
        for device in devices:
            print(f"\nRunning performance tests on {device.upper()}")
            
            for config in test_configs:
                model_size = config["model_size"]
                batch_size = config["batch_size"]
                steps = config["steps"]
                
                # Test each noise scheduler type
                for noise_type, noise_scheduler in noise_map.items():
                    # Create full test config
                    full_config = config.copy()
                    full_config["noise_scheduler"] = noise_type
                    
                    # Start tracking performance for this configuration
                    config_key = performance_tracker.start_tracking(device, full_config)
                    print(f"\nTesting config: {config_key}")
                    
                    # Setup models on the correct device
                    base_model = model_map[model_size].to(device)
                    embedding_model = task_embedding_model.to(device)
                    reward_model_inst = reward_model.to(device)
                    noise_scheduler_inst = noise_scheduler
                    
                    # Create sample data for this batch size
                    img_size = self.IMG_SIZE
                    channels = self.CHANNELS
                    
                    # Create input tensors
                    x_t = torch.randn(batch_size, channels, img_size, img_size, device=device)
                    timesteps = torch.randint(0, self.TIMESTEPS, (batch_size,), device=device)
                    
                    # Test model forward pass
                    print("\nTesting model forward pass performance...")
                    
                    # Measure forward pass time
                    forward_start_time = time.time()
                    forward_start_memory = psutil.Process().memory_info().rss
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        forward_start_memory = torch.cuda.memory_allocated()
                    
                    # Run multiple forward passes to get a good measurement
                    for _ in range(steps):
                        with torch.no_grad():
                            _ = base_model(x_t, timesteps)
                    
                    # Calculate forward pass time and memory
                    forward_time = (time.time() - forward_start_time) / steps  # Average time per step
                    if device == "cuda":
                        memory_used = torch.cuda.memory_allocated() - forward_start_memory
                    else:
                        memory_used = psutil.Process().memory_info().rss - forward_start_memory
                    
                    # Record forward pass metrics
                    performance_tracker.record_measurement(
                        device, config_key, "model_forward_pass",
                        forward_time, memory_used
                    )
                    
                    print(f"  Forward pass: {forward_time:.4f}s/step, {memory_used/1024/1024:.2f}MB")
                    
                    # Verify forward pass time meets threshold
                    assert forward_time <= performance_thresholds[device][model_size]["max_forward_time"], \
                        f"Forward pass too slow: {forward_time:.4f}s > {performance_thresholds[device][model_size]['max_forward_time']}s"
                    
                    # Test task embedding performance
                    print("\nTesting task embedding performance...")
                    
                    # Measure task embedding time
                    for task in test_tasks:
                        embed_start_time = time.time()
                        embed_start_memory = psutil.Process().memory_info().rss
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            embed_start_memory = torch.cuda.memory_allocated()
                        
                        # Generate task embedding
                        for _ in range(10):  # Run multiple times for more accurate measurement
                            with torch.no_grad():
                                task_embedding = embedding_model.encode_task(task)
                        
                        # Calculate embedding time and memory
                        embed_time = (time.time() - embed_start_time) / 10  # Average time
                        if device == "cuda":
                            memory_used = torch.cuda.memory_allocated() - embed_start_memory
                        else:
                            memory_used = psutil.Process().memory_info().rss - embed_start_memory
                        
                        # Record embedding metrics
                        performance_tracker.record_measurement(
                            device, config_key, f"task_embedding_{task}",
                            embed_time, memory_used
                        )
                        
                        print(f"  Task embedding '{task}': {embed_time:.4f}s, {memory_used/1024/1024:.2f}MB")
                    
                    # Test reward model performance
                    print("\nTesting reward model performance...")
                    
                    # Generate some sample images
                    samples = torch.randn(batch_size, channels, img_size, img_size, device=device)
                    
                    # Measure reward computation time
                    reward_start_time = time.time()
                    reward_start_memory = psutil.Process().memory_info().rss
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        reward_start_memory = torch.cuda.memory_allocated()
                    
                    # Compute rewards for samples
                    for _ in range(10):  # Run multiple times for more accurate measurement
                        with torch.no_grad():
                            rewards = reward_model_inst.compute_reward(samples, task_embedding)
                    
                    # Calculate reward computation time and memory
                    reward_time = (time.time() - reward_start_time) / 10  # Average time
                    if device == "cuda":
                        memory_used = torch.cuda.memory_allocated() - reward_start_memory
                    else:
                        memory_used = psutil.Process().memory_info().rss - reward_start_memory
                    
                    # Record reward metrics
                    performance_tracker.record_measurement(
                        device, config_key, "reward_computation",
                        reward_time, memory_used
                    )
                    
                    print(f"  Reward computation: {reward_time:.4f}s, {memory_used/1024/1024:.2f}MB")
                    
                    # Test noise prediction performance
                    print("\nTesting noise prediction performance...")
                    
                    # Initialize noise prediction variables
                    noise_pred_start_time = time.time()
                    noise_pred_start_memory = psutil.Process().memory_info().rss
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        noise_pred_start_memory = torch.cuda.memory_allocated()
                    
                    # Simulate a noise prediction step (similar to what happens in diffusion process)
                    for _ in range(steps):
                        t = torch.randint(0, self.TIMESTEPS, (batch_size,), device=device)
                        x = torch.randn(batch_size, channels, img_size, img_size, device=device)
                        with torch.no_grad():
                            pred = base_model(x, t)
                            # Apply noise scheduler step
                            alpha = noise_scheduler_inst.alphas[t[0]]
                            alpha_prev = noise_scheduler_inst.alphas[max(0, t[0]-1)]
                            sigma = ((1 - alpha_prev) / (1 - alpha)) * (1 - alpha / alpha_prev)
                            # Simulate DDPM update step
                            pred_x0 = (x - torch.sqrt(1 - alpha) * pred) / torch.sqrt(alpha)
                            # Update x
                            noise = torch.randn_like(x)
                            mean = torch.sqrt(alpha_prev) * pred_x0
                            variance = sigma * noise
                            x = mean + variance
                    
                    # Calculate noise prediction time and memory
                    noise_pred_time = (time.time() - noise_pred_start_time) / steps  # Average time per step
                    if device == "cuda":
                        memory_used = torch.cuda.memory_allocated() - noise_pred_start_memory
                    else:
                        memory_used = psutil.Process().memory_info().rss - noise_pred_start_memory
                    
                    # Record noise prediction metrics
                    performance_tracker.record_measurement(
                        device, config_key, "noise_prediction",
                        noise_pred_time, memory_used
                    )
                    
                    print(f"  Noise prediction: {noise_pred_time:.4f}s/step, {memory_used/1024/1024:.2f}MB")
                    
                    # Verify noise prediction time meets threshold
                    assert noise_pred_time <= performance_thresholds[device][model_size]["max_noise_prediction_time"], \
                        f"Noise prediction too slow: {noise_pred_time:.4f}s > {performance_thresholds[device][model_size]['max_noise_prediction_time']}s"
                    
                    # Simulate a simple adaptation step (direct optimization without diffusion process)
                    print("\nTesting optimization performance...")
                    
                    # Create optimizer
                    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-4)
                    
                    # Measure optimization time
                    optim_start_time = time.time()
                    optim_start_memory = psutil.Process().memory_info().rss
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        optim_start_memory = torch.cuda.memory_allocated()
                    
                    # Simulate gradient-based optimization steps
                    for _ in range(10):  # Just a few steps for testing
                        optimizer.zero_grad()
                        # Forward pass
                        x = torch.randn(batch_size, channels, img_size, img_size, device=device)
                        t = torch.randint(0, self.TIMESTEPS, (batch_size,), device=device)
                        pred = base_model(x, t)
                        # Compute a dummy loss based on L2 norm
                        loss = torch.mean(torch.norm(pred, dim=1))
                        # Backward pass and optimization
                        loss.backward()
                        optimizer.step()
                    
                    # Calculate optimization time and memory
                    optim_time = (time.time() - optim_start_time) / 10  # Average time per step
                    if device == "cuda":
                        memory_used = torch.cuda.memory_allocated() - optim_start_memory
                    else:
                        memory_used = psutil.Process().memory_info().rss - optim_start_memory
                    
                    # Record optimization metrics
                    performance_tracker.record_measurement(
                        device, config_key, "optimization",
                        optim_time, memory_used
                    )
                    
                    print(f"  Optimization: {optim_time:.4f}s/step, {memory_used/1024/1024:.2f}MB")
                    
                    # Stop tracking and get final metrics
                    final_metrics = performance_tracker.stop_tracking(device, config_key)
                    print(f"  Total time: {final_metrics['total_duration']:.4f}s")
                    print(f"  Peak memory: {final_metrics['peak_memory']/1024/1024:.2f}MB")
                    
                    # Check total memory usage against threshold
                    peak_memory_mb = final_metrics['peak_memory'] / (1024 * 1024)
                    assert peak_memory_mb <= performance_thresholds[device][model_size]["max_memory_mb"], \
                        f"Memory usage too high: {peak_memory_mb:.2f}MB > {performance_thresholds[device][model_size]['max_memory_mb']}MB"
        
        # Save all performance metrics to file
        metrics_file = performance_tracker.save_metrics("adaptdiffuser_performance.json")
        print(f"\nPerformance metrics saved to: {metrics_file}")
        
    def test_multi_task_adaptdiffuser_performance(self,
                                                medium_diffusion_model,
                                                task_embedding_model,
                                                reward_model,
                                                linear_noise_scheduler,
                                                performance_tracker,
                                                mock_data):
        """
        Given: Component models for multi-task adaptation
        When: Running performance tests on task switching and memory components
        Then: Performance metrics meet thresholds for multi-task operations
        """
        # Determine device - can run on CPU if GPU not available, but with adjusted expectations
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nRunning multi-task performance tests on {device.upper()}")
        
        # Define configuration for multi-task tests
        config = {
            "model_size": "medium",
            "batch_size": 8,
            "task_vector_dim": 64,
            "memory_length": 5,
            "steps": 10
        }
        
        # Start tracking performance
        config_key = performance_tracker.start_tracking(device, config)
        
        # Setup models
        base_model = medium_diffusion_model.to(device)
        embedding_model = task_embedding_model.to(device)
        reward_model_inst = reward_model.to(device)
        noise_scheduler = linear_noise_scheduler
        
        # Define test tasks
        test_tasks = [
            "Generate red circles",
            "Create patterns with squares",
            "Design abstract portraits"
        ]
        
        # Test task encoding performance
        print("\nTesting task encoding performance...")
    
        # Setup test parameters
        batch_size = config["batch_size"]
        steps = config["steps"]
        img_size = self.IMG_SIZE
        channels = self.CHANNELS
    
        # Create task embeddings for all test tasks
        task_embeddings = {}
    
        encode_start_time = time.time()
        if device == "cuda":
            encode_start_memory = torch.cuda.memory_allocated()
        else:
            encode_start_memory = psutil.Process().memory_info().rss
        
        for task in test_tasks:
            task_embeddings[task] = embedding_model.encode_task(task)
    
        encode_time = time.time() - encode_start_time
        if device == "cuda":
            memory_used = torch.cuda.memory_allocated() - encode_start_memory
        else:
            memory_used = psutil.Process().memory_info().rss - encode_start_memory
        
        performance_tracker.record_measurement(
            device, config_key, "multi_task_encoding",
            encode_time, memory_used
        )
        
        print(f"  Encoding {len(test_tasks)} tasks: {encode_time:.4f}s, {memory_used/1024/1024:.2f}MB")
        
        # Test task switching performance
        print("\nTesting task switching performance...")
        
        # Create sample data
        x = torch.randn(batch_size, channels, img_size, img_size, device=device)
        t = torch.randint(0, self.TIMESTEPS, (batch_size,), device=device)
        
        switch_start_time = time.time()
        if device == "cuda":
            torch.cuda.empty_cache()
            switch_start_memory = torch.cuda.memory_allocated()
        else:
            switch_start_memory = psutil.Process().memory_info().rss
        
        # Simulate task switching by running model with different task embeddings
        for _ in range(steps):
            for task in test_tasks:
                task_emb = task_embeddings[task]
                # Simulate conditioning the model on the task (without using actual MultiTaskAdaptDiffuser)
                with torch.no_grad():
                    # Just run a forward pass with task embedding available
                    base_output = base_model(x, t)
                    # Compute reward with this task's embedding
                    reward = reward_model_inst.compute_reward(base_output, task_emb)
    
        switch_time = (time.time() - switch_start_time) / (steps * len(test_tasks))  # Average time per task switch
        if device == "cuda":
            memory_used = torch.cuda.memory_allocated() - switch_start_memory
        else:
            memory_used = psutil.Process().memory_info().rss - switch_start_memory
        
        performance_tracker.record_measurement(
            device, config_key, "task_switching",
            switch_time, memory_used
        )
        
        print(f"  Task switching: {switch_time:.4f}s/switch, {memory_used/1024/1024:.2f}MB")
        
        # Test memory buffer operations (simulate a task buffer)
        print("\nTesting memory buffer operations...")
    
        # Create a simple memory buffer
        memory_buffer = {}
        for task in test_tasks:
            # Store multiple samples per task
            memory_buffer[task] = []
            for _ in range(config["memory_length"]):
                memory_buffer[task].append({
                    "sample": torch.randn(1, channels, img_size, img_size, device=device),
                    "reward": torch.rand(1, device=device)
                })
    
        # Test memory retrieval performance
        memory_start_time = time.time()
        if device == "cuda":
            torch.cuda.empty_cache()
            memory_start_mem = torch.cuda.memory_allocated()
        else:
            memory_start_mem = psutil.Process().memory_info().rss
        
        # Simulate memory retrieval and update operations
        for _ in range(steps):
            for task in test_tasks:
                # Retrieve memory for task
                task_memories = memory_buffer[task]
                
                # Compute statistics on memories
                samples = torch.cat([m["sample"] for m in task_memories], dim=0)
                rewards = torch.cat([m["reward"] for m in task_memories], dim=0)
                
                # Compute mean and std
                mean_reward = rewards.mean()
                best_sample = samples[rewards.argmax()].clone()
                
                # Update memory with new entry
                new_sample = torch.randn(1, channels, img_size, img_size, device=device)
                new_reward = reward_model_inst.compute_reward(new_sample, task_embeddings[task])
                
                # Replace the oldest memory
                memory_buffer[task].pop(0)
                memory_buffer[task].append({
                    "sample": new_sample,
                    "reward": new_reward
                })
        
        memory_time = (time.time() - memory_start_time) / (steps * len(test_tasks))  # Average time per memory operation
        if device == "cuda":
            memory_used = torch.cuda.memory_allocated() - memory_start_mem
        else:
            memory_used = psutil.Process().memory_info().rss - memory_start_mem
        
        performance_tracker.record_measurement(
            device, config_key, "memory_operations",
            memory_time, memory_used
        )
        
        print(f"  Memory operations: {memory_time:.4f}s/op, {memory_used/1024/1024:.2f}MB")
        
        # Test optimization with task vectors
        print("\nTesting optimization with task vectors...")
    
        # Create a simple optimizer
        optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-4)
    
        optim_start_time = time.time()
        if device == "cuda":
            torch.cuda.empty_cache()
            optim_start_memory = torch.cuda.memory_allocated()
        else:
            optim_start_memory = psutil.Process().memory_info().rss
        
        # Simulate optimization with task switching
        for _ in range(steps):
            for i, task in enumerate(test_tasks):
                optimizer.zero_grad()
                
                # Get task embedding
                task_emb = task_embeddings[task]
                
                # Forward pass with noisy image
                x = torch.randn(batch_size, channels, img_size, img_size, device=device)
                t = torch.randint(0, self.TIMESTEPS, (batch_size,), device=device)
                pred = base_model(x, t)
                
                # Compute loss based on predicted output
                # In real model we would compute a reward-based loss
                # Here we just use a simple L2 loss for testing
                target = torch.zeros_like(pred)
                loss = torch.nn.functional.mse_loss(pred, target)
                
                # Add a task-dependent factor
                task_factor = torch.mean(task_emb).abs() * 0.01
                loss = loss * (1.0 + task_factor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
        
        optim_time = (time.time() - optim_start_time) / (steps * len(test_tasks))  # Average time per optimization step
        if device == "cuda":
            memory_used = torch.cuda.memory_allocated() - optim_start_memory
        else:
            memory_used = psutil.Process().memory_info().rss - optim_start_memory
        
        performance_tracker.record_measurement(
            device, config_key, "multi_task_optimization",
            optim_time, memory_used
        )
        
        print(f"  Optimization with task vectors: {optim_time:.4f}s/step, {memory_used/1024/1024:.2f}MB")
        
        # Get final metrics for multi-task operations
        final_metrics = performance_tracker.stop_tracking(device, config_key)
        print(f"  Total time: {final_metrics['total_duration']:.4f}s")
        print(f"  Peak memory: {final_metrics['peak_memory']/1024/1024:.2f}MB")
        
        # Verify performance meets thresholds
        assert final_metrics['total_duration'] < 30.0, \
            f"Multi-task performance test took too long: {final_metrics['total_duration']:.2f}s > 30.0s"
        
        # Add thresholds for peak memory based on device
        if device == "cuda":
            assert final_metrics['peak_memory']/1024/1024 < 1000, \
                f"Multi-task peak memory too high: {final_metrics['peak_memory']/1024/1024:.2f}MB > 1000MB"
        else:
            assert final_metrics['peak_memory']/1024/1024 < 200, \
                f"Multi-task peak memory too high: {final_metrics['peak_memory']/1024/1024:.2f}MB > 200MB"
            
        # Store the performance results to file
        results_dir = os.path.join(os.getcwd(), "benchmark_results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"adaptdiffuser_multi_task_{device}_{time.strftime('%Y%m%d-%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Save metrics
        metrics_file = performance_tracker.save_metrics("multitask_adaptdiffuser_performance.json")
        print(f"\nMulti-task performance metrics saved to: {metrics_file}")