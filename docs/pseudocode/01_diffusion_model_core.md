# Diffusion Model Core Pseudocode

This document outlines the pseudocode for the core diffusion model implementation, which serves as the foundation for the Agentic Diffusion system.

## DiffusionModel Class

```python
class DiffusionModel:
    """
    Core implementation of diffusion model for generation tasks.
    
    Supports both unconditional and conditional generation through a configurable
    diffusion process.
    """
    
    def __init__(self, config):
        """
        Initialize diffusion model with configuration.
        
        Args:
            config: Configuration object with model parameters
                   {
                     "model_type": "conditional" | "unconditional",
                     "embedding_dim": int,
                     "hidden_dim": int,
                     "num_layers": int,
                     "num_heads": int,
                     "dropout": float,
                     "num_diffusion_steps": int,
                     "noise_schedule": "linear" | "cosine" | "custom",
                     "device": "cpu" | "cuda",
                     "precision": "fp32" | "fp16" | "bf16"
                   }
        """
        # TEST: Configuration validation ensures all required parameters are present
        self._validate_config(config)
        
        self.config = config
        self.model_type = config.get("model_type", "conditional")
        self.num_diffusion_steps = config.get("num_diffusion_steps", 1000)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.precision = config.get("precision", "fp32")
        
        # Setup noise schedule
        # TEST: Noise schedule correctly initialized based on type
        self.noise_schedule = self._create_noise_schedule(
            config.get("noise_schedule", "linear")
        )
        
        # Create denoiser network
        # TEST: Denoiser network architecture matches configuration
        self.denoiser = self._create_denoiser(config)
        
        # Create optimizer
        self.optimizer = None
        self.lr_scheduler = None
        
        # Setup training state
        self.training_step = 0
        self.adaptation_step = 0
        self.eval_metrics = {}
        
        # Initialize trajectory buffer
        # TEST: Trajectory buffer initialized with correct capacity
        self.trajectory_buffer = TrajectoryBuffer(
            capacity=config.get("buffer_capacity", 10000)
        )
        
        # Register hooks for adaptation
        self.adaptation_hooks = []
    
    def _validate_config(self, config):
        """
        Validate configuration object.
        
        Args:
            config: Configuration object to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = [
            "model_type",
            "embedding_dim",
            "hidden_dim",
            "num_layers",
            "num_diffusion_steps"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")
    
    def _create_noise_schedule(self, schedule_type):
        """
        Create noise schedule for diffusion process.
        
        Args:
            schedule_type: Type of noise schedule ("linear", "cosine", "custom")
            
        Returns:
            NoiseSchedule: Initialized noise schedule
        """
        if schedule_type == "linear":
            return LinearNoiseSchedule(self.num_diffusion_steps)
        elif schedule_type == "cosine":
            return CosineNoiseSchedule(self.num_diffusion_steps)
        elif schedule_type == "custom":
            return CustomNoiseSchedule(
                self.config.get("custom_schedule_values", []),
                self.num_diffusion_steps
            )
        else:
            raise ValueError(f"Unsupported noise schedule type: {schedule_type}")
    
    def _create_denoiser(self, config):
        """
        Create denoiser network.
        
        Args:
            config: Configuration object
            
        Returns:
            DenoiserNetwork: Initialized denoiser network
        """
        # TEST: Denoiser is correctly initialized for different model types
        if self.model_type == "conditional":
            return ConditionalDenoiser(
                embedding_dim=config.get("embedding_dim"),
                hidden_dim=config.get("hidden_dim"),
                num_layers=config.get("num_layers"),
                num_heads=config.get("num_heads", 8),
                dropout=config.get("dropout", 0.1)
            )
        else:
            return UnconditionalDenoiser(
                embedding_dim=config.get("embedding_dim"),
                hidden_dim=config.get("hidden_dim"),
                num_layers=config.get("num_layers"),
                num_heads=config.get("num_heads", 8),
                dropout=config.get("dropout", 0.1)
def setup_training(self, optimizer_config=None):
        """
        Set up training state with optimizer and learning rate scheduler.
        
        Args:
            optimizer_config: Configuration for optimizer
                             {
                               "optimizer_type": "adam" | "adamw",
                               "learning_rate": float,
                               "weight_decay": float,
                               "scheduler_type": "cosine" | "linear" | "constant",
                               "scheduler_steps": int,
                               "warmup_steps": int
                             }
        """
        if optimizer_config is None:
            optimizer_config = {
                "optimizer_type": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 1e-6,
                "scheduler_type": "cosine",
                "scheduler_steps": 100000,
                "warmup_steps": 10000
            }
        
        # Create optimizer
        optim_cls = torch.optim.AdamW if optimizer_config.get("optimizer_type") == "adamw" else torch.optim.Adam
        self.optimizer = optim_cls(
            self.denoiser.parameters(),
            lr=optimizer_config.get("learning_rate", 1e-4),
            weight_decay=optimizer_config.get("weight_decay", 1e-6)
        )
        
        # Create learning rate scheduler
        # TEST: Learning rate scheduler correctly initialized based on type
        scheduler_type = optimizer_config.get("scheduler_type", "cosine")
        scheduler_steps = optimizer_config.get("scheduler_steps", 100000)
        warmup_steps = optimizer_config.get("warmup_steps", 10000)
        
        if scheduler_type == "cosine":
            self.lr_scheduler = CosineSchedulerWithWarmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=scheduler_steps
            )
        elif scheduler_type == "linear":
            self.lr_scheduler = LinearSchedulerWithWarmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=scheduler_steps
            )
        else:
            self.lr_scheduler = None
    
    def forward_diffusion(self, x_0, noise_level):
        """
        Apply forward diffusion to add noise to data.
        
        Args:
            x_0: Original clean data
            noise_level: Amount of noise to add (0.0 to 1.0)
            
        Returns:
            x_t: Noisy data
            noise: Added noise
        """
        # TEST: Forward diffusion correctly adds controlled noise
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(1 - noise_level**2) * x_0 + noise_level * noise
        return x_t, noise
    
    def single_denoising_step(self, x_t, t, condition=None):
        """
        Perform a single denoising step.
        
        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            condition: Optional conditioning information
            
        Returns:
            x_t_minus_1: Data with less noise at timestep t-1
        """
        # Calculate noise level from schedule
        noise_level = self.noise_schedule.get_noise_level(t)
        next_noise_level = self.noise_schedule.get_noise_level(t-1)
        
        # Predict noise using denoiser
        # TEST: Denoiser correctly predicts noise component
        if condition is not None and self.model_type == "conditional":
            predicted_noise = self.denoiser(x_t, noise_level, condition)
        else:
            predicted_noise = self.denoiser(x_t, noise_level)
        
        # Apply denoising formula
        # x_{t-1} = (x_t - predicted_noise * (1-αt)/sqrt(1-ᾱt)) / sqrt(αt)
        # where αt is the noise schedule parameter at timestep t
        coefficient = (1 - next_noise_level**2) / (1 - noise_level**2)
        coefficient = coefficient.sqrt()
        
        x_t_minus_1 = (x_t - predicted_noise * (1 - noise_level**2).sqrt() / (1 - noise_level**2)) / noise_level
        x_t_minus_1 = next_noise_level * x_t_minus_1 + (1 - next_noise_level**2).sqrt() * predicted_noise
        
        return x_t_minus_1
    
    def generate(self, shape, condition=None, use_guidance=False, guidance_scale=1.0, reward_fn=None):
        """
        Generate new data using the diffusion model.
        
        Args:
            shape: Shape of data to generate
            condition: Optional conditioning information
            use_guidance: Whether to use guidance during generation
            guidance_scale: Scale factor for guidance
            reward_fn: Optional reward function for guidance
            
        Returns:
            x_0: Generated data
            trajectory: Diffusion trajectory information
        """
        # Start from pure noise
        # TEST: Generation starts from appropriate noise distribution
        x_t = torch.randn(shape, device=self.device)
        trajectory = {"steps": []}
        
        # Iterate through diffusion steps in reverse
        for t in range(self.num_diffusion_steps, 0, -1):
            # Apply guidance if requested and if reward function is provided
            # TEST: Guidance correctly influences generation when enabled
            if use_guidance and reward_fn is not None:
                with torch.enable_grad():
                    x_t_tensor = x_t.detach().requires_grad_(True)
                    reward_value = reward_fn(x_t_tensor)
                    reward_grad = torch.autograd.grad(reward_value, x_t_tensor)[0]
                    guidance = guidance_scale * reward_grad
                    x_t = x_t + guidance
            
def train_step(self, x_0, condition=None):
        """
        Perform a single training step on clean data.
        
        Args:
            x_0: Clean data
            condition: Optional conditioning information
            
        Returns:
            loss: Training loss value
        """
        # Sample timestep
        # TEST: Random timestep sampling covers the full range
        t = torch.randint(1, self.num_diffusion_steps + 1, (x_0.shape[0],), device=self.device)
        
        # Get noise level from schedule
        noise_level = self.noise_schedule.get_noise_level(t).to(self.device)
        
        # Add noise to data
        # TEST: Noise addition matches the forward process
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(1 - noise_level.view(-1, 1, 1, 1)**2) * x_0 + noise_level.view(-1, 1, 1, 1) * noise
        
        # Predict noise using denoiser
        if condition is not None and self.model_type == "conditional":
            predicted_noise = self.denoiser(x_t, noise_level, condition)
        else:
            predicted_noise = self.denoiser(x_t, noise_level)
        
        # Calculate loss (mean squared error between actual and predicted noise)
        # TEST: Loss correctly calculates MSE between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)
        
        # Update training state
        self.training_step += 1
        
        return loss
    
    def training_loop(self, dataloader, num_epochs, condition_fn=None, eval_fn=None):
        """
        Run training loop for specified number of epochs.
        
        Args:
            dataloader: DataLoader providing training data
            num_epochs: Number of epochs to train
            condition_fn: Function to extract condition from data
            eval_fn: Function for evaluation during training
            
        Returns:
            metrics: Dictionary of training metrics
        """
        # Ensure model is in training mode
        self.denoiser.train()
        
        metrics = {
            "epoch_losses": [],
            "eval_metrics": [],
            "best_eval_score": float('-inf'),
            "best_model_state": None
        }
        
        # Training loop
        # TEST: Training progresses through all epochs and samples
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch in dataloader:
                # Extract data and optional condition
                if isinstance(batch, tuple) and len(batch) == 2:
                    x_0, cond_data = batch
                    condition = condition_fn(cond_data) if condition_fn else cond_data
                else:
                    x_0 = batch
                    condition = None
                
                # Move data to device
                x_0 = x_0.to(self.device)
                if condition is not None:
                    condition = condition.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass and loss calculation
                # TEST: Loss is calculated correctly for each batch
                loss = self.train_step(x_0, condition)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Update learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Record loss
                epoch_losses.append(loss.item())
            
            # Calculate average epoch loss
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            metrics["epoch_losses"].append(avg_epoch_loss)
def adapt(self, task, num_steps, reward_fn, adaptation_config=None):
        """
        Adapt model to a new task using reward gradient guidance.
        
        Args:
            task: Task description or embedding
            num_steps: Number of adaptation steps
            reward_fn: Reward function for task
            adaptation_config: Configuration for adaptation process
                              {
                                "learning_rate": float,
                                "guidance_scale": float,
                                "buffer_sampling_ratio": float,
                                "generation_batch_size": int
                              }
        
        Returns:
            adaptation_metrics: Dictionary of adaptation metrics
        """
        if adaptation_config is None:
            adaptation_config = {
                "learning_rate": 5e-5,
                "guidance_scale": 0.1,
                "buffer_sampling_ratio": 0.5,
                "generation_batch_size": 16
            }
        
        # Setup optimizer for adaptation
        adapt_optimizer = torch.optim.Adam(
            self.denoiser.parameters(),
            lr=adaptation_config.get("learning_rate", 5e-5)
        )
        
        metrics = {
            "reward_values": [],
            "loss_values": []
        }
        
        # Adaptation loop
        # TEST: Adaptation improves performance on target task
        for step in range(num_steps):
            # Generate samples for the task
            batch_size = adaptation_config.get("generation_batch_size", 16)
            shape = (batch_size,) + self._get_data_shape()
            
            # Mix generated samples with buffer samples
            if step > 0 and self.trajectory_buffer.size() > 0:
                buffer_ratio = adaptation_config.get("buffer_sampling_ratio", 0.5)
                buffer_samples = self.trajectory_buffer.sample(
                    int(batch_size * buffer_ratio)
                )
                gen_samples_count = batch_size - len(buffer_samples)
                
                # Generate new samples
                if gen_samples_count > 0:
                    gen_shape = (gen_samples_count,) + self._get_data_shape()
                    gen_samples, _ = self.generate(gen_shape, task, True, 
                                               adaptation_config.get("guidance_scale", 0.1),
                                               reward_fn)
                    
                    # Combine buffer and generated samples
                    all_samples = torch.cat([buffer_samples, gen_samples], dim=0)
                else:
                    all_samples = buffer_samples
            else:
                # Generate samples with reward guidance
                all_samples, _ = self.generate(shape, task, True, 
                                           adaptation_config.get("guidance_scale", 0.1),
                                           reward_fn)
            
            # Evaluate rewards for each sample
            with torch.no_grad():
                rewards = torch.stack([reward_fn(sample) for sample in all_samples])
                metrics["reward_values"].append(rewards.mean().item())
            
            # Add high-reward samples to buffer
            # TEST: Trajectory buffer correctly stores high-quality samples
            if rewards.numel() > 0:
                reward_threshold = rewards.mean() + rewards.std()
                high_reward_indices = (rewards >= reward_threshold).nonzero().squeeze(1)
                
                for idx in high_reward_indices:
                    self.trajectory_buffer.add(all_samples[idx], rewards[idx].item())
            
            # Train on mixed data
            adapt_optimizer.zero_grad()
            loss = self.train_step(all_samples)
            loss.backward()
            adapt_optimizer.step()
            
            metrics["loss_values"].append(loss.item())
            self.adaptation_step += 1
            
            # Run adaptation hooks
            self._run_adaptation_hooks(step, metrics, all_samples, rewards)
        
        return metrics
    
    def _run_adaptation_hooks(self, step, metrics, samples, rewards):
        """
        Run registered adaptation hooks.
        
        Args:
            step: Current adaptation step
            metrics: Current metrics
            samples: Current batch of samples
            rewards: Rewards for current samples
        """
        for hook in self.adaptation_hooks:
            hook(self, step, metrics, samples, rewards)
    
    def register_adaptation_hook(self, hook):
        """
        Register a hook to be called during adaptation.
        
        Args:
            hook: Function to be called during adaptation
        """
        self.adaptation_hooks.append(hook)
    
    def _get_data_shape(self):
        """
        Get shape of data for generation.
        
        Returns:
            tuple: Shape of data (excluding batch dimension)
        """
        # This should be implemented by subclasses for specific data types
        raise NotImplementedError("Subclasses must implement _get_data_shape")
    
    def _log_training_progress(self, epoch, loss, metrics):
        """
        Log training progress.
        
        Args:
            epoch: Current epoch
            loss: Current loss
            metrics: Current metrics
        """
        # Implement logging logic (e.g. to console, tensorboard, etc.)
        pass
    
    def get_state_dict(self):
        """
        Get state dictionary for model serialization.
        
        Returns:
            dict: State dictionary
        """
        return {
            "config": self.config,
            "model_type": self.model_type,
            "num_diffusion_steps": self.num_diffusion_steps,
            "denoiser": self.denoiser.state_dict(),
            "training_step": self.training_step,
            "adaptation_step": self.adaptation_step
        }
    
    def load_state_dict(self, state_dict):
        """
        Load state dictionary for model deserialization.
        
        Args:
            state_dict: State dictionary
        """
        self.config = state_dict["config"]
        self.model_type = state_dict["model_type"]
        self.num_diffusion_steps = state_dict["num_diffusion_steps"]
        self.denoiser.load_state_dict(state_dict["denoiser"])
        self.training_step = state_dict["training_step"]
        self.adaptation_step = state_dict["adaptation_step"]
```

## NoiseSchedule Classes

```python
class NoiseSchedule:
    """Base class for noise schedules used in diffusion models."""
    
    def __init__(self, num_steps):
        """
        Initialize noise schedule.
        
        Args:
            num_steps: Number of diffusion steps
        """
        self.num_steps = num_steps
    
    def get_noise_level(self, timestep):
        """
        Get noise level for specified timestep.
        
        Args:
            timestep: Current timestep
            
        Returns:
            float: Noise level (between 0 and 1)
        """
        raise NotImplementedError("Subclasses must implement get_noise_level")


class LinearNoiseSchedule(NoiseSchedule):
    """Linear noise schedule."""
    
    def __init__(self, num_steps, beta_start=1e-4, beta_end=0.02):
        """
def get_noise_level(self, timestep):
        """
        Get noise level for specified timestep.
        
        Args:
            timestep: Current timestep
            
        Returns:
            float: Noise level (between 0 and 1)
        """
        if isinstance(timestep, int):
            return self.sqrt_one_minus_alphas_cumprod[timestep-1]
        else:
            # Handle batched timesteps
            return self.sqrt_one_minus_alphas_cumprod[timestep-1]


class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule for improved sample quality."""
    
    def __init__(self, num_steps, s=0.008):
        """
        Initialize cosine noise schedule.
        
        Args:
            num_steps: Number of diffusion steps
            s: Offset parameter for schedule
        """
        super().__init__(num_steps)
        self.s = s
        
        # Pre-compute cosine schedule
        # TEST: Cosine schedule follows correct curve with specified parameters
        steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
        alpha_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        self.alphas_cumprod = alpha_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def get_noise_level(self, timestep):
        """
        Get noise level for specified timestep.
        
        Args:
            timestep: Current timestep
            
        Returns:
            float: Noise level (between 0 and 1)
        """
        if isinstance(timestep, int):
            return self.sqrt_one_minus_alphas_cumprod[timestep-1]
        else:
            # Handle batched timesteps
            return self.sqrt_one_minus_alphas_cumprod[timestep-1]


class CustomNoiseSchedule(NoiseSchedule):
    """Custom noise schedule with user-defined values."""
    
    def __init__(self, values, num_steps):
        """
        Initialize custom noise schedule.
        
        Args:
            values: List of noise values (will be interpolated if length != num_steps)
            num_steps: Number of diffusion steps
        """
        super().__init__(num_steps)
        
        # Interpolate values if necessary
        # TEST: Custom schedule correctly interpolates provided values
        if len(values) != num_steps:
            indices = torch.linspace(0, len(values) - 1, num_steps)
            values_tensor = torch.tensor(values, dtype=torch.float32)
            interp_values = torch.zeros(num_steps, dtype=torch.float32)
            
            for i in range(num_steps):
                idx = indices[i]
                idx_low = int(idx)
                idx_high = min(idx_low + 1, len(values) - 1)
                weight = idx - idx_low
                interp_values[i] = (1 - weight) * values_tensor[idx_low] + weight * values_tensor[idx_high]
            
            self.noise_values = interp_values
        else:
            self.noise_values = torch.tensor(values, dtype=torch.float32)
        
        # Ensure values are between 0 and 1
        self.noise_values = torch.clamp(self.noise_values, 0.0, 1.0)
    
    def get_noise_level(self, timestep):
        """
        Get noise level for specified timestep.
        
        Args:
            timestep: Current timestep
            
        Returns:
            float: Noise level (between 0 and 1)
        """
        if isinstance(timestep, int):
            return self.noise_values[timestep-1]
        else:
            # Handle batched timesteps
            return self.noise_values[timestep-1]
```

## DenoiserNetwork Classes

```python
class DenoiserNetwork(nn.Module):
    """Base class for denoiser networks used in diffusion models."""
    
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_heads, dropout):
        """
        Initialize denoiser network.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
    
    def forward(self, x, noise_level, condition=None):
        """
        Forward pass through denoiser network.
        
        Args:
            x: Input data
            noise_level: Current noise level
            condition: Optional conditioning information
            
        Returns:
            Predicted noise
        """
        raise NotImplementedError("Subclasses must implement forward")


class UnconditionalDenoiser(DenoiserNetwork):
    """Denoiser network for unconditional generation."""
    
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_heads, dropout):
        """
        Initialize unconditional denoiser.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__(embedding_dim, hidden_dim, num_layers, num_heads, dropout)
        
        # Noise level embedding
        # TEST: Noise embedding correctly transforms scalar noise level to vector
        self.noise_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encoder layers
        self.encoder = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Input projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, x, noise_level, condition=None):
        """
        Forward pass through unconditional denoiser.
        
        Args:
            x: Input data [batch_size, *data_dims, embedding_dim]
            noise_level: Current noise level [batch_size]
            condition: Ignored for unconditional model
            
        Returns:
            Predicted noise [batch_size, *data_dims, embedding_dim]
        """
        # Flatten data dimensions for transformer
        batch_size = x.shape[0]
        data_dims = x.shape[1:-1]
        seq_len = np.prod(data_dims)
        
        x_flat = x.view(batch_size, seq_len, self.embedding_dim)
        
        # Project input to hidden dimension
        h = self.input_proj(x_flat)
        
        # Embed noise level
        # TEST: Noise embedding correctly influences model behavior at different noise levels
        noise_emb = self.noise_embedding(noise_level.view(-1, 1))
        
        # Add noise embedding to input
        h = h + noise_emb.unsqueeze(1)
        
        # Apply transformer blocks
        for layer in self.encoder:
            h = layer(h)
class ConditionalDenoiser(DenoiserNetwork):
    """Denoiser network that supports conditional generation."""
    
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_heads, dropout):
        """
        Initialize conditional denoiser.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__(embedding_dim, hidden_dim, num_layers, num_heads, dropout)
        
        # Noise level embedding
        self.noise_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding
        # TEST: Condition embedding correctly processes different condition types
        self.condition_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention layers
        # TEST: Cross-attention correctly incorporates condition information
        self.cross_attention = nn.ModuleList([
            CrossAttentionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Self-attention layers
        self.self_attention = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Input projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, x, noise_level, condition):
        """
        Forward pass through conditional denoiser.
        
        Args:
            x: Input data [batch_size, *data_dims, embedding_dim]
            noise_level: Current noise level [batch_size]
            condition: Conditioning information [batch_size, cond_dims, embedding_dim]
            
        Returns:
            Predicted noise [batch_size, *data_dims, embedding_dim]
        """
        # Flatten data dimensions for transformer
        batch_size = x.shape[0]
        data_dims = x.shape[1:-1]
        seq_len = np.prod(data_dims)
        
        x_flat = x.view(batch_size, seq_len, self.embedding_dim)
        
        # Project input to hidden dimension
        h = self.input_proj(x_flat)
        
        # Embed noise level
        noise_emb = self.noise_embedding(noise_level.view(-1, 1))
        
        # Add noise embedding to input
        h = h + noise_emb.unsqueeze(1)
        
        # Process condition
        # TEST: Condition processing adapts to different condition shapes
        if condition is not None:
            # Flatten condition dimensions if needed
            cond_dims = condition.shape[1:-1]
            cond_seq_len = np.prod(cond_dims)
            cond_flat = condition.view(batch_size, cond_seq_len, self.embedding_dim)
            
            # Project condition to hidden dimension
            cond_emb = self.condition_embedding(cond_flat)
            
            # Apply interleaved self-attention and cross-attention
            for self_attn, cross_attn in zip(self.self_attention, self.cross_attention):
                h = self_attn(h)
                h = cross_attn(h, cond_emb)
        else:
            # Apply only self-attention if no condition is provided
            for layer in self.self_attention:
                h = layer(h)
        
        # Project back to embedding dimension
        output = self.output_proj(h)
        
        # Reshape back to input shape
        output = output.view(batch_size, *data_dims, self.embedding_dim)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with self-attention."""
    
    def __init__(self, dim, num_heads, dropout):
        """
        Initialize transformer block.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x)
        )
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for conditioning."""
    
    def __init__(self, dim, num_heads, dropout):
        """
        Initialize cross-attention block.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context):
        """
        Forward pass through cross-attention block.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            context: Context tensor for conditioning [batch_size, context_len, dim]
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Cross-attention with residual connection
        attn_out, _ = self.attention(
            query=self.norm1(x),
            key=self.norm2(context),
            value=self.norm2(context)
        )
        x = x + self.dropout(attn_out)
        
        return x


class TrajectoryBuffer:
    """Buffer for storing and sampling high-quality trajectories."""
    
    def __init__(self, capacity):
        """
        Initialize trajectory buffer.
        
        Args:
            capacity: Maximum number of trajectories to store
        """
        self.capacity = capacity
        self.trajectories = []
        self.rewards = []
        self.priorities = []
    
    def add(self, trajectory, reward):
        """
        Add trajectory to buffer.
        
        Args:
            trajectory: Trajectory data
            reward: Reward value for trajectory
        """
        # If buffer is full, remove lowest priority trajectory
        if len(self.trajectories) >= self.capacity:
            idx = np.argmin(self.priorities)
            self.trajectories.pop(idx)
            self.rewards.pop(idx)
            self.priorities.pop(idx)
        
        # Add new trajectory
        self.trajectories.append(trajectory)
        self.rewards.append(reward)
        
        # Calculate priority based on reward
        priority = np.exp(reward)
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        """
        Sample trajectories from buffer.
        
        Args:
            batch_size: Number of trajectories to sample
            
        Returns:
            List of sampled trajectories
        """
        # Return empty list if buffer is empty
        if len(self.trajectories) == 0:
            return []
        
        # Ensure batch_size is not larger than buffer size
        batch_size = min(batch_size, len(self.trajectories))
        
        # Calculate sampling probabilities
        probs = np.array(self.priorities) / sum(self.priorities)
        
        # Sample indices
        indices = np.random.choice(
            len(self.trajectories),
            size=batch_size,
            replace=False,
            p=probs
        )
        
        # Return sampled trajectories
        return [self.trajectories[idx] for idx in indices]
    
    def size(self):
        """
        Get current buffer size.
        
        Returns:
            Number of trajectories in buffer
        """
        return len(self.trajectories)
        
        # Project back to embedding dimension
        output = self.output_proj(h)
        
        # Reshape back to input shape
        output = output.view(batch_size, *data_dims, self.embedding_dim)
        
        return output
        Initialize linear noise schedule.
        
        Args:
            num_steps: Number of diffusion steps
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        super().__init__(num_steps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Pre-compute noise levels for all timesteps
        # TEST: Linear schedule increases monotonically from start to end
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
            
            # Run evaluation if provided
            if eval_fn is not None:
                # TEST: Evaluation function correctly assesses model performance
                eval_score = eval_fn(self)
                metrics["eval_metrics"].append(eval_score)
                
                # Save best model
                if eval_score > metrics["best_eval_score"]:
                    metrics["best_eval_score"] = eval_score
                    metrics["best_model_state"] = self.get_state_dict()
            
            # Log progress
            self._log_training_progress(epoch, avg_epoch_loss, metrics)
        
        return metrics
            # Apply single denoising step
            # TEST: Each denoising step reduces noise appropriately
            x_t = self.single_denoising_step(x_t, t, condition)
            
            # Record trajectory information
            if t % 10 == 0 or t == 1:  # Store every 10th step to save memory
                trajectory["steps"].append({
                    "timestep": t, 
                    "data": x_t.detach().cpu(), 
                    "noise_level": self.noise_schedule.get_noise_level(t).item()
                })
        
        # Final cleanup step
        # TEST: Final generated result has no remaining noise
        x_0 = x_t
        trajectory["final"] = x_0.detach().cpu()
        
        return x_0, trajectory
            )