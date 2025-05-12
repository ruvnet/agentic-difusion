# Adaptation Mechanism Pseudocode

This document outlines the pseudocode for the adaptation mechanisms that enable the Agentic Diffusion system to adapt to new tasks and improve over time.

## AdaptationMechanism Class

```python
class AdaptationMechanism:
    """
    Base class for adaptation mechanisms in Agentic Diffusion.
    
    This provides the foundation for different adaptation strategies
    to enhance diffusion models for new tasks.
    """
    
    def __init__(self, model, config=None):
        """
        Initialize adaptation mechanism.
        
        Args:
            model: DiffusionModel instance to adapt
            config: Configuration object with adaptation parameters
                   {
                     "adaptation_type": "gradient" | "memory" | "hybrid",
                     "learning_rate": float,
                     "max_steps": int,
                     "evaluation_frequency": int,
                     "early_stopping_patience": int,
                     "min_improvement": float
                   }
        """
        self.model = model
        self.config = config or {
            "adaptation_type": "gradient",
            "learning_rate": 1e-4,
            "max_steps": 1000,
            "evaluation_frequency": 50,
            "early_stopping_patience": 5,
            "min_improvement": 0.01
        }
        
        self.adaptation_history = []
        self.best_performance = float('-inf')
        self.patience_counter = 0
    
    def adapt(self, task, reward_fn, initial_samples=None):
        """
        Adapt the model to a new task.
        
        Args:
            task: Task description or embedding
            reward_fn: Reward function for task
            initial_samples: Optional initial samples for task
            
        Returns:
            metrics: Dictionary of adaptation metrics
        """
        raise NotImplementedError("Subclasses must implement adapt")
    
    def evaluate(self, task, reward_fn, num_samples=32):
        """
        Evaluate the model on a task.
        
        Args:
            task: Task description or embedding
            reward_fn: Reward function for task
            num_samples: Number of samples to generate for evaluation
            
        Returns:
            score: Evaluation score
        """
        # Generate samples
        # TEST: Evaluation generates appropriate number of samples
        batch_size = min(num_samples, 16)  # Generate in batches if needed
        all_rewards = []
        
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            shape = (current_batch_size,) + self.model._get_data_shape()
            
            # Generate samples
            samples, _ = self.model.generate(
                shape=shape,
                condition=task,
                use_guidance=False  # No guidance during evaluation
            )
            
            # Evaluate samples
            with torch.no_grad():
                rewards = [reward_fn(sample).item() for sample in samples]
                all_rewards.extend(rewards)
        
        # Calculate evaluation metrics
        # TEST: Evaluation metrics accurately reflect quality of generated samples
        avg_reward = sum(all_rewards) / len(all_rewards)
        reward_std = np.std(all_rewards)
        success_rate = len([r for r in all_rewards if r > 0.5]) / len(all_rewards)
        
        metrics = {
            "avg_reward": avg_reward,
            "reward_std": reward_std,
            "success_rate": success_rate,
            "min_reward": min(all_rewards),
            "max_reward": max(all_rewards)
        }
        
        return metrics
    
    def log_adaptation_step(self, step, metrics):
        """
        Log adaptation step information.
        
        Args:
            step: Current step number
            metrics: Metrics for current step
        """
        self.adaptation_history.append({
            "step": step,
            "metrics": metrics
        })
        
        # Check for best performance
        current_performance = metrics.get("avg_reward", 0)
        if current_performance > self.best_performance + self.config.get("min_improvement", 0.01):
            self.best_performance = current_performance
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False
    
    def should_stop_early(self):
        """
        Check if adaptation should stop early.
        
        Returns:
            bool: Whether to stop early
        """
        patience = self.config.get("early_stopping_patience", 5)
        return self.patience_counter >= patience


class GradientBasedAdaptation(AdaptationMechanism):
    """
    Adaptation mechanism based on reward gradient guidance.
    
    Adapts the diffusion model using gradients of a reward function
    to guide generation toward high-reward outputs.
    """
    
    def __init__(self, model, config=None):
        """
        Initialize gradient-based adaptation.
        
        Args:
            model: DiffusionModel instance to adapt
            config: Configuration object with adaptation parameters
        """
        super().__init__(model, config)
        
        default_gradient_config = {
            "guidance_scale": 0.1,
            "buffer_sampling_ratio": 0.5,
            "generation_batch_size": 16,
            "optimization_batch_size": 8
        }
        
        # Merge with provided config
        if config:
            for key, value in default_gradient_config.items():
                if key not in config:
                    config[key] = value
        else:
            self.config.update(default_gradient_config)
    
    def adapt(self, task, reward_fn, initial_samples=None):
        """
        Adapt model using reward gradient guidance.
        
        Args:
            task: Task description or embedding
            reward_fn: Reward function for task
            initial_samples: Optional initial samples for task
            
        Returns:
            metrics: Dictionary of adaptation metrics
        """
        # Initialize optimizer
        # TEST: Optimizer correctly initialized with appropriate learning rate
        optimizer = torch.optim.Adam(
            self.model.denoiser.parameters(),
            lr=self.config.get("learning_rate", 1e-4)
        )
        
        # Initialize metrics tracking
        metrics = {
            "step_rewards": [],
            "evaluation_metrics": [],
            "losses": []
        }
        
        # Initialize trajectory buffer with initial samples if provided
        # TEST: Initial samples correctly added to trajectory buffer
        if initial_samples is not None:
            for sample in initial_samples:
                with torch.no_grad():
                    reward = reward_fn(sample)
                self.model.trajectory_buffer.add(sample, reward.item())
        
        # Adaptation loop
        # TEST: Adaptation loop improves performance over iterations
        max_steps = self.config.get("max_steps", 1000)
        for step in range(max_steps):
            # Generate samples for the task
            batch_size = self.config.get("generation_batch_size", 16)
            shape = (batch_size,) + self.model._get_data_shape()
            
            # Mix generated samples with buffer samples
            if step > 0 and self.model.trajectory_buffer.size() > 0:
                buffer_ratio = self.config.get("buffer_sampling_ratio", 0.5)
                buffer_samples = self.model.trajectory_buffer.sample(
                    int(batch_size * buffer_ratio)
                )
                gen_samples_count = batch_size - len(buffer_samples)
                
                # Generate new samples with guidance
                if gen_samples_count > 0:
                    gen_shape = (gen_samples_count,) + self.model._get_data_shape()
                    gen_samples, _ = self.model.generate(
                        shape=gen_shape, 
                        condition=task, 
                        use_guidance=True,
                        guidance_scale=self.config.get("guidance_scale", 0.1),
                        reward_fn=reward_fn
                    )
                    
                    # Combine buffer and generated samples
                    all_samples = torch.cat([buffer_samples, gen_samples], dim=0)
                else:
                    all_samples = buffer_samples
            else:
                # Generate samples with reward guidance
                all_samples, _ = self.model.generate(
                    shape=shape, 
                    condition=task, 
                    use_guidance=True,
                    guidance_scale=self.config.get("guidance_scale", 0.1),
                    reward_fn=reward_fn
                )
            
            # Evaluate rewards for each sample
            # TEST: Reward calculation correctly evaluates sample quality
            with torch.no_grad():
                rewards = torch.stack([reward_fn(sample) for sample in all_samples])
                step_reward_avg = rewards.mean().item()
                metrics["step_rewards"].append(step_reward_avg)
            
            # Add high-reward samples to buffer
            if rewards.numel() > 0:
                reward_threshold = rewards.mean() + rewards.std()
                high_reward_indices = (rewards >= reward_threshold).nonzero().squeeze(1)
                
                for idx in high_reward_indices:
                    self.model.trajectory_buffer.add(all_samples[idx], rewards[idx].item())
            
            # Optimization step
            # Process in smaller batches if needed
            optim_batch_size = self.config.get("optimization_batch_size", 8)
            avg_loss = 0.0
            
            for i in range(0, len(all_samples), optim_batch_size):
                batch = all_samples[i:i+optim_batch_size]
                
                optimizer.zero_grad()
                loss = self.model.train_step(batch, task)
                loss.backward()
                optimizer.step()
                
                avg_loss += loss.item() * len(batch)
            
            avg_loss /= len(all_samples)
            metrics["losses"].append(avg_loss)
            
            # Periodic evaluation
            if step % self.config.get("evaluation_frequency", 50) == 0:
                eval_metrics = self.evaluate(task, reward_fn)
                metrics["evaluation_metrics"].append(eval_metrics)
                
                # Log progress
                improved = self.log_adaptation_step(step, eval_metrics)
                
                # Early stopping check
                if self.should_stop_early():
                    print(f"Early stopping at step {step}")
                    break
        
        return metrics


class MemoryBasedAdaptation(AdaptationMechanism):
    """
    Adaptation mechanism based on memory and experience replay.
    
    Adapts the diffusion model by collecting and utilizing a memory
    of high-quality samples for various tasks.
    """
    
    def __init__(self, model, config=None):
        """
        Initialize memory-based adaptation.
        
        Args:
            model: DiffusionModel instance to adapt
            config: Configuration object with adaptation parameters
        """
        super().__init__(model, config)
        
        default_memory_config = {
            "initial_exploration_steps": 100,
            "exploitation_ratio": 0.8,
            "memory_sampling_temperature": 2.0,
            "dynamic_task_weighting": True
        }
        
        # Merge with provided config
        if config:
            for key, value in default_memory_config.items():
                if key not in config:
                    config[key] = value
        else:
            self.config.update(default_memory_config)
        
        # Initialize task-specific memories
        self.task_memories = {}
        self.task_performance = {}
    
    def adapt(self, task, reward_fn, initial_samples=None):
        """
        Adapt model using memory-based approach.
        
        Args:
            task: Task description or embedding
            reward_fn: Reward function for task
            initial_samples: Optional initial samples for task
            
        Returns:
            metrics: Dictionary of adaptation metrics
        """
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            self.model.denoiser.parameters(),
            lr=self.config.get("learning_rate", 1e-4)
        )
        
        # Initialize metrics tracking
        metrics = {
            "step_rewards": [],
            "evaluation_metrics": [],
            "losses": [],
            "exploration_ratio": []
        }
        
        # Initialize task memory if not already present
        task_id = self._get_task_id(task)
        if task_id not in self.task_memories:
            self.task_memories[task_id] = []
            self.task_performance[task_id] = 0.0
        
        # Add initial samples to task memory if provided
        # TEST: Initial samples correctly added to task memory
        if initial_samples is not None:
            for sample in initial_samples:
                with torch.no_grad():
                    reward = reward_fn(sample)
                self.task_memories[task_id].append((sample, reward.item()))
                self.model.trajectory_buffer.add(sample, reward.item())
        
        # Adaptation loop
        max_steps = self.config.get("max_steps", 1000)
        for step in range(max_steps):
            # Calculate exploration ratio (starts high, decreases over time)
            exploration_ratio = max(
                0.1,
                1.0 - (step / self.config.get("initial_exploration_steps", 100))
            ) if step < self.config.get("initial_exploration_steps", 100) else 0.1
            metrics["exploration_ratio"].append(exploration_ratio)
            
            # Generate samples: mix exploration and exploitation
            batch_size = self.config.get("generation_batch_size", 16)
            exploitation_count = int(batch_size * (1 - exploration_ratio))
            exploration_count = batch_size - exploitation_count
            
            # Exploration: Generate new samples
            # TEST: Exploration-exploitation balance changes appropriately during adaptation
            if exploration_count > 0:
                explore_shape = (exploration_count,) + self.model._get_data_shape()
                explore_samples, _ = self.model.generate(
                    shape=explore_shape,
                    condition=task,
                    use_guidance=False  # Pure exploration
                )
            else:
                explore_samples = torch.tensor([])
            
            # Exploitation: Sample from task memory and related tasks
            if exploitation_count > 0 and (len(self.task_memories[task_id]) > 0 or len(self.model.trajectory_buffer) > 0):
                # Decide whether to use task-specific memory or general buffer
                use_task_memory = len(self.task_memories[task_id]) > 0 and random.random() < self.config.get("exploitation_ratio", 0.8)
                
                if use_task_memory:
                    # Sample from task memory using reward-weighted sampling
                    exploit_samples = self._sample_from_task_memory(
                        task_id, 
                        exploitation_count,
                        self.config.get("memory_sampling_temperature", 2.0)
                    )
                else:
                    # Sample from general trajectory buffer
                    exploit_samples = self.model.trajectory_buffer.sample(exploitation_count)
            else:
                exploit_samples = torch.tensor([])
            
            # Combine exploration and exploitation samples
            if len(explore_samples) > 0 and len(exploit_samples) > 0:
                all_samples = torch.cat([explore_samples, exploit_samples], dim=0)
            elif len(explore_samples) > 0:
                all_samples = explore_samples
            elif len(exploit_samples) > 0:
                all_samples = exploit_samples
            else:
                # Generate some samples if both are empty
                all_samples, _ = self.model.generate(
                    shape=(batch_size,) + self.model._get_data_shape(),
                    condition=task
                )
            
            # Evaluate rewards for each sample
            with torch.no_grad():
                rewards = torch.stack([reward_fn(sample) for sample in all_samples])
                step_reward_avg = rewards.mean().item()
                metrics["step_rewards"].append(step_reward_avg)
            
            # Update task memory and buffer with high-reward samples
            # TEST: High-reward samples correctly update task memory
            if rewards.numel() > 0:
                reward_threshold = rewards.mean()
                high_reward_indices = (rewards >= reward_threshold).nonzero().squeeze(1)
                
                for idx in high_reward_indices:
                    sample = all_samples[idx]
                    reward_val = rewards[idx].item()
                    
                    # Add to task memory
                    self.task_memories[task_id].append((sample, reward_val))
                    
                    # Add to trajectory buffer
                    self.model.trajectory_buffer.add(sample, reward_val)
            
            # Limit task memory size
            max_memory_size = self.config.get("max_memory_size", 1000)
            if len(self.task_memories[task_id]) > max_memory_size:
                # Sort by reward and keep top samples
                self.task_memories[task_id].sort(key=lambda x: x[1], reverse=True)
                self.task_memories[task_id] = self.task_memories[task_id][:max_memory_size]
            
            # Optimization step
            optimizer.zero_grad()
            loss = self.model.train_step(all_samples, task)
            loss.backward()
            optimizer.step()
            metrics["losses"].append(loss.item())
            
            # Periodic evaluation
            if step % self.config.get("evaluation_frequency", 50) == 0:
                eval_metrics = self.evaluate(task, reward_fn)
                metrics["evaluation_metrics"].append(eval_metrics)
                
                # Update task performance
                self.task_performance[task_id] = eval_metrics["avg_reward"]
                
                # Log progress
                improved = self.log_adaptation_step(step, eval_metrics)
                
                # Early stopping check
                if self.should_stop_early():
                    print(f"Early stopping at step {step}")
                    break
        
        return metrics
    
    def _get_task_id(self, task):
        """
        Get unique identifier for a task.
        
        Args:
            task: Task description or embedding
            
        Returns:
            task_id: Unique task identifier
        """
        if isinstance(task, str):
            return task
        elif isinstance(task, torch.Tensor):
            # Create a hash from the tensor
            return hash(task.cpu().numpy().tobytes())
        else:
            return str(task)
    
    def _sample_from_task_memory(self, task_id, count, temperature=2.0):
        """
        Sample from task memory with reward-weighted sampling.
        
        Args:
            task_id: Task identifier
            count: Number of samples to retrieve
            temperature: Sampling temperature (higher = more uniform)
            
        Returns:
            List of sampled tensors
        """
        if len(self.task_memories[task_id]) == 0:
            return []
        
        # Get samples and rewards
        samples, rewards = zip(*self.task_memories[task_id])
        
        # Convert rewards to sampling weights
        weights = np.array([max(0.1, r) ** (1.0 / temperature) for r in rewards])
        weights = weights / weights.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(samples),
            size=min(count, len(samples)),
            replace=False,
            p=weights
        )
        
        # Return sampled tensors
        return torch.stack([samples[i] for i in indices])