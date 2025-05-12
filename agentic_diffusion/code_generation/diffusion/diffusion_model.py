"""
CodeDiffusionModel implementation for code generation.

This module provides a diffusion model specialized for code generation,
adapting the core diffusion model framework to work with code tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

from agentic_diffusion.core.diffusion_model import DiffusionModel
from agentic_diffusion.core.noise_schedules import NoiseScheduler, CosineScheduler
from agentic_diffusion.code_generation.models.code_unet import CodeUNet
from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
from agentic_diffusion.code_generation.utils.diffusion_utils import categorical_diffusion


class CodeDiffusionModel(DiffusionModel):
    """
    Diffusion model specialized for code generation.
    
    This model adapts the diffusion framework to work with discrete code tokens
    and provides methods for generating code through the diffusion process.
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 256,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        noise_scheduler: Optional[NoiseScheduler] = None,
        num_timesteps: int = 1000,
        device: Optional[str] = None
    ):
        """
        Initialize the code diffusion model.
        
        Args:
            vocab_size: Size of the code token vocabulary
            max_seq_len: Maximum sequence length
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            noise_scheduler: Scheduler for noise levels
            num_timesteps: Number of diffusion timesteps
            device: Device to use for computation
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the UNet model for denoising
        self.model = CodeUNet(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            condition_dim=128  # Fixed dimension for specification conditioning
        )
        
        # Initialize noise scheduler
        if noise_scheduler is None:
            self.noise_scheduler = CosineScheduler(
                num_timesteps=num_timesteps,
                device=self.device
            )
        else:
            self.noise_scheduler = noise_scheduler
            
        # Special token for masking
        self.mask_token_id = vocab_size - 1
        
        # Initialize reward models for code quality assessment
        self.reward_models = self._initialize_reward_models()
        
        # Tokenizer cache
        self.tokenizers = {}
        
        # Training state
        self.is_trained = False
        
        self.logger.info(f"Initialized CodeDiffusionModel with vocab_size={vocab_size}, embedding_dim={embedding_dim}")
        
    def _initialize_reward_models(self) -> Dict[str, Any]:
        """
        Initialize reward models for code quality assessment.
        
        Returns:
            Dictionary of reward models
        """
        try:
            from agentic_diffusion.code_generation.rewards.syntax_reward import SyntaxReward
            from agentic_diffusion.code_generation.rewards.quality_reward import QualityReward
            from agentic_diffusion.code_generation.rewards.relevance_reward import RelevanceReward
            
            # Use correct reward models that match the API implementation
            reward_models = {
                "syntax": SyntaxReward(),
                "quality": QualityReward(),
                "relevance": RelevanceReward()
            }
            
            self.logger.info(f"Initialized {len(reward_models)} reward models for code quality assessment")
            return reward_models
        except ImportError as e:
            self.logger.warning(f"Could not import reward models: {e}. Code quality assessment will be limited.")
            return {}
    
    def _encode_specification(self, specification: str) -> torch.Tensor:
        """
        Encode a natural language specification into a conditioning tensor.
        
        Args:
            specification: Natural language specification
            
        Returns:
            Conditioning tensor for the diffusion model
        """
        # Create a consistent but unique embedding based on the specification text
        # This uses a simple bag-of-words approach weighted by character position
        words = specification.lower().split()
        if not words:
            return torch.zeros(1, 128, device=self.device)
        
        # Initialize an embedding vector
        embedding = torch.zeros(1, 128, device=self.device)
        
        # For each word, add a contribution to the embedding
        for i, word in enumerate(words):
            # Create a word hash that's stable for the same word
            word_hash = sum(ord(c) * (i+1) for i, c in enumerate(word))
            
            # Use the hash to create a direction in embedding space
            word_vector = torch.sin(torch.arange(128, device=self.device) * word_hash * 0.01)
            
            # Weight words differently based on position (later words have more weight)
            position_weight = 0.5 + 0.5 * (i / len(words))
            
            # Add to the embedding
            embedding += word_vector.unsqueeze(0) * position_weight
        
        # Normalize the embedding
        embedding = embedding / (embedding.norm() + 1e-8)
        
        return embedding
            
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for the diffusion model.
        
        Args:
            x: Token indices [batch_size, seq_len]
            t: Timestep values [batch_size]
            **kwargs: Additional arguments
            
        Returns:
            Token logits [batch_size, seq_len, vocab_size]
        """
        return self.model(x, t, **kwargs)
    
    def noising_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply noise to code tokens based on diffusion timesteps.
        
        Args:
            x_0: Original token indices [batch_size, seq_len]
            t: Timestep values [batch_size]
            noise: Optional pre-generated noise
            
        Returns:
            Noisy token indices [batch_size, seq_len]
        """
        # Use the utility function for categorical diffusion
        return categorical_diffusion(
            x_0=x_0, 
            t=t, 
            num_classes=self.vocab_size, 
            noise_schedule=self.noise_scheduler.alphas_cumprod
        )
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a training step on a batch of code tokens.
        
        Args:
            batch: Dictionary containing 'x' key with token indices
            
        Returns:
            Dictionary with loss values
        """
        # Extract tokens from batch
        x_0 = batch.get("x")
        if x_0 is None:
            raise ValueError("Batch must contain 'x' key with token indices")
        
        # Sample random timesteps
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=x_0.device)
        
        # Apply noise to tokens
        x_t = self.noising_process(x_0, t)
        
        # Get model predictions
        logits = self.forward(x_t, t, **batch.get("condition_kwargs", {}))
        
        # Compute loss (cross-entropy for token prediction)
        # Reshape logits and target for cross-entropy loss
        logits = logits.view(-1, self.vocab_size)
        target = x_0.view(-1)
        
        # Calculate loss
        loss = F.cross_entropy(logits, target)
        
        return {"loss": loss, "x_t": x_t}
    
    def sample(
        self,
        shape: Union[Tuple[int, ...], List[int]],
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Generate code samples from the diffusion model.
        
        Args:
            shape: Shape of samples to generate [batch_size, seq_len]
            condition: Optional conditioning tensor
            guidance_scale: Scale for classifier-free guidance
            temperature: Sampling temperature (lower = more deterministic)
            **kwargs: Additional sampling parameters
            
        Returns:
            List of token sequences
        """
        device = kwargs.get("device", self.device)
        batch_size, seq_len = shape
        
        # Start from random tokens
        x_t = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=device)
        
        # Iteratively denoise
        for t in range(self.noise_scheduler.num_timesteps - 1, -1, -1):
            # Create timestep tensor
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get model prediction
            with torch.no_grad():
                # Classifier-free guidance implementation
                if condition is not None and guidance_scale > 1.0:
                    # Predict with condition
                    logits_cond = self(x_t, timesteps, condition=condition)
                    
                    # Predict without condition
                    logits_uncond = self(x_t, timesteps, condition=None)
                    
                    # Apply guidance
                    logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
                else:
                    logits = self(x_t, timesteps, condition=condition)
            
            # Apply temperature
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                # Sample from the distribution
                next_x_t = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                next_x_t = next_x_t.view(batch_size, seq_len)
            else:
                # Greedy decoding
                next_x_t = torch.argmax(logits, dim=-1)
            
            # Alpha-blending old and new tokens based on timestep
            alpha = t / self.noise_scheduler.num_timesteps
            mask = (torch.rand(batch_size, seq_len, device=device) < alpha).long()
            x_t = mask * x_t + (1 - mask) * next_x_t
        
        # Return the final token sequences
        return x_t.cpu().tolist()
    
    def generate(
        self,
        specification: str,
        language: str,
        partial_code: Optional[str] = None,
        tokenizer: Optional[CodeTokenizer] = None,
        max_length: int = 512,
        guidance_scale: float = 1.5,
        temperature: float = 0.7,
        num_samples: int = 1
    ) -> str:
        """
        Generate code from a specification or partial code using the diffusion model.
        
        Args:
            specification: Text description of the code to generate
            language: Programming language to generate
            partial_code: Optional partial code to complete
            tokenizer: Code tokenizer
            max_length: Maximum code length to generate
            guidance_scale: Scale for classifier-free guidance
            temperature: Sampling temperature
            num_samples: Number of code samples to generate
            
        Returns:
            Generated code as a string
        """
        # Create a tokenizer if not provided
        if tokenizer is None:
            tokenizer = CodeTokenizer(language=language)
        
        # Encode the specification as conditioning
        condition = self._encode_specification(specification)
        
        # Initialize code sequence
        if partial_code:
            # Tokenize partial code
            code_tokens = tokenizer.tokenize(partial_code)
            # Pad or truncate to max_length
            if len(code_tokens) > max_length:
                code_tokens = code_tokens[:max_length]
                partial_length = max_length
            else:
                partial_length = len(code_tokens)
                # Pad with mask tokens
                code_tokens = code_tokens + ["<MASK>"] * (max_length - len(code_tokens))
            
            # Convert tokens to indices
            code_indices = self._tokens_to_indices(code_tokens, tokenizer)
            
            # Create tensor
            x_start = torch.tensor([code_indices], device=self.device)
            
            # Mask out partial code area during generation
            mask = torch.zeros_like(x_start, dtype=torch.bool)
            mask[:, :partial_length] = True  # Mask the partial code (don't change it)
        else:
            # Start with all mask tokens
            x_start = torch.full((1, max_length), self.mask_token_id, device=self.device)
            mask = None
        
        # Set up batched condition tensor
        batched_condition = condition.repeat(num_samples, 1)
        
        # Sample from the model with classifier-free guidance
        with torch.no_grad():
            # Shape input for batch processing
            x_t = x_start.repeat(num_samples, 1)
            
            # Prepare noise schedule
            timesteps = list(range(self.noise_scheduler.num_timesteps))[::-1]
            
            # Add noise to the input based on the noise schedule
            noise_level = 0.8  # How much noise to add (0.0 to 1.0)
            t = torch.full((num_samples,), int(noise_level * (len(timesteps) - 1)), device=self.device)
            x_t = self.noising_process(x_t, t)
            
            # Gradually remove noise using the reverse diffusion process
            for i in timesteps:
                # Current timestep
                t_tensor = torch.full((num_samples,), i, device=self.device)
                
                # Prepare model input
                model_input = x_t.clone()
                
                # Get model prediction (forward pass)
                pred = self.model(model_input, t_tensor, condition=batched_condition)
                
                # Apply classifier-free guidance if specified
                if guidance_scale > 1.0:
                    # Get unconditional prediction
                    uncond_pred = self.model(model_input, t_tensor)
                    # Apply guidance formula
                    guided_pred = uncond_pred + guidance_scale * (pred - uncond_pred)
                    pred = guided_pred
                
                # Sample from the predicted distribution
                if temperature > 0:
                    # Apply temperature scaling
                    pred = pred / temperature
                    
                    # Calculate probabilities
                    probs = F.softmax(pred, dim=-1)
                    
                    # Sample from the distribution
                    next_x_t = torch.multinomial(
                        probs.reshape(-1, self.vocab_size),
                        num_samples=1
                    ).reshape(num_samples, max_length)
                else:
                    # Greedy decoding
                    next_x_t = torch.argmax(pred, dim=-1)
                
                # Alpha-blending previous tokens and new predictions
                alpha = i / self.noise_scheduler.num_timesteps
                update_mask = (torch.rand(num_samples, max_length, device=self.device) < alpha).long()
                x_t = update_mask * x_t + (1 - update_mask) * next_x_t
                
                # If we have a mask for partial code, apply it
                if mask is not None:
                    repeated_mask = mask.repeat(num_samples, 1)
                    x_t = torch.where(repeated_mask, x_start.repeat(num_samples, 1), x_t)
        
        # Convert the sampled indices back to tokens
        sampled_indices = x_t[0].cpu().tolist()  # Take the first sample
        
        # Process the output tokens
        sampled_tokens = []
        for idx in sampled_indices:
            # Skip special tokens when building the final code
            token = tokenizer.idx_to_token(idx) if hasattr(tokenizer, 'idx_to_token') else f"tok_{idx}"
            if not token.startswith("<") or not token.endswith(">"):  # Skip special tokens
                sampled_tokens.append(token)
        
        # Combine tokens into code
        try:
            # Try to use tokenizer's detokenize method
            if hasattr(tokenizer, 'detokenize'):
                generated_code = tokenizer.detokenize(sampled_tokens)
            else:
                # Fall back to simple joining
                generated_code = "".join(sampled_tokens)
        except Exception as e:
            self.logger.error(f"Error detokenizing: {e}")
            generated_code = "".join(sampled_tokens)
        
        # Apply post-processing with rewards for quality enhancement
        if num_samples > 1:
            # Generate multiple samples and select the best one based on rewards
            all_samples = []
            sample_rewards = []
            
            for i in range(num_samples):
                # Get sample from batch
                if i == 0:
                    # Already processed first sample
                    sample_code = generated_code
                else:
                    # Process additional samples
                    sample_indices = x_t[i].cpu().tolist()
                    sample_tokens = []
                    for idx in sample_indices:
                        token = tokenizer.idx_to_token(idx) if hasattr(tokenizer, 'idx_to_token') else f"tok_{idx}"
                        if not token.startswith("<") or not token.endswith(">"):
                            sample_tokens.append(token)
                    
                    try:
                        if hasattr(tokenizer, 'detokenize'):
                            sample_code = tokenizer.detokenize(sample_tokens)
                        else:
                            sample_code = "".join(sample_tokens)
                    except Exception as e:
                        self.logger.error(f"Error detokenizing sample {i}: {e}")
                        sample_code = "".join(sample_tokens)
                
                all_samples.append(sample_code)
                
                # Compute reward for this sample
                reward = self._compute_quality_reward(sample_code, specification, language)
                sample_rewards.append(reward)
            
            # Select the best sample based on reward
            if all_samples:
                best_idx = torch.tensor(sample_rewards).argmax().item()
                return all_samples[best_idx]
        
        return generated_code
    
    def _compute_quality_reward(self, code: str, specification: str, language: str = "python") -> float:
        """
        Compute a quality reward score for generated code.
        
        Args:
            code: Generated code to evaluate
            specification: Original specification
            language: Programming language
            
        Returns:
            Combined reward score (0.0 to 1.0)
        """
        # Check if reward models are available
        if not hasattr(self, 'reward_models') or not self.reward_models:
            self.logger.warning("No reward models available for quality assessment")
            return 0.5
        
        try:
            # Use the reward models
            syntax_reward = self.reward_models.get("syntax")
            quality_reward = self.reward_models.get("quality")
            relevance_reward = self.reward_models.get("relevance")
            
            # Initialize scores
            syntax_score = 0.5
            quality_score = 0.5
            relevance_score = 0.5
            
            # Calculate rewards if models are available
            if syntax_reward:
                syntax_score = syntax_reward.evaluate(code, language)
            
            if quality_reward:
                quality_score = quality_reward.evaluate(code, language)
            
            # For relevance, handle the API difference
            if relevance_reward:
                if hasattr(relevance_reward, 'evaluate_with_reference'):
                    relevance_score = relevance_reward.evaluate_with_reference(code, specification, language)
                else:
                    relevance_score = relevance_reward.evaluate(code, reference=specification, language=language)
            
            # Combine rewards with weights
            weights = {
                "syntax": 0.4,  # Syntax correctness is most important
                "quality": 0.3,  # Code quality is second most important
                "relevance": 0.3  # Relevance to specification is also important
            }
            
            # Calculate combined score
            combined_score = (
                weights["syntax"] * syntax_score +
                weights["quality"] * quality_score +
                weights["relevance"] * relevance_score
            )
            
            # Log scores for debugging
            self.logger.debug(
                f"Code reward scores - Syntax: {syntax_score:.2f}, Quality: {quality_score:.2f}, "
                f"Relevance: {relevance_score:.2f}, Combined: {combined_score:.2f}"
            )
            
            return combined_score
        except Exception as e:
            self.logger.error(f"Error computing quality reward: {e}")
            return 0.5
    
    def _tokens_to_indices(self, tokens: List[str], tokenizer: Optional[CodeTokenizer] = None) -> List[int]:
        """
        Convert token strings to indices.
        
        Args:
            tokens: List of token strings
            tokenizer: Optional code tokenizer with vocabulary
            
        Returns:
            List of token indices
        """
        # Use tokenizer's conversion method if available
        if tokenizer and hasattr(tokenizer, 'convert_tokens_to_ids'):
            return tokenizer.convert_tokens_to_ids(tokens)
            
        # Create a vocabulary mapping for the tokens
        # This creates a consistent mapping for the same tokens
        vocab = {}
        special_tokens = {"<MASK>", "<PAD>", "<SPECIAL_TOKEN>"}
        
        # First add special tokens
        for token in special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        # Then add other tokens
        for token in tokens:
            if token not in vocab and token not in special_tokens:
                vocab[token] = len(vocab)
        
        # Make sure we don't exceed vocabulary size
        if len(vocab) >= self.vocab_size:
            self.logger.warning(f"Vocabulary size ({len(vocab)}) exceeds model vocabulary ({self.vocab_size})")
        
        # Convert tokens to indices
        indices = []
        for token in tokens:
            if token in vocab and vocab[token] < self.vocab_size:
                indices.append(vocab[token])
            else:
                # Use mask token for unknown tokens
                indices.append(self.mask_token_id)
        
        return indices