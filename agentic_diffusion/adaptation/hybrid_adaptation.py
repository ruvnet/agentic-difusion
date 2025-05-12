"""
Hybrid Adaptation Mechanism.

This module implements a hybrid adaptation mechanism that combines
gradient-based and memory-based approaches.
"""

import os
import pickle
from agentic_diffusion.adaptation.adaptation_mechanism import AdaptationMechanism
from agentic_diffusion.adaptation.gradient_adaptation import GradientBasedAdaptation
from agentic_diffusion.adaptation.memory_adaptation import MemoryAdaptation


class HybridAdaptation(AdaptationMechanism):
    """
    Adaptation mechanism that combines gradient-based and memory-based approaches.
    
    This mechanism uses both gradient updates and memory of past examples
    to provide robust code adaptation.
    """
    
    def __init__(self, diffusion_model, code_generator, 
                 gradient_weight=0.5, memory_weight=0.5,
                 adaptation_rate=0.1, memory_size=100):
        """
        Initialize the hybrid adaptation mechanism.
        
        Args:
            diffusion_model: The diffusion model to adapt
            code_generator: The code generator to use for memory-based adaptation
            gradient_weight (float): Weight for gradient-based adaptation (0-1)
            memory_weight (float): Weight for memory-based adaptation (0-1)
            adaptation_rate (float): Learning rate for gradient adaptation
            memory_size (int): Maximum number of examples for memory adaptation
        """
        self.diffusion_model = diffusion_model
        self.code_generator = code_generator
        self.gradient_weight = gradient_weight
        self.memory_weight = memory_weight
        
        # Create the individual adaptation mechanisms
        self.gradient_adaptation = GradientBasedAdaptation(
            diffusion_model, 
            adaptation_rate=adaptation_rate
        )
        
        self.memory_adaptation = MemoryAdaptation(
            code_generator,
            memory_size=memory_size
        )
    
    def adapt(self, code=None, feedback=None, language=None, **kwargs):
        """
        Adapt code using both gradient and memory-based approaches.
        
        Args:
            code (str, optional): Code to adapt. Defaults to None.
            feedback (dict, optional): Feedback to incorporate. Defaults to None.
            language (str, optional): Programming language. Defaults to None.
            **kwargs: Additional parameters.
                
        Returns:
            str: The adapted code
        """
        if not code:
            return None
            
        # Both mechanisms operate on different underlying models (diffusion vs code generator),
        # so we can run them sequentially
        
        # First, run gradient-based adaptation if the gradient weight is significant
        gradient_result = None
        if self.gradient_weight > 0.1:  # Threshold to avoid unnecessary computation
            gradient_result = self.gradient_adaptation.adapt(
                code=code, 
                feedback=feedback, 
                language=language,
                **kwargs
            )
            
        # Then, run memory-based adaptation if the memory weight is significant
        memory_result = None
        if self.memory_weight > 0.1:  # Threshold to avoid unnecessary computation
            # If we have a gradient result, use it as input to memory adaptation
            memory_input = gradient_result if gradient_result else code
            memory_result = self.memory_adaptation.adapt(
                code=memory_input,
                feedback=feedback,
                language=language,
                **kwargs
            )
            
        # Determine the final result based on weights and available results
        if memory_result and gradient_result:
            # Both mechanisms produced results
            # For simplicity, prioritize memory-based results if both are available
            # In a real system, we might do more sophisticated combination
            if self.memory_weight >= self.gradient_weight:
                return memory_result
            else:
                return gradient_result
        elif memory_result:
            return memory_result
        elif gradient_result:
            return gradient_result
        else:
            return code  # No adaptation performed
    
    def save_state(self, path):
        """
        Save the adaptation mechanism state to a file.
        
        Args:
            path (str): Path where to save the state
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory structure for the hybrid adaptation
            hybrid_dir = os.path.dirname(path)
            os.makedirs(hybrid_dir, exist_ok=True)
            
            # Save the hybrid configuration
            config = {
                "gradient_weight": self.gradient_weight,
                "memory_weight": self.memory_weight
            }
            
            with open(path, "wb") as f:
                pickle.dump(config, f)
                
            # Save the individual mechanisms
            gradient_path = os.path.join(hybrid_dir, "gradient_adaptation.pkl")
            memory_path = os.path.join(hybrid_dir, "memory_adaptation.pkl")
            
            gradient_success = self.gradient_adaptation.save_state(gradient_path)
            memory_success = self.memory_adaptation.save_state(memory_path)
            
            return gradient_success and memory_success
        except Exception:
            return False
    
    def load_state(self, path):
        """
        Load the adaptation mechanism state from a file.
        
        Args:
            path (str): Path from where to load the state
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the hybrid configuration
            with open(path, "rb") as f:
                config = pickle.load(f)
                
            self.gradient_weight = config.get("gradient_weight", self.gradient_weight)
            self.memory_weight = config.get("memory_weight", self.memory_weight)
            
            # Load the individual mechanisms
            hybrid_dir = os.path.dirname(path)
            gradient_path = os.path.join(hybrid_dir, "gradient_adaptation.pkl")
            memory_path = os.path.join(hybrid_dir, "memory_adaptation.pkl")
            
            gradient_success = self.gradient_adaptation.load_state(gradient_path)
            memory_success = self.memory_adaptation.load_state(memory_path)
            
            return gradient_success and memory_success
        except Exception:
            return False