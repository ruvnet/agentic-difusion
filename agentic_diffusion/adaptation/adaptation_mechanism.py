"""
Adaptation Mechanism Base Interface.

This module defines the base interface for all adaptation mechanisms used in the Agentic Diffusion system.
"""

from abc import ABC, abstractmethod


class AdaptationMechanism(ABC):
    """Base abstract class for all adaptation mechanisms."""
    
    @abstractmethod
    def adapt(self, code=None, feedback=None, language=None, **kwargs):
        """
        Adapt code based on feedback and other parameters.
        
        Args:
            code (str, optional): The code to adapt. Defaults to None.
            feedback (dict, optional): Feedback to incorporate. Defaults to None.
            language (str, optional): Programming language of the code. Defaults to None.
            **kwargs: Additional parameters specific to the adaptation mechanism.
            
        Returns:
            str: The adapted code.
        """
        pass
    
    @abstractmethod
    def save_state(self, path):
        """
        Save the adaptation mechanism state to a file.
        
        Args:
            path (str): Path where to save the state.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def load_state(self, path):
        """
        Load the adaptation mechanism state from a file.
        
        Args:
            path (str): Path from where to load the state.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        pass