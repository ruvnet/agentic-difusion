"""
Memory-based Adaptation Mechanism.

This module implements memory-based adaptation for the diffusion models.
"""

import os
import pickle
from collections import deque
import numpy as np
from agentic_diffusion.adaptation.adaptation_mechanism import AdaptationMechanism


class MemoryAdaptation(AdaptationMechanism):
    """
    Adaptation mechanism that uses memory of past successful adaptations.
    
    This mechanism stores successful code adaptations in memory and uses them
    as examples to guide future adaptations.
    """
    
    def __init__(self, code_generator, memory_size=100, similarity_threshold=0.7):
        """
        Initialize the memory-based adaptation mechanism.
        
        Args:
            code_generator: The code generator to use for adaptations
            memory_size (int): Maximum number of examples to store
            similarity_threshold (float): Threshold for considering examples similar
        """
        self.code_generator = code_generator
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.memory = deque(maxlen=memory_size)
    
    def adapt(self, code=None, feedback=None, language=None, **kwargs):
        """
        Adapt code using memory of similar past examples.
        
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
            
        # Find similar examples in memory
        similar_examples = self._find_similar_examples(code, language)
        
        # If we have similar examples, use them to guide adaptation
        if similar_examples:
            # Create a specification that includes the original code and examples
            specification = {
                "original_code": code,
                "similar_examples": similar_examples,
                "feedback": feedback,
                "language": language
            }
            
            # Generate adapted code using the code generator
            adapted_code = self.code_generator.generate_code(
                specification=str(specification),
                language=language,
                partial_code=code
            )
            
            # Store the successful adaptation in memory
            if adapted_code and adapted_code != code:
                self._store_example(code, adapted_code, feedback, language)
                
            return adapted_code
            
        # If no similar examples found, just return the original code
        return code
    
    def _find_similar_examples(self, code, language):
        """
        Find similar examples in memory.
        
        Args:
            code (str): The code to find similar examples for
            language (str): The programming language
            
        Returns:
            list: List of similar examples
        """
        similar_examples = []
        
        for example in self.memory:
            if example["language"] != language:
                continue
                
            # Simple similarity check (could be improved with embeddings)
            similarity = self._compute_similarity(code, example["original_code"])
            if similarity >= self.similarity_threshold:
                similar_examples.append(example)
                
        return similar_examples
    
    def _compute_similarity(self, code1, code2):
        """
        Compute similarity between two code snippets.
        
        Args:
            code1 (str): First code snippet
            code2 (str): Second code snippet
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple implementation using character overlap
        # In a real system, this would use embeddings or more sophisticated methods
        if not code1 or not code2:
            return 0
            
        tokens1 = set(code1.split())
        tokens2 = set(code2.split())
        
        if not tokens1 or not tokens2:
            return 0
            
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def _store_example(self, original_code, adapted_code, feedback, language):
        """
        Store an example in memory.
        
        Args:
            original_code (str): Original code
            adapted_code (str): Adapted code
            feedback (dict): Feedback used for adaptation
            language (str): Programming language
        """
        example = {
            "original_code": original_code,
            "adapted_code": adapted_code,
            "feedback": feedback,
            "language": language
        }
        
        self.memory.append(example)
    
    def save_state(self, path):
        """
        Save the adaptation mechanism state to a file.
        
        Args:
            path (str): Path where to save the state
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            state = {
                "memory_size": self.memory_size,
                "similarity_threshold": self.similarity_threshold,
                "memory": list(self.memory)
            }
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(state, f)
                
            return True
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
            with open(path, "rb") as f:
                state = pickle.load(f)
                
            self.memory_size = state.get("memory_size", self.memory_size)
            self.similarity_threshold = state.get("similarity_threshold", 
                                                self.similarity_threshold)
            
            # Create a new deque with the loaded memory
            self.memory = deque(state.get("memory", []), maxlen=self.memory_size)
            
            return True
        except Exception:
            return False