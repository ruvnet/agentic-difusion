"""
RelevanceReward: Computes a relevance score for generated code with respect to a reference specification or code.

This module provides sophisticated metrics for evaluating how well generated code
matches reference specifications or requirements. It performs keyword analysis,
semantic matching, and can identify whether required functionality is implemented.
"""

import re
import logging
import math
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

class RelevanceReward:
    """
    Computes a reward score for code relevance.
    
    This reward model evaluates how well generated code addresses the reference
    requirements using various similarity and matching techniques.
    """

    def __init__(self, similarity_fn=None, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the relevance reward model.
        
        Args:
            similarity_fn: Optional custom similarity function
            weights: Optional custom weights for different relevance aspects
        """
        self.similarity_fn = similarity_fn or self.improved_similarity
        
        # Default weights for relevance metrics
        self.weights = weights or {
            "keyword": 0.30,
            "semantic": 0.35,
            "functional": 0.35
        }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            for key in self.weights:
                self.weights[key] /= total_weight
        
        logger.info("Initialized RelevanceReward with weights: %s", self.weights)
    
    def __call__(self, code: str, reference: Optional[str] = None, language: str = "python") -> float:
        """
        Compute relevance score for the given code.
        
        Args:
            code: Code to evaluate
            reference: Reference code or specification
            language: Programming language of the code
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not reference:
            return 1.0  # No reference, assume max relevance
        
        return self.evaluate(code, reference, language)
    
    def evaluate(self, code: str, reference: Optional[str] = None, language: str = "python") -> float:
        """
        Evaluate code relevance.
        
        Args:
            code: Code to evaluate
            reference: Reference code or specification
            language: Programming language of the code
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not reference:
            return 1.0
        
        try:
            # Check if code is empty or whitespace
            if not code or code.isspace():
                return 0.0
            
            # Evaluate different relevance aspects
            keyword_score = self._keyword_relevance(code, reference, language)
            semantic_score = self._semantic_relevance(code, reference, language)
            functional_score = self._functional_relevance(code, reference, language)
            
            # Compute weighted score
            weighted_score = (
                self.weights["keyword"] * keyword_score +
                self.weights["semantic"] * semantic_score +
                self.weights["functional"] * functional_score
            )
            
            return weighted_score
        
        except Exception as e:
            logger.error("Error evaluating relevance: %s", str(e))
            return 0.0
    
    def improved_similarity(self, code: str, reference: str, language: str = "python") -> float:
        """
        Improved similarity metric that combines multiple relevance aspects.
        
        Args:
            code: Code to evaluate
            reference: Reference code or specification
            language: Programming language of the code
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Combine multiple relevance metrics
        keyword_score = self._keyword_relevance(code, reference, language)
        semantic_score = self._semantic_relevance(code, reference, language)
        functional_score = self._functional_relevance(code, reference, language)
        
        # Weighted combination
        return (
            self.weights["keyword"] * keyword_score +
            self.weights["semantic"] * semantic_score +
            self.weights["functional"] * functional_score
        )
    
    def _keyword_relevance(self, code: str, reference: str, language: str) -> float:
        """
        Calculate keyword-based relevance.
        
        Args:
            code: Code to evaluate
            reference: Reference code or specification
            language: Programming language of the code
            
        Returns:
            Keyword relevance score between 0.0 and 1.0
        """
        # Extract keywords from reference
        reference_keywords = self._extract_keywords(reference, language)
        if not reference_keywords:
            return 0.0
        
        # Extract keywords from code
        code_keywords = self._extract_keywords(code, language)
        
        # Calculate keyword overlap
        overlap = reference_keywords & code_keywords
        
        # Calculate weighted score based on keyword importance
        if not overlap:
            return 0.0
        
        # Calculate TF-IDF like weighting
        score = 0.0
        for keyword in overlap:
            # More weight for longer, more specific keywords
            weight = min(1.0, math.log(len(keyword) + 1) / 2)
            score += weight
        
        # Normalize by total possible score
        max_score = sum(min(1.0, math.log(len(keyword) + 1) / 2) for keyword in reference_keywords)
        if max_score == 0:
            return 0.0
        
        return min(1.0, score / max_score)
    
    def _semantic_relevance(self, code: str, reference: str, language: str) -> float:
        """
        Calculate semantic relevance based on concept matching.
        
        Args:
            code: Code to evaluate
            reference: Reference code or specification
            language: Programming language of the code
            
        Returns:
            Semantic relevance score between 0.0 and 1.0
        """
        # Extract concepts from reference spec
        reference_concepts = self._extract_concepts(reference, language)
        if not reference_concepts:
            return 0.0
        
        # Extract concepts from code
        code_concepts = self._extract_concepts(code, language)
        
        # Calculate concept match score
        matched_concepts = 0
        for concept in reference_concepts:
            for code_concept in code_concepts:
                # Check if concepts are related (simple substring check)
                if concept.lower() in code_concept.lower() or code_concept.lower() in concept.lower():
                    matched_concepts += 1
                    break
        
        return min(1.0, matched_concepts / len(reference_concepts))
    
    def _functional_relevance(self, code: str, reference: str, language: str) -> float:
        """
        Calculate functional relevance based on required functionality.
        
        Args:
            code: Code to evaluate
            reference: Reference code or specification
            language: Programming language of the code
            
        Returns:
            Functional relevance score between 0.0 and 1.0
        """
        # Extract required functionality from reference
        required_functions = self._extract_required_functions(reference, language)
        if not required_functions:
            return 1.0  # No specific functions required
        
        # Extract implemented functions from code
        implemented_functions = self._extract_implemented_functions(code, language)
        
        # Calculate function match score
        matched_functions = 0
        for req_func in required_functions:
            for impl_func in implemented_functions:
                # Check if function names are similar
                if self._are_functions_similar(req_func, impl_func, language):
                    matched_functions += 1
                    break
        
        return min(1.0, matched_functions / len(required_functions))
    
    def _extract_keywords(self, text: str, language: str) -> Set[str]:
        """
        Extract relevant keywords from text.
        
        Args:
            text: Text to extract keywords from
            language: Programming language context
            
        Returns:
            Set of keywords
        """
        # Remove common code syntax
        cleaned_text = re.sub(r'[{}()\[\]:;,.]', ' ', text)
        
        # Remove language-specific keywords
        language_keywords = {
            "python": {"def", "class", "if", "else", "elif", "for", "while", "try", "except", "with", "as", "import", "from", "return", "pass", "break", "continue"},
            "javascript": {"function", "var", "let", "const", "if", "else", "for", "while", "try", "catch", "return", "break", "continue", "switch", "case"},
            "java": {"public", "private", "protected", "class", "interface", "extends", "implements", "static", "final", "void", "if", "else", "for", "while", "try", "catch", "return", "break", "continue", "switch", "case"},
            "go": {"func", "var", "const", "package", "import", "if", "else", "for", "switch", "case", "defer", "return", "break", "continue"}
        }
        
        words = cleaned_text.lower().split()
        stopwords = language_keywords.get(language.lower(), set())
        
        # Filter out language keywords and short words
        keywords = {word for word in words if word not in stopwords and len(word) > 2}
        
        return keywords
    
    def _extract_concepts(self, text: str, language: str) -> List[str]:
        """
        Extract high-level concepts from text.
        
        Args:
            text: Text to extract concepts from
            language: Programming language context
            
        Returns:
            List of concepts
        """
        # For simplicity, extract noun phrases and multi-word terms
        concepts = []
        
        # Extract noun phrases (simplified)
        noun_phrase_pattern = r'\b([A-Za-z]+(?:\s+[A-Za-z]+){1,3})\b'
        noun_phrases = re.findall(noun_phrase_pattern, text)
        concepts.extend([np for np in noun_phrases if len(np.split()) >= 2])
        
        # Extract camelCase and snake_case identifiers
        identifier_pattern = r'\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+(?:_[a-z]+)+'
        identifiers = re.findall(identifier_pattern, text)
        concepts.extend(identifiers)
        
        return concepts
    
    def _extract_required_functions(self, reference: str, language: str) -> List[str]:
        """
        Extract required functions or methods from reference.
        
        Args:
            reference: Reference text
            language: Programming language context
            
        Returns:
            List of required function names
        """
        functions = []
        
        # Look for function-like statements in the reference
        if language.lower() == "python":
            # Look for "function to X" or "method that X" patterns
            func_patterns = [
                r'function\s+to\s+([^.,:;]+)',
                r'method\s+to\s+([^.,:;]+)',
                r'implement\s+([^.,:;]+\s+function)',
                r'create\s+a\s+function\s+(?:to|that)\s+([^.,:;]+)'
            ]
            
            for pattern in func_patterns:
                matches = re.findall(pattern, reference, re.IGNORECASE)
                functions.extend(matches)
                
            # Look for explicit function names with def
            def_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            functions.extend(re.findall(def_pattern, reference))
            
        elif language.lower() in ["javascript", "java"]:
            # Patterns for JS/Java function descriptions
            func_patterns = [
                r'function\s+(?:to|that)\s+([^.,:;]+)',
                r'method\s+(?:to|that)\s+([^.,:;]+)',
                r'implement\s+([^.,:;]+\s+function)',
                r'create\s+a\s+(?:function|method)\s+(?:to|that)\s+([^.,:;]+)'
            ]
            
            for pattern in func_patterns:
                matches = re.findall(pattern, reference, re.IGNORECASE)
                functions.extend(matches)
                
            # Look for explicit function names
            if language.lower() == "javascript":
                def_pattern = r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)'
                functions.extend(re.findall(def_pattern, reference))
            else:  # Java
                def_pattern = r'(?:public|private|protected|static)?\s+\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                functions.extend(re.findall(def_pattern, reference))
                
        elif language.lower() == "go":
            # Patterns for Go function descriptions
            func_patterns = [
                r'function\s+(?:to|that)\s+([^.,:;]+)',
                r'method\s+(?:to|that)\s+([^.,:;]+)',
                r'implement\s+([^.,:;]+\s+function)',
                r'create\s+a\s+(?:function|method)\s+(?:to|that)\s+([^.,:;]+)'
            ]
            
            for pattern in func_patterns:
                matches = re.findall(pattern, reference, re.IGNORECASE)
                functions.extend(matches)
                
            # Look for explicit function names
            def_pattern = r'func\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            functions.extend(re.findall(def_pattern, reference))
            
        return functions
    
    def _extract_implemented_functions(self, code: str, language: str) -> List[str]:
        """
        Extract implemented functions or methods from code.
        
        Args:
            code: Code to analyze
            language: Programming language context
            
        Returns:
            List of implemented function names
        """
        functions = []
        
        # Extract function definitions based on language
        if language.lower() == "python":
            # Python function definitions
            pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            functions = re.findall(pattern, code)
            
        elif language.lower() == "javascript":
            # JavaScript function definitions (multiple patterns)
            patterns = [
                r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # function declarations
                r'(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*function',  # function expressions
                r'(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\(',  # arrow functions
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*function'  # object methods
            ]
            
            for pattern in patterns:
                functions.extend(re.findall(pattern, code))
                
        elif language.lower() == "java":
            # Java method definitions
            pattern = r'(?:public|private|protected|static)?\s+\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            functions = re.findall(pattern, code)
            
        elif language.lower() == "go":
            # Go function definitions
            pattern = r'func\s+(?:\([^)]*\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)'
            functions = re.findall(pattern, code)
            
        return functions
    
    def _are_functions_similar(self, func1: str, func2: str, language: str) -> bool:
        """
        Determine if two function names or descriptions are similar.
        
        Args:
            func1: First function name/description
            func2: Second function name/description
            language: Programming language context
            
        Returns:
            True if functions are similar, False otherwise
        """
        # Convert to lowercase for comparison
        func1 = func1.lower()
        func2 = func2.lower()
        
        # Direct match
        if func1 == func2:
            return True
            
        # Clean function names - keep only letters and convert to lowercase
        clean_func1 = re.sub(r'[^a-zA-Z]', '', func1).lower()
        clean_func2 = re.sub(r'[^a-zA-Z]', '', func2).lower()
        
        # Check if one is a substring of the other
        if clean_func1 in clean_func2 or clean_func2 in clean_func1:
            return True
            
        # Check for word similarity
        words1 = set(re.findall(r'\b\w+\b', func1))
        words2 = set(re.findall(r'\b\w+\b', func2))
        
        # If more than 50% of words match
        if words1 and words2:
            overlap = words1 & words2
            similarity = len(overlap) / min(len(words1), len(words2))
            if similarity > 0.5:
                return True
                
        return False