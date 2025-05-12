"""
RelevanceRewardModel: Advanced model for computing code relevance to a specification or reference.
"""

import re
from typing import Optional, Set, List, Dict
import difflib

class RelevanceRewardModel:
    """
    Advanced model for computing a reward score for code relevance to a specification or reference.
    This model uses sophisticated text analysis methods beyond simple token overlap.
    """

    def __init__(self):
        self.keyword_importance = {
            "python": {
                "high": ["def", "class", "import", "from", "return"],
                "medium": ["if", "elif", "else", "for", "while", "try", "except", "with"],
                "low": ["and", "or", "not", "is", "in", "as"]
            },
            "javascript": {
                "high": ["function", "class", "import", "export", "return", "const", "let"],
                "medium": ["if", "else", "for", "while", "try", "catch", "await", "async"],
                "low": ["var", "typeof", "instanceof", "null", "undefined"]
            },
            "java": {
                "high": ["class", "interface", "extends", "implements", "import", "package", "return"],
                "medium": ["public", "private", "protected", "static", "final", "void", "new"],
                "low": ["if", "else", "for", "while", "try", "catch", "null"]
            }
        }
        
    def evaluate(self, code: str, reference: Optional[str] = None, language: str = "python") -> float:
        """
        Evaluate code relevance to a specification or reference.
        
        Args:
            code: Code to evaluate
            reference: Reference specification or code
            language: Programming language of the code
            
        Returns:
            Float score between 0.0 and 1.0
        """
        if not reference:
            return 1.0  # No reference to compare against
            
        if not code:
            return 0.0  # No code to evaluate
            
        # Calculate multiple relevance metrics
        semantic_similarity = self._calculate_semantic_similarity(code, reference)
        structural_similarity = self._calculate_structural_similarity(code, reference, language)
        functionality_coverage = self._calculate_functionality_coverage(code, reference, language)
        
        # Weighted combination of metrics
        relevance_score = (
            0.5 * semantic_similarity +
            0.3 * structural_similarity +
            0.2 * functionality_coverage
        )
        
        return max(0.0, min(1.0, relevance_score))
    
    def _calculate_semantic_similarity(self, code: str, reference: str) -> float:
        """
        Calculate semantic similarity between code and reference.
        Uses token overlap with weighting for important tokens.
        
        Args:
            code: Code to evaluate
            reference: Reference specification or code
            
        Returns:
            Float score between 0.0 and 1.0
        """
        # Clean and tokenize
        code_tokens = self._clean_and_tokenize(code)
        ref_tokens = self._clean_and_tokenize(reference)
        
        if not ref_tokens:
            return 0.0
            
        # Calculate overlap
        overlap = code_tokens.intersection(ref_tokens)
        
        # Calculate Jaccard similarity
        jaccard = len(overlap) / (len(code_tokens.union(ref_tokens)) or 1)
        
        # Calculate token recall (how many ref tokens are in code)
        recall = len(overlap) / (len(ref_tokens) or 1)
        
        # Combine metrics
        return 0.7 * recall + 0.3 * jaccard
    
    def _calculate_structural_similarity(self, code: str, reference: str, language: str) -> float:
        """
        Calculate structural similarity between code and reference.
        Focuses on similar code structure patterns.
        
        Args:
            code: Code to evaluate
            reference: Reference specification or code
            language: Programming language
            
        Returns:
            Float score between 0.0 and 1.0
        """
        # Extract structural elements based on language
        code_structure = self._extract_structural_elements(code, language)
        ref_structure = self._extract_structural_elements(reference, language)
        
        # Calculate similarity using difflib's SequenceMatcher
        matcher = difflib.SequenceMatcher(None, code_structure, ref_structure)
        
        return matcher.ratio()
    
    def _calculate_functionality_coverage(self, code: str, reference: str, language: str) -> float:
        """
        Calculate how well the code covers functionality described in the reference.
        
        Args:
            code: Code to evaluate
            reference: Reference specification or code
            language: Programming language
            
        Returns:
            Float score between 0.0 and 1.0
        """
        # Extract key terms from reference (potential function names, classes, etc.)
        ref_key_terms = self._extract_key_terms(reference, language)
        
        if not ref_key_terms:
            return 0.5  # Default middle score if no key terms found
        
        # Count how many key terms appear in the code
        code_lower = code.lower()
        matches = sum(1 for term in ref_key_terms if term.lower() in code_lower)
        
        # Calculate coverage ratio
        coverage = matches / len(ref_key_terms)
        
        return coverage
    
    def _clean_and_tokenize(self, text: str) -> Set[str]:
        """
        Clean and tokenize text for comparison.
        
        Args:
            text: Text to process
            
        Returns:
            Set of cleaned tokens
        """
        # Convert to lowercase and split by non-alphanumeric chars
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', text.lower())
        
        # Remove common stopwords
        stopwords = {"a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
                     "at", "from", "by", "on", "off", "for", "in", "out", "over", "to"}
        
        return {token for token in tokens if token not in stopwords}
    
    def _extract_structural_elements(self, text: str, language: str) -> List[str]:
        """
        Extract structural elements from code or text based on language.
        
        Args:
            text: Code or specification text
            language: Programming language
            
        Returns:
            List of structural elements
        """
        structure = []
        
        if language.lower() == "python":
            # Extract Python structural elements
            # Find function definitions
            for match in re.finditer(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', text):
                structure.append(f"function:{match.group(1)}")
                
            # Find class definitions
            for match in re.finditer(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', text):
                structure.append(f"class:{match.group(1)}")
                
            # Find import statements
            for match in re.finditer(r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)', text):
                structure.append(f"import:{match.group(1)}")
                
        elif language.lower() in ["javascript", "typescript"]:
            # Extract JS/TS structural elements
            # Find function definitions
            for match in re.finditer(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)', text):
                structure.append(f"function:{match.group(1)}")
                
            # Find arrow functions assigned to variables
            for match in re.finditer(r'(const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\([^)]*\)|[a-zA-Z_][a-zA-Z0-9_]*)\s*=>', text):
                structure.append(f"function:{match.group(2)}")
                
            # Find class definitions
            for match in re.finditer(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', text):
                structure.append(f"class:{match.group(1)}")
                
            # Find imports
            for match in re.finditer(r'import\s+.*\s+from\s+[\'"]([^\'"]+)', text):
                structure.append(f"import:{match.group(1)}")
                
        elif language.lower() == "java":
            # Extract Java structural elements
            # Find class definitions
            for match in re.finditer(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', text):
                structure.append(f"class:{match.group(1)}")
                
            # Find method definitions
            for match in re.finditer(r'(public|private|protected|static|\s)+[a-zA-Z_][a-zA-Z0-9_]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(\{|throws)', text):
                structure.append(f"method:{match.group(2)}")
                
            # Find imports
            for match in re.finditer(r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*);', text):
                structure.append(f"import:{match.group(1)}")
        
        # For other languages or if nothing specific was found, use generic approach
        if not structure:
            # Generic approach: just look for potential function/method names
            for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', text):
                structure.append(f"function:{match.group(1)}")
        
        return structure
    
    def _extract_key_terms(self, text: str, language: str) -> List[str]:
        """
        Extract key terms that might represent functionality requirements.
        
        Args:
            text: Text to process
            language: Programming language
            
        Returns:
            List of key terms
        """
        # Extract camelCase and snake_case terms as likely important identifiers
        key_terms = []
        
        # Find camelCase identifiers
        for match in re.finditer(r'\b([a-z][a-z0-9]*[A-Z][a-zA-Z0-9]*)\b', text):
            key_terms.append(match.group(1))
            
        # Find snake_case identifiers
        for match in re.finditer(r'\b([a-z][a-z0-9]*_[a-z][a-zA-Z0-9_]*)\b', text):
            key_terms.append(match.group(1))
            
        # Find terms that might be method or function names
        for match in re.finditer(r'\b([a-zA-Z][a-zA-Z0-9]*)\(', text):
            key_terms.append(match.group(1))
            
        # Look for capitalized terms (potential class names)
        for match in re.finditer(r'\b([A-Z][a-zA-Z0-9]*)\b', text):
            key_terms.append(match.group(1))
            
        # Add language-specific keywords based on importance
        if language.lower() in self.keyword_importance:
            lang_keywords = self.keyword_importance[language.lower()]
            
            # Look for high importance keywords in the text
            for keyword in lang_keywords["high"]:
                if f" {keyword} " in f" {text} " or f"\n{keyword} " in f"\n{text} ":
                    key_terms.append(keyword)
            
        # Return unique terms
        return list(set(key_terms))