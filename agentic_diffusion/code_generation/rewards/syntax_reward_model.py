"""
SyntaxRewardModel: Advanced model for computing syntax correctness score for generated code.
"""

from agentic_diffusion.code_generation.syntax_model import SyntaxModel
from agentic_diffusion.code_generation.syntax_parsers.base_parser import get_parser_for_language

class SyntaxRewardModel:
    """
    Advanced model for computing a reward score for syntax correctness.
    This model provides a more nuanced scoring approach compared to the basic SyntaxReward.
    """

    def __init__(self):
        self.syntax_model = SyntaxModel()
        
    def evaluate(self, code: str, language: str = "python") -> float:
        """
        Evaluate syntax correctness with a nuanced scoring approach.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            
        Returns:
            Float score between 0.0 and 1.0
        """
        # First check basic syntax validity
        is_valid = self.syntax_model.validate(code, language)
        if not is_valid:
            # Try to determine severity of syntax errors
            return self._evaluate_partial_correctness(code, language)
        
        return 1.0
    
    def _evaluate_partial_correctness(self, code: str, language: str) -> float:
        """
        Evaluate partial correctness when code has syntax errors.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            
        Returns:
            Float score between 0.0 and 0.7 (since it has errors)
        """
        try:
            # Get appropriate parser for the language
            parser = get_parser_for_language(language)
            
            # Split code into lines
            lines = code.strip().split('\n')
            if not lines:
                return 0.0
                
            # Count lines that parse correctly individually
            valid_lines = 0
            for line in lines:
                if line.strip() and not line.strip().startswith("#"):
                    try:
                        # Simple line-by-line validity check
                        is_line_valid = not parser.has_obvious_errors(line)
                        if is_line_valid:
                            valid_lines += 1
                    except:
                        # Line has severe syntax issues
                        pass
            
            # Calculate partial score based on valid lines ratio
            valid_ratio = valid_lines / max(1, len([l for l in lines if l.strip() and not l.strip().startswith("#")]))
            
            # Scale to max 0.7 since the code has errors
            return min(0.7, valid_ratio * 0.7)
            
        except Exception as e:
            # Fallback if parsing fails completely
            return 0.2  # Minimal score for any code that's not completely invalid