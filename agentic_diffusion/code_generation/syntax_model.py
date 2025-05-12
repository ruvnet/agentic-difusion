"""
SyntaxModel: Provides syntax validation and language-specific constraints for code generation.
"""

class SyntaxModel:
    """
    Syntax validation and constraint enforcement for code generation.
    """

    def __init__(self, python_parser_cls=None, javascript_parser_cls=None, java_parser_cls=None, go_parser_cls=None):
        """
        Optionally inject parser classes for different languages (for testing/mocking).
        """
        self.python_parser = python_parser_cls() if python_parser_cls else None
        self.javascript_parser = javascript_parser_cls() if javascript_parser_cls else None
        self.java_parser = java_parser_cls() if java_parser_cls else None
        self.go_parser = go_parser_cls() if go_parser_cls else None

    def validate(self, code: str, language: str = "python") -> bool:
        """
        Returns True if the code is syntactically valid for the given language.

        Args:
            code (str): The code to validate.
            language (str): Programming language.

        Returns:
            bool: True if valid, False otherwise.
        """
        if language == "python":
            try:
                compile(code, "<string>", "exec")
                return True
            except Exception:
                return False
        # For other languages, always return True (stub)
        return True

    def validate_syntax(self, code: str, language: str = "python") -> bool:
        """
        Alias for validate. Returns True if code is syntactically valid.
        """
        return self.validate(code, language)

    def enforce_constraints(self, code: str, language: str = "python") -> bool:
        """
        Enforce language-specific constraints. Stub: always returns True unless check_constraints is overridden.
        """
        if hasattr(self, "check_constraints"):
            return self.check_constraints(code, language)
        return True

    def syntax_aware_sample(self, samples, language="python"):
        """
        Returns only syntactically valid samples for the given language.
        """
        return [s for s in samples if self.validate_syntax(s, language)]