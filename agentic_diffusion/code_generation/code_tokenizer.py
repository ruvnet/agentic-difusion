"""
Tokenizer for code generation.

This module provides tokenization utilities for code in different
programming languages, supporting the diffusion-based code generation pipeline.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

class CodeTokenizer:
    """
    Tokenizer for code in different programming languages.
    
    This class handles tokenization and detokenization of code
    for use in diffusion-based code generation models.
    """
    
    def __init__(
        self,
        language: str = "python",
        vocab_size: int = 10000,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize the code tokenizer.
        
        Args:
            language: Programming language to tokenize
            vocab_size: Size of the vocabulary
            special_tokens: List of special tokens to include
        """
        self.logger = logging.getLogger(__name__)
        self.language = language.lower()
        self.vocab_size = vocab_size
        
        # Initialize special tokens
        self.special_tokens = special_tokens or [
            "<PAD>",   # Padding token
            "<MASK>",  # Mask token for masked language modeling
            "<BOS>",   # Beginning of sequence
            "<EOS>",   # End of sequence
            "<UNK>"    # Unknown token
        ]
        
        # Initialize regex patterns based on language
        self._init_language_patterns()
        
        # Initialize vocabulary (will be built on first use)
        self.token_to_id = {}
        self.id_to_token = {}
        self._initialize_vocabulary()
        
        self.logger.info(f"Initialized CodeTokenizer for {language} with vocab size {vocab_size}")
    
    def _init_language_patterns(self) -> None:
        """
        Initialize regex patterns for tokenization based on the language.
        """
        # Common patterns for most languages
        identifier_pattern = r'[a-zA-Z_]\w*'
        string_pattern = r'\"(?:\\.|[^\"\\])*\"|\'(?:\\.|[^\'\\])*\''
        number_pattern = r'\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
        
        # Language-specific patterns
        if self.language == "python":
            self.token_patterns = [
                (r'#.*?$', 'COMMENT'),                     # Comments
                (string_pattern, 'STRING'),                # Strings
                (number_pattern, 'NUMBER'),                # Numbers
                (r'(from|import|as|def|class|if|elif|else|for|while|return|try|except|finally|raise|with|assert|lambda)',
                 'KEYWORD'),                               # Keywords
                (r'(\+|\-|\*|\/|\%|\*\*|\/\/|\=|\<|\>|\<\=|\>\=|\=\=|\!\=)', 'OPERATOR'),  # Operators
                (r'(\(|\)|\[|\]|\{|\}|\,|\:|\.|\'|\"|\;)', 'PUNCTUATION'),      # Punctuation
                (identifier_pattern, 'IDENTIFIER'),        # Identifiers
                (r'\s+', 'WHITESPACE'),                    # Whitespace
            ]
        elif self.language == "javascript":
            self.token_patterns = [
                (r'\/\/.*?$', 'COMMENT'),                  # Single-line comments
                (r'\/\*.*?\*\/', 'COMMENT'),               # Multi-line comments
                (string_pattern, 'STRING'),                # Strings
                (number_pattern, 'NUMBER'),                # Numbers
                (r'(var|let|const|function|if|else|for|while|return|try|catch|finally|throw|class|new|this|super)',
                 'KEYWORD'),                               # Keywords
                (r'(\+|\-|\*|\/|\%|\+\+|\-\-|\=|\<|\>|\<\=|\>\=|\=\=|\=\=\=|\!\=|\!\=\=)', 'OPERATOR'),  # Operators
                (r'(\(|\)|\[|\]|\{|\}|\,|\:|\.|\'|\"|\;)', 'PUNCTUATION'),      # Punctuation
                (identifier_pattern, 'IDENTIFIER'),        # Identifiers
                (r'\s+', 'WHITESPACE'),                    # Whitespace
            ]
        else:
            # Generic patterns for other languages
            self.token_patterns = [
                (r'\/\/.*?$|\/\*.*?\*\/|#.*?$', 'COMMENT'),  # Comments
                (string_pattern, 'STRING'),                  # Strings
                (number_pattern, 'NUMBER'),                  # Numbers
                (r'\b[a-zA-Z_]\w*\b', 'IDENTIFIER'),        # Identifiers
                (r'[\+\-\*\/\%\=\<\>\!\&\|\^\~\?\:]+', 'OPERATOR'),  # Operators
                (r'[\(\)\[\]\{\}\,\.\;\'\"]', 'PUNCTUATION'),  # Punctuation
                (r'\s+', 'WHITESPACE'),                    # Whitespace
            ]
    
    def _initialize_vocabulary(self) -> None:
        """
        Initialize the vocabulary with special tokens.
        """
        # Add special tokens to vocabulary
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Reserved ids for special tokens
        self.pad_id = self.token_to_id.get("<PAD>", 0)
        self.mask_id = self.token_to_id.get("<MASK>", 1)
        self.bos_id = self.token_to_id.get("<BOS>", 2)
        self.eos_id = self.token_to_id.get("<EOS>", 3)
        self.unk_id = self.token_to_id.get("<UNK>", 4)
        
        # Pre-calculate the starting index for regular tokens
        self.regular_token_start_idx = len(self.special_tokens)
    
    def tokenize(self, code: str) -> List[str]:
        """
        Tokenize code into a list of token strings.
        
        Args:
            code: Code string to tokenize
            
        Returns:
            List of tokens
        """
        tokens = []
        
        # Add beginning of sequence token
        tokens.append("<BOS>")
        
        # Tokenize the code based on the language patterns
        remaining_code = code
        while remaining_code:
            matched = False
            for pattern, token_type in self.token_patterns:
                regex = re.compile(pattern, re.MULTILINE)
                match = regex.match(remaining_code)
                if match:
                    token_text = match.group(0)
                    # Skip whitespace tokens
                    if token_type != 'WHITESPACE':
                        # Special handling for different token types
                        if token_type == 'IDENTIFIER' and len(token_text) > 20:
                            # Split long identifiers
                            subtokens = self._split_long_identifier(token_text)
                            tokens.extend(subtokens)
                        else:
                            tokens.append(token_text)
                    
                    # Move past this token
                    remaining_code = remaining_code[len(token_text):]
                    matched = True
                    break
            
            # If no pattern matched, add a character as an unknown token
            if not matched:
                tokens.append(remaining_code[0])
                remaining_code = remaining_code[1:]
        
        # Add end of sequence token
        tokens.append("<EOS>")
        
        return tokens
    
    def _split_long_identifier(self, identifier: str) -> List[str]:
        """
        Split long identifiers into smaller parts.
        
        Args:
            identifier: Long identifier to split
            
        Returns:
            List of smaller identifier parts
        """
        # Common word boundaries in code
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+|\W+', identifier)
        
        # If the split didn't work well, use fixed-length chunks
        if not parts or sum(len(p) for p in parts) < len(identifier) * 0.8:
            # Fall back to fixed-length chunks of 8 characters
            parts = [identifier[i:i+8] for i in range(0, len(identifier), 8)]
        
        return parts
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to code.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed code as a string
        """
        # Filter out special tokens
        regular_tokens = [
            token for token in tokens 
            if not (token.startswith("<") and token.endswith(">"))
        ]
        
        # Reconstruct code based on the language
        if self.language == "python":
            # Handle Python-specific detokenization
            return self._detokenize_python(regular_tokens)
        elif self.language == "javascript":
            # Handle JavaScript-specific detokenization
            return self._detokenize_javascript(regular_tokens)
        else:
            # Generic detokenization for other languages
            return "".join(regular_tokens)
    
    def _detokenize_python(self, tokens: List[str]) -> str:
        """
        Python-specific detokenization.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed Python code
        """
        code = ""
        for i, token in enumerate(tokens):
            # Special handling for Python tokens
            if i > 0:
                prev_token = tokens[i-1]
                # Add space between most tokens
                if (
                    not prev_token.endswith(("(", "[", "{", ".", ",", ":")) and
                    not token.startswith((".", ")", "]", "}", ",", ":")) and
                    not (prev_token == "-" and token.isdigit())
                ):
                    code += " "
            
            # Add the token to the code
            code += token
        
        return code
    
    def _detokenize_javascript(self, tokens: List[str]) -> str:
        """
        JavaScript-specific detokenization.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed JavaScript code
        """
        code = ""
        for i, token in enumerate(tokens):
            # Special handling for JavaScript tokens
            if i > 0:
                prev_token = tokens[i-1]
                # Add space between most tokens
                if (
                    not prev_token.endswith(("(", "[", "{", ".", ",", ":")) and
                    not token.startswith((".", ")", "]", "}", ",", ":")) and
                    not (prev_token == "-" and token.isdigit())
                ):
                    code += " "
            
            # Add the token to the code
            code += token
        
        return code
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert token strings to token IDs.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token IDs
        """
        # Make sure vocabulary is initialized
        if not self.token_to_id:
            self._initialize_vocabulary()
        
        ids = []
        for token in tokens:
            # Get token ID from vocabulary or add new entry if necessary
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # If vocabulary is not full, add new token
                if len(self.token_to_id) < self.vocab_size:
                    new_id = len(self.token_to_id)
                    self.token_to_id[token] = new_id
                    self.id_to_token[new_id] = token
                    ids.append(new_id)
                else:
                    # Vocabulary is full, use unknown token
                    ids.append(self.unk_id)
        
        return ids
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to token strings.
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of token strings
        """
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                # Unknown ID, use unknown token
                tokens.append("<UNK>")
        
        return tokens
    
    def encode(self, code: str) -> List[int]:
        """
        Encode code as a list of token IDs.
        
        Args:
            code: Code string
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(code)
        ids = self.convert_tokens_to_ids(tokens)
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to code.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Reconstructed code as a string
        """
        tokens = self.convert_ids_to_tokens(ids)
        code = self.detokenize(tokens)
        return code
    
    def idx_to_token(self, idx: int) -> str:
        """
        Convert a token index to its string representation.
        
        Args:
            idx: Token index
            
        Returns:
            Token string
        """
        return self.id_to_token.get(idx, "<UNK>")
    
    def token_to_idx(self, token: str) -> int:
        """
        Convert a token string to its index.
        
        Args:
            token: Token string
            
        Returns:
            Token index
        """
        return self.token_to_id.get(token, self.unk_id)