import pytest

from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer

@pytest.mark.parametrize("language,code,expected_tokens", [
    ("python", "def foo(x):\n    return x+1", ["def", "foo", "(", "x", ")", ":", "return", "x", "+", "1"]),
    ("javascript", "function foo(x) { return x + 1; }", ["function", "foo", "(", "x", ")", "{", "return", "x", "+", "1", ";", "}"]),
    ("java", "public int foo(int x) { return x + 1; }", ["public", "int", "foo", "(", "int", "x", ")", "{", "return", "x", "+", "1", ";", "}"]),
    ("go", "func foo(x int) int { return x + 1 }", ["func", "foo", "(", "x", "int", ")", "int", "{", "return", "x", "+", "1", "}"]),
])
def test_tokenize_basic(language, code, expected_tokens):
    tokenizer = CodeTokenizer(language=language)
    tokens = tokenizer.tokenize(code)
    # Allow for tokenizer-specific tokenization, but must contain all expected tokens in order
    idx = 0
    for token in tokens:
        if idx < len(expected_tokens) and token == expected_tokens[idx]:
            idx += 1
    assert idx == len(expected_tokens), f"Missing tokens in {language} tokenization: {tokens}"

@pytest.mark.parametrize("language,code", [
    ("python", "def foo(x):\n    return x+1"),
    ("javascript", "function foo(x) { return x + 1; }"),
    ("java", "public int foo(int x) { return x + 1; }"),
    ("go", "func foo(x int) int { return x + 1 }"),
])
def test_detokenize_roundtrip(language, code):
    tokenizer = CodeTokenizer(language=language)
    tokens = tokenizer.tokenize(code)
    detok = tokenizer.detokenize(tokens)
    # Detokenized code should contain all original identifiers and operators
    for part in ["foo", "x", "+", "1"]:
        assert part in detok, f"Detokenized code missing '{part}' for {language}: {detok}"

@pytest.mark.parametrize("language,special_code,special_tokens", [
    ("python", "", []),
    ("python", "# just a comment", ["#", "just", "a", "comment"]),
    ("python", "def\n\n\n", ["def"]),
    ("javascript", "// comment only", ["//", "comment", "only"]),
    ("java", "", []),
    ("go", "\n", []),
])
def test_tokenizer_edge_cases(language, special_code, special_tokens):
    tokenizer = CodeTokenizer(language=language)
    tokens = tokenizer.tokenize(special_code)
    for t in special_tokens:
        assert t in tokens, f"Expected token '{t}' missing in {language} edge case: {tokens}"

def test_tokenizer_handles_special_tokens():
    tokenizer = CodeTokenizer(language="python")
    code = "def foo():\n    return <SPECIAL_TOKEN>"
    tokens = tokenizer.tokenize(code)
    assert "<SPECIAL_TOKEN>" in tokens
    detok = tokenizer.detokenize(tokens)
    assert "<SPECIAL_TOKEN>" in detok

def test_tokenizer_multiline_and_unicode():
    tokenizer = CodeTokenizer(language="python")
    code = "def foo():\n    return 'π = 3.14'\n"
    tokens = tokenizer.tokenize(code)
    assert "π" in tokens or "'π" in tokens
    detok = tokenizer.detokenize(tokens)
    assert "π" in detok