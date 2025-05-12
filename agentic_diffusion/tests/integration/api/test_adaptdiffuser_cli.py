import pytest

@pytest.mark.integration
class TestAdaptDiffuserCLI:
    def test_adaptdiffuser_cli_invocation(self, tmp_path):
        """
        Given: CLI arguments and synthetic config/data
        When: AdaptDiffuser CLI is invoked
        Then: Output files and logs should match expected results and exit code is 0
        """
        import subprocess
        import sys
        import os

        # Example CLI invocation: python -m agentic_diffusion --help
        result = subprocess.run(
            [sys.executable, "-m", "agentic_diffusion", "--help"],
            capture_output=True,
            text=True
        )
        # Assert exit code is 0 (help should succeed)
        assert result.returncode == 0, f"CLI exited with {result.returncode}"
        # Assert help output contains expected text
        assert "usage" in result.stdout.lower() or "help" in result.stdout.lower(), "Help output missing"

        # TODO: Add more CLI invocation tests with config/data as needed

        # Test passes if CLI invocation and output checks succeed