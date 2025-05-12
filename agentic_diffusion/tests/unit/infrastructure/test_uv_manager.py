"""
Unit tests for the UV Package Manager.

These tests verify the functionality of the UVManager class for managing
package dependencies using UV.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agentic_diffusion.infrastructure.uv_manager import UVManager


class TestUVManager:
    """Test suite for UVManager class."""

    def test_init_with_default_path(self):
        """Test initialization with default requirements path."""
        manager = UVManager()
        assert manager.requirements_path is not None
        assert manager.requirements_path.name == "requirements.txt"
        assert manager.requirements_path.exists()
    
    def test_init_with_custom_path(self, tmp_path):
        """Test initialization with custom requirements path."""
        custom_path = tmp_path / "custom_requirements.txt"
        custom_path.touch()
        
        manager = UVManager(requirements_path=custom_path)
        assert manager.requirements_path == custom_path
    
    @patch("subprocess.run")
    def test_check_uv_installation_success(self, mock_run):
        """Test checking UV installation when UV is installed."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        manager = UVManager()
        assert manager._check_uv_installation() is True
        mock_run.assert_called_once_with(
            ["uv", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
    
    @patch("subprocess.run")
    def test_check_uv_installation_failure(self, mock_run):
        """Test checking UV installation when UV is not installed."""
        mock_run.side_effect = FileNotFoundError("No such file")
        
        manager = UVManager()
        assert manager._check_uv_installation() is False
    
    @patch("subprocess.run")
    def test_install_uv_success(self, mock_run):
        """Test installing UV successfully."""
        # Set up UVManager with UV not installed
        manager = UVManager()
        manager.uv_installed = False
        
        # Mock successful installation
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        # Test installation
        result = manager.install_uv()
        assert result is True
        assert manager.uv_installed is True
        mock_run.assert_called_once()
    
    @patch("subprocess.run")
    def test_install_dependencies_success(self, mock_run, tmp_path):
        """Test installing dependencies successfully."""
        # Create a test requirements file
        req_path = tmp_path / "requirements.txt"
        req_path.write_text("pytest>=7.0.0\nblack>=23.0.0\n")
        
        # Set up UVManager
        manager = UVManager(requirements_path=req_path)
        manager.uv_installed = True
        
        # Mock successful installation
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        # Test installation without venv
        result = manager.install_dependencies(use_venv=False)
        assert result is True
        mock_run.assert_called_once_with(
            ["uv", "pip", "install", "-r", str(req_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_install_dependencies_with_venv(self, mock_exists, mock_run, tmp_path):
        """Test installing dependencies with virtual environment."""
        # Create a test requirements file
        req_path = tmp_path / "requirements.txt"
        req_path.write_text("pytest>=7.0.0\nblack>=23.0.0\n")
        
        # Set up UVManager
        manager = UVManager(requirements_path=req_path)
        manager.uv_installed = True
        
        # Mock venv doesn't exist, then exists after creation
        mock_exists.return_value = False
        
        # Mock successful command execution
        mock_run.return_value.returncode = 0
        
        # Test installation with venv
        venv_path = tmp_path / ".venv"
        result = manager.install_dependencies(use_venv=True, venv_path=venv_path)
        
        # Verify results
        assert result is True
        
        # Check venv creation command
        venv_call = mock_run.call_args_list[0]
        assert venv_call[0][0] == ["uv", "venv", str(venv_path)]
        
        # Check install command with venv
        install_call = mock_run.call_args_list[1]
        assert install_call[0][0] == ["uv", "pip", "install", "-r", str(req_path), "--venv", str(venv_path)]
    
    @patch("subprocess.run")
    def test_upgrade_packages_specific(self, mock_run):
        """Test upgrading specific packages."""
        # Set up UVManager
        manager = UVManager()
        manager.uv_installed = True
        
        # Mock successful upgrade
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        # Test upgrading specific packages
        packages = ["pytest", "black"]
        result = manager.upgrade_packages(packages=packages, use_venv=False)
        
        # Verify results
        assert result is True
        mock_run.assert_called_once_with(
            ["uv", "pip", "install", "--upgrade", "pytest", "black"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )