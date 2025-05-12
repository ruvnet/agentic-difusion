"""
UV Package Manager for Agentic Diffusion.

This module provides utilities for managing package dependencies using UV,
a fast pip-compatible installer for Python written in Rust.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

class UVManager:
    """
    Manages Python package dependencies using UV.
    
    This class provides methods for installing, upgrading, and managing
    dependencies using UV, a fast and efficient Python package installer.
    """
    
    def __init__(self, requirements_path: Optional[Union[str, Path]] = None):
        """
        Initialize the UV Manager.
        
        Args:
            requirements_path: Path to requirements.txt file. Defaults to
                project root requirements.txt if None.
        """
        if requirements_path is None:
            self.requirements_path = self._get_default_requirements_path()
        else:
            self.requirements_path = Path(requirements_path)
        
        # Check if UV is installed
        self.uv_installed = self._check_uv_installation()
    
    def _get_default_requirements_path(self) -> Path:
        """
        Get the default requirements path (project root requirements.txt).
        
        Returns:
            Path to requirements.txt
        """
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        return project_root / "requirements.txt"
    
    def _check_uv_installation(self) -> bool:
        """
        Check if UV is installed.
        
        Returns:
            True if UV is installed, False otherwise
        """
        try:
            result = subprocess.run(
                ["uv", "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def install_uv(self) -> bool:
        """
        Install UV if not already installed.
        
        Returns:
            True if UV is installed (either previously or newly installed),
            False if installation failed.
        """
        if self.uv_installed:
            logger.info("UV already installed")
            return True
        
        logger.info("Installing UV...")
        try:
            # Install UV using pip or pipx
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "uv"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info("UV installed successfully")
            self.uv_installed = True
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install UV: {e.stderr}")
            return False
    
    def install_dependencies(self, requirements_path: Optional[Path] = None, 
                           use_venv: bool = True, venv_path: Optional[Path] = None) -> bool:
        """
        Install dependencies using UV.
        
        Args:
            requirements_path: Path to requirements.txt. If None, uses default path.
            use_venv: Whether to install dependencies in a virtual environment
            venv_path: Path to virtual environment. If None and use_venv is True,
                creates a .venv in the project root.
                
        Returns:
            True if dependencies are installed successfully, False otherwise
        """
        if not self.uv_installed and not self.install_uv():
            logger.error("Cannot install dependencies as UV is not installed")
            return False
        
        req_path = requirements_path or self.requirements_path
        
        # Prepare command
        cmd = ["uv", "pip", "install", "-r", str(req_path)]
        
        # Set up virtual environment if requested
        if use_venv:
            if venv_path is None:
                venv_path = self._get_default_requirements_path().parent / ".venv"
            
            # Create venv if it doesn't exist
            if not os.path.exists(venv_path):
                logger.info(f"Creating virtual environment at {venv_path}")
                venv_cmd = ["uv", "venv", str(venv_path)]
                try:
                    subprocess.run(venv_cmd, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to create virtual environment: {e}")
                    return False
            
            # Add venv path to command
            cmd.extend(["--venv", str(venv_path)])
        
        # Run installation
        try:
            logger.info(f"Installing dependencies from {req_path}")
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e.stderr}")
            return False
    
    def upgrade_packages(self, packages: Optional[List[str]] = None, 
                       use_venv: bool = True, venv_path: Optional[Path] = None) -> bool:
        """
        Upgrade specified packages or all packages if none specified.
        
        Args:
            packages: List of packages to upgrade. If None, upgrades all.
            use_venv: Whether to use a virtual environment
            venv_path: Path to virtual environment
            
        Returns:
            True if upgrade successful, False otherwise
        """
        if not self.uv_installed and not self.install_uv():
            logger.error("Cannot upgrade packages as UV is not installed")
            return False
        
        # Prepare command
        cmd = ["uv", "pip", "install", "--upgrade"]
        
        if packages:
            cmd.extend(packages)
        else:
            # Get all packages from requirements.txt
            with open(self.requirements_path, 'r') as f:
                # Parse requirements file, skipping comments and empty lines
                package_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                # Extract package names (remove version specifiers)
                packages = [line.split('>=')[0].split('==')[0].split('<')[0].strip() for line in package_lines]
            cmd.extend(packages)
        
        # Set up virtual environment if requested
        if use_venv:
            if venv_path is None:
                venv_path = self._get_default_requirements_path().parent / ".venv"
            
            # Add venv path to command
            cmd.extend(["--venv", str(venv_path)])
        
        # Run upgrade
        try:
            logger.info(f"Upgrading packages: {', '.join(packages) if packages else 'all'}")
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info("Packages upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upgrade packages: {e.stderr}")
            return False