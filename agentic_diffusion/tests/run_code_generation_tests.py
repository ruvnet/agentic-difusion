#!/usr/bin/env python
"""
Test runner script for code generation components.

This script runs all unit, integration, and system tests for the code generation
components and reports the code coverage. It ensures that we have at least 90%
code coverage across the code generation pipeline.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path


def run_tests(test_type="all", verbose=False):
    """
    Run the specified tests and report coverage.
    
    Args:
        test_type (str): Type of tests to run: "unit", "integration", "system", or "all"
        verbose (bool): Whether to show verbose output
    
    Returns:
        tuple: (success, coverage_data)
            - success (bool): Whether all tests passed
            - coverage_data (dict): Coverage data with percentages by module
    """
    # Determine the pytest arguments based on test type
    if test_type == "unit":
        test_path = "agentic_diffusion/tests/unit/code_generation"
    elif test_type == "integration":
        test_path = "agentic_diffusion/tests/integration/code_generation"
    elif test_type == "system":
        test_path = "agentic_diffusion/tests/system/end_to_end"
    else:  # all
        test_path = " ".join([
            "agentic_diffusion/tests/unit/code_generation",
            "agentic_diffusion/tests/integration/code_generation",
            "agentic_diffusion/tests/system/end_to_end"
        ])
    
    # Build the pytest command
    cmd = [
        "pytest",
        test_path,
        "--cov=agentic_diffusion/code_generation",
        "--cov=agentic_diffusion/adaptation",
        "--cov=agentic_diffusion/api/code_generation_api.py",
        "--cov-report=term",
        "--cov-report=json:coverage.json"
    ]
    
    if verbose:
        cmd.append("-v")
    
    # Run the tests
    print(f"Running {test_type} tests...")
    result = subprocess.run(cmd, capture_output=not verbose)
    success = result.returncode == 0
    
    if not verbose and not success:
        print(result.stdout.decode())
        print(result.stderr.decode())
    
    # Load the coverage data
    coverage_data = {}
    try:
        if os.path.exists("coverage.json"):
            with open("coverage.json") as f:
                coverage_json = json.load(f)
                
            # Extract the relevant coverage data
            coverage_data = {
                "total": coverage_json.get("totals", {}).get("percent_covered", 0),
                "modules": {}
            }
            
            # Get coverage by module
            for file_path, file_data in coverage_json.get("files", {}).items():
                if "code_generation" in file_path or "adaptation" in file_path:
                    module_name = os.path.basename(file_path)
                    coverage_data["modules"][module_name] = file_data.get("summary", {}).get("percent_covered", 0)
    except Exception as e:
        print(f"Error loading coverage data: {e}")
    
    return success, coverage_data


def print_coverage_report(coverage_data):
    """
    Print a formatted coverage report.
    
    Args:
        coverage_data (dict): Coverage data with percentages by module
    """
    print("\n=== Code Coverage Report ===")
    print(f"Total coverage: {coverage_data.get('total', 0):.2f}%")
    
    if coverage_data.get("modules"):
        print("\nCoverage by module:")
        for module, coverage in sorted(coverage_data["modules"].items()):
            print(f"  {module}: {coverage:.2f}%")
    
    # Check if we meet the target coverage
    target_coverage = 90.0
    total_coverage = coverage_data.get("total", 0)
    
    if total_coverage >= target_coverage:
        print(f"\n✓ Target coverage of {target_coverage}% achieved: {total_coverage:.2f}%")
    else:
        print(f"\n✗ Target coverage of {target_coverage}% not achieved: {total_coverage:.2f}%")
        print(f"  Missing: {target_coverage - total_coverage:.2f}%")


def main():
    """Run the test script."""
    parser = argparse.ArgumentParser(description="Run code generation tests and measure coverage")
    parser.add_argument("--type", choices=["unit", "integration", "system", "all"],
                       default="all", help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    # Ensure we're running from the project root
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(project_root)
    
    # Run the tests
    success, coverage_data = run_tests(args.type, args.verbose)
    
    # Print the coverage report
    print_coverage_report(coverage_data)
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())