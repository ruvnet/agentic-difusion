#!/usr/bin/env python3
"""
Custom test script to verify whether the code generator uses real diffusion or templates.
This script will use an uncommon prompt that wouldn't match any templates.
"""

import sys
import logging
import time

from agentic_diffusion.code_generation.code_diffusion import CodeDiffusion
from agentic_diffusion.api.code_generation_api import create_code_generation_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("custom_test")

def run_custom_test():
    """Run test with custom prompt to check for template-based generation."""
    
    # Uncommon prompt that wouldn't match typical templates
    custom_prompt = "Create a function that simulates the movement of ants in a 2D grid based on pheromone trails"
    
    print("\n" + "=" * 80)
    print("TESTING CODE GENERATION WITH CUSTOM PROMPT")
    print("=" * 80)
    print(f"Prompt: {custom_prompt}")
    print("=" * 80)
    
    logger.info("Initializing code generation API")
    diffusion_model = CodeDiffusion()
    api = create_code_generation_api(diffusion_model)
    
    # Generate code
    start_time = time.time()
    try:
        print("\nGenerating code...")
        code, metadata = api.generate_code(
            specification=custom_prompt,
            language="python"
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\nGenerated code in {elapsed_time:.2f}s:")
        print("-" * 40)
        print(code)
        print("-" * 40)
        
        # Check for template indicators
        template_indicators = [
            "TODO: Implement",
            "Implementation for:",
            "# TODO: Complete implementation"
        ]
        
        uses_template = any(indicator in code for indicator in template_indicators)
        
        print("\nANALYSIS:")
        if uses_template:
            print("✘ Code appears to be using templates rather than genuine diffusion")
            for indicator in template_indicators:
                if indicator in code:
                    print(f"  - Found template indicator: '{indicator}'")
        else:
            print("✓ Code appears to be genuinely generated rather than template-based")
            
        # Check for customization to our specific prompt
        ant_related_terms = ["ant", "pheromone", "grid", "trail", "move"]
        has_relevant_terms = any(term in code.lower() for term in ant_related_terms)
        
        if has_relevant_terms:
            print("✓ Code contains terms specific to our prompt")
            for term in ant_related_terms:
                if term in code.lower():
                    print(f"  - Found relevant term: '{term}'")
        else:
            print("✘ Code lacks terms specific to our prompt")
            
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        print(f"\nError: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = run_custom_test()
    sys.exit(0 if success else 1)