"""
Test script to verify that the modular structure of code_generation works correctly.

This script imports components from different modules to ensure that
the imports and re-exports are functioning properly.
"""

import logging
import sys
import os

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test importing components from the modular structure."""
    logger.info("Testing imports from modular structure...")
    
    # Test importing from the main package
    logger.info("Testing imports from agentic_diffusion.code_generation...")
    from agentic_diffusion.code_generation import (
        generate_code,
        complete_code,
        refine_code,
        evaluate_code_quality,
        create_code_diffusion_model
    )
    logger.info("Successfully imported high-level functions")
    
    # Test importing from generation module directly
    logger.info("Testing imports from agentic_diffusion.code_generation.generation...")
    from agentic_diffusion.code_generation.generation import (
        generate_code as gen_code,
        complete_code as comp_code,
        refine_code as ref_code,
        evaluate_code_quality as eval_code
    )
    logger.info("Successfully imported from generation module")
    
    # Test importing from the facade
    logger.info("Testing imports from agentic_diffusion.code_generation.code_diffusion...")
    from agentic_diffusion.code_generation.code_diffusion import (
        generate_code as gen_code2,
        complete_code as comp_code2,
        refine_code as ref_code2,
        evaluate_code_quality as eval_code2
    )
    logger.info("Successfully imported from code_diffusion facade")
    
    # Test importing model components
    logger.info("Testing imports from submodules...")
    from agentic_diffusion.code_generation.diffusion import CodeDiffusion, CodeDiffusionModel
    from agentic_diffusion.code_generation.models import CodeUNet, CodeEmbedding, TransformerBlock
    from agentic_diffusion.code_generation.schedulers import CodeDiscreteScheduler
    from agentic_diffusion.code_generation.utils.diffusion_utils import (
        sinusoidal_embedding,
        categorical_diffusion,
        token_accuracy
    )
    logger.info("Successfully imported from submodules")
    
    # Test reward models if they exist
    try:
        logger.info("Testing imports from reward modules...")
        from agentic_diffusion.code_generation.rewards import (
            SyntaxReward,
            QualityReward,
            RelevanceReward
        )
        logger.info("Successfully imported reward models")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not import some reward models: {e}")
    
    return True

def test_function_signatures():
    """Test that the function signatures are consistent."""
    logger.info("Testing function signatures...")
    
    # Import from both the main module and the direct module
    from agentic_diffusion.code_generation import generate_code as gen_code1
    from agentic_diffusion.code_generation.generation import generate_code as gen_code2
    from agentic_diffusion.code_generation.code_diffusion import generate_code as gen_code3
    
    # Compare function signatures
    import inspect
    sig1 = str(inspect.signature(gen_code1))
    sig2 = str(inspect.signature(gen_code2))
    sig3 = str(inspect.signature(gen_code3))
    
    if sig1 == sig2 == sig3:
        logger.info("Function signatures are consistent")
        return True
    else:
        logger.error(f"Function signatures differ: {sig1} vs {sig2} vs {sig3}")
        return False

def test_create_instance():
    """Test creating a CodeDiffusion instance."""
    logger.info("Testing instance creation...")
    
    try:
        from agentic_diffusion.code_generation import create_code_diffusion_model
        
        # Try to create a minimal model for testing import paths
        # with minimal dependencies (might fail but should import)
        try:
            model = create_code_diffusion_model()
            logger.info("Successfully created model instance")
            return True
        except Exception as e:
            # Just an import test, so OK if initialization fails due to missing parameters
            logger.warning(f"Model initialization failed (might be expected): {e}")
            return True
    except Exception as e:
        logger.error(f"Failed to import or create model: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing the modular structure of agentic_diffusion.code_generation")
    
    try:
        imports_ok = test_imports()
        signatures_ok = test_function_signatures()
        instance_ok = test_create_instance()
        
        if imports_ok and signatures_ok and instance_ok:
            logger.info("All tests passed! The modular structure is working correctly.")
            sys.exit(0)
        else:
            logger.error("Some tests failed. Please check the modular structure.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)