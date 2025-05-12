#!/usr/bin/env python3
"""
Script to inject reward model support into AdaptDiffuser API.

This script adds the necessary code to initialize and use reward models
in the main AdaptDiffuser CLI commands.
"""

import os
import sys
import logging
import argparse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backup_file(file_path):
    """Create a backup of a file."""
    backup_path = f"{file_path}.bak"
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False


def inject_into_main_module():
    """Inject reward model support into main module."""
    main_module_path = os.path.join("agentic_diffusion", "__main__.py")
    
    if not os.path.exists(main_module_path):
        logger.error(f"Main module not found at {main_module_path}")
        return False
    
    # Create backup
    if not backup_file(main_module_path):
        return False
    
    # Read current content
    with open(main_module_path, 'r') as f:
        content = f.read()
    
    # Add imports if needed
    new_content = []
    import_section_end = False
    import_added = False
    
    for line in content.split('\n'):
        if not import_section_end and not line.strip() and not import_added:
            # Found the end of the import section
            import_section_end = True
            # Add our imports
            new_content.append("# Add reward model support")
            new_content.append("from agentic_diffusion.core.reward_functions import SimpleRewardModel, AdaptDiffuserTestRewardModel, CompositeRewardModel")
            import_added = True
        
        new_content.append(line)
    
    # Find create_adapt_diffuser_api call
    target_line = "api = create_adapt_diffuser_api(config)"
    reward_model_code = """
            # Initialize reward model if configured
            if 'adaptdiffuser' in config and 'reward_model' in config['adaptdiffuser']:
                reward_config = config['adaptdiffuser']['reward_model']
                reward_type = reward_config.get('type', 'simple')
                
                logger.info(f"Initializing reward model of type '{reward_type}'")
                
                if reward_type == 'test':
                    reward_model = AdaptDiffuserTestRewardModel(reward_config)
                elif reward_type == 'simple':
                    reward_model = SimpleRewardModel(reward_config)
                elif reward_type == 'composite':
                    reward_model = CompositeRewardModel(reward_config)
                else:
                    reward_model = SimpleRewardModel(reward_config)
                
                # Register reward model with API
                if hasattr(api, 'register_reward_model'):
                    api.register_reward_model(reward_model)
                    logger.info("Reward model registered with AdaptDiffuser API")
"""
    
    modified_content = []
    for line in new_content:
        modified_content.append(line)
        if line.strip() == target_line:
            # Insert reward model initialization after API creation
            for reward_line in reward_model_code.split('\n'):
                if reward_line.strip():
                    modified_content.append(reward_line)
    
    # Write the modified content
    with open(main_module_path, 'w') as f:
        f.write('\n'.join(modified_content))
    
    logger.info(f"Successfully injected reward model support into {main_module_path}")
    return True


def inject_into_api_module():
    """Add register_reward_model method to AdaptDiffuserAPI if not present."""
    api_module_path = os.path.join("agentic_diffusion", "api", "adapt_diffuser_api.py")
    
    if not os.path.exists(api_module_path):
        logger.error(f"API module not found at {api_module_path}")
        return False
    
    # Create backup
    if not backup_file(api_module_path):
        return False
    
    # Check if method already exists
    with open(api_module_path, 'r') as f:
        content = f.read()
    
    if "def register_reward_model" in content:
        logger.info("register_reward_model method already exists in API module")
        return True
    
    # Find the AdaptDiffuserAPI class
    api_class_pattern = "class AdaptDiffuserAPI:"
    api_method_content = """
    def register_reward_model(self, reward_model):
        \"\"\"Register a reward model with the AdaptDiffuser API.\"\"\"
        logger.info("Registering reward model with AdaptDiffuser API")
        self.reward_model = reward_model
        
        # Register with the underlying model if it exists
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'register_reward_model'):
                self.model.register_reward_model(reward_model)
            else:
                # Direct assignment
                self.model.reward_model = reward_model
        
        return True
"""
    
    if api_class_pattern not in content:
        logger.error(f"Could not find AdaptDiffuserAPI class in {api_module_path}")
        return False
    
    # Find a good place to insert the method
    lines = content.split('\n')
    modified_lines = []
    in_api_class = False
    method_added = False
    
    for i, line in enumerate(lines):
        modified_lines.append(line)
        
        if not in_api_class and api_class_pattern in line:
            in_api_class = True
            continue
        
        if in_api_class and not method_added:
            # Look for a good insertion point - after the last method
            if i < len(lines) - 1 and line.strip() == '' and lines[i+1].strip().startswith('def '):
                # Insert before the next method
                for method_line in api_method_content.split('\n'):
                    if method_line.strip():
                        modified_lines.append(method_line)
                method_added = True
                continue
    
    # If we couldn't find a good insertion point, add at the end of the class
    if in_api_class and not method_added:
        # Find the end of the class
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith('def ') and lines[i-1].strip() == '':
                # Insert after the last method
                modified_lines.append('')
                for method_line in api_method_content.split('\n'):
                    if method_line.strip():
                        modified_lines.append(method_line)
                method_added = True
                break
    
    if not method_added:
        logger.error("Could not find a suitable insertion point for register_reward_model method")
        return False
    
    # Write the modified content
    with open(api_module_path, 'w') as f:
        f.write('\n'.join(modified_lines))
    
    logger.info(f"Successfully added register_reward_model method to {api_module_path}")
    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Inject reward model support")
    parser.add_argument("--skip-main", action="store_true", help="Skip modifying the main module")
    parser.add_argument("--skip-api", action="store_true", help="Skip modifying the API module")
    
    args = parser.parse_args()
    
    success = True
    
    # Inject into main module
    if not args.skip_main:
        if not inject_into_main_module():
            logger.error("Failed to inject into main module")
            success = False
    
    # Inject into API module
    if not args.skip_api:
        if not inject_into_api_module():
            logger.error("Failed to inject into API module")
            success = False
    
    if success:
        logger.info("Successfully injected reward model support")
        return 0
    else:
        logger.error("Failed to inject reward model support")
        return 1


if __name__ == "__main__":
    sys.exit(main())