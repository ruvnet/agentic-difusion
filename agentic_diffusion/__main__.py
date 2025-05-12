#!/usr/bin/env python3
"""
Entry point for Agentic Diffusion.

This module serves as the entry point when the package is run directly.
It provides a command-line interface for common operations such as
generating code, adapting models, and running tests.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml

# Add reward model support
from agentic_diffusion.core.reward_functions import (
    RewardModel, SimpleRewardModel, AdaptDiffuserTestRewardModel, CompositeRewardModel
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("agentic_diffusion")


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Use default config path
        config_dir = Path(__file__).parent.parent / "config"
        config_path = config_dir / "development.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Load AdaptDiffuser config if it exists and command requires it
        if sys.argv[1:] and sys.argv[1] == "adaptdiffuser":
            adaptdiffuser_config_path = config_dir / "adaptdiffuser.yaml"
            if os.path.exists(adaptdiffuser_config_path):
                try:
                    with open(adaptdiffuser_config_path, 'r') as f:
                        adaptdiffuser_config = yaml.safe_load(f)
                    logger.info(f"Loaded AdaptDiffuser configuration from {adaptdiffuser_config_path}")
                    
                    # Merge configurations with adaptdiffuser taking precedence
                    if "adaptdiffuser" not in config:
                        config["adaptdiffuser"] = {}
                    config["adaptdiffuser"].update(adaptdiffuser_config)
                except Exception as e:
                    logger.warning(f"Failed to load AdaptDiffuser configuration: {e}")
        
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic Diffusion: Advanced Diffusion-based Code Generation Framework"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate code command
    generate_parser = subparsers.add_parser("generate", help="Generate code")
    generate_parser.add_argument("prompt", help="Natural language prompt")
    generate_parser.add_argument(
        "--language", "-l", default="python", help="Target programming language"
    )
    generate_parser.add_argument(
        "--output", "-o", help="Output file (if not specified, prints to stdout)"
    )
    generate_parser.add_argument(
        "--approach", "-a", choices=["diffusion", "hybrid"], default="diffusion",
        help="Code generation approach (standard diffusion or hybrid LLM+diffusion)"
    )
    generate_parser.add_argument(
        "--llm-provider", default="openai", 
        help="LLM provider for hybrid approach (openai, anthropic, etc.)"
    )
    generate_parser.add_argument(
        "--llm-model", default="gpt-4", 
        help="LLM model for hybrid approach"
    )
    generate_parser.add_argument(
        "--refinement-iterations", type=int, default=3,
        help="Number of diffusion refinement iterations for hybrid approach"
    )
    generate_parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Temperature for generation (higher = more creative)"
    )
    
    # Adapt model command
    adapt_parser = subparsers.add_parser("adapt", help="Adapt a model")
    adapt_parser.add_argument("task", help="Task description")
    adapt_parser.add_argument(
        "--examples", "-e", help="Path to examples file (JSON format)"
    )
    adapt_parser.add_argument(
        "--model", "-m", default="default", help="Model identifier"
    )
    
    # Run tests command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    test_parser.add_argument(
        "--unit", action="store_true", help="Run only unit tests"
    )
    test_parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    test_parser.add_argument(
        "--system", action="store_true", help="Run only system tests"
    )
    test_parser.add_argument(
        "--hybrid", action="store_true", help="Run hybrid approach tests"
    )
    
    # AdaptDiffuser command
    adaptdiffuser_parser = subparsers.add_parser(
        "adaptdiffuser", help="AdaptDiffuser operations"
    )
    adaptdiffuser_subparsers = adaptdiffuser_parser.add_subparsers(
        dest="adaptdiffuser_command", help="AdaptDiffuser command"
    )
    
    # AdaptDiffuser generate command
    adaptdiffuser_generate = adaptdiffuser_subparsers.add_parser(
        "generate", help="Generate trajectories using AdaptDiffuser"
    )
    adaptdiffuser_generate.add_argument(
        "task", help="Task description or identifier"
    )
    adaptdiffuser_generate.add_argument(
        "--batch-size", "-b", type=int, default=4, help="Number of trajectories to generate"
    )
    adaptdiffuser_generate.add_argument(
        "--guidance-scale", "-g", type=float, default=1.0, help="Scale for reward guidance"
    )
    adaptdiffuser_generate.add_argument(
        "--output", "-o", help="Output file to save trajectories (JSON format)"
    )
    
    # AdaptDiffuser adapt command
    adaptdiffuser_adapt = adaptdiffuser_subparsers.add_parser(
        "adapt", help="Adapt AdaptDiffuser to a specific task"
    )
    adaptdiffuser_adapt.add_argument(
        "task", help="Task description or identifier"
    )
    adaptdiffuser_adapt.add_argument(
        "--examples", "-e", help="Path to examples file (JSON format)"
    )
    adaptdiffuser_adapt.add_argument(
        "--iterations", "-i", type=int, default=5, help="Number of adaptation iterations"
    )
    adaptdiffuser_adapt.add_argument(
        "--batch-size", "-b", type=int, default=16, help="Batch size for adaptation"
    )
    adaptdiffuser_adapt.add_argument(
        "--learning-rate", "-lr", type=float, default=1e-5, help="Learning rate for adaptation"
    )
    adaptdiffuser_adapt.add_argument(
        "--quality-threshold", "-q", type=float, default=0.7, help="Quality threshold for filtering samples"
    )
    adaptdiffuser_adapt.add_argument(
        "--save-checkpoint", "-s", action="store_true", help="Save adaptation checkpoint"
    )
    
    # AdaptDiffuser improve command
    adaptdiffuser_improve = adaptdiffuser_subparsers.add_parser(
        "improve", help="Self-improve AdaptDiffuser on a task"
    )
    adaptdiffuser_improve.add_argument(
        "task", help="Task description or identifier"
    )
    adaptdiffuser_improve.add_argument(
        "--iterations", "-i", type=int, default=3, help="Number of improvement iterations"
    )
    adaptdiffuser_improve.add_argument(
        "--trajectories", "-t", type=int, default=50, help="Trajectories per iteration"
    )
    adaptdiffuser_improve.add_argument(
        "--quality-threshold", "-q", type=float, default=0.7, help="Quality threshold for filtering samples"
    )
    
    # Run server command
    server_parser = subparsers.add_parser("server", help="Run API server")
    server_parser.add_argument(
        "--host", default="localhost", help="Server host"
    )
    server_parser.add_argument(
        "--port", "-p", type=int, default=8000, help="Server port"
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run code generation benchmarks")
    benchmark_parser.add_argument(
        "--dataset", default="benchmark_dataset.json", 
        help="Path to benchmark dataset file"
    )
    benchmark_parser.add_argument(
        "--approaches", choices=["diffusion", "hybrid", "both"], default="both",
        help="Which approaches to benchmark"
    )
    benchmark_parser.add_argument(
        "--output-dir", default="benchmark_results",
        help="Directory to output benchmark results"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.command == "generate":
        logger.info(f"Generating {args.language} code from prompt: {args.prompt}")
        try:
            if args.approach == "diffusion":
                # Use standard diffusion approach
                from agentic_diffusion.api.code_generation_api import CodeGenerationAPI
                from agentic_diffusion.code_generation.code_diffusion import CodeDiffusion
                
                # Initialize the diffusion model
                try:
                    diffusion_model = CodeDiffusion()
                    
                    # Initialize the API with the diffusion model and config
                    api = CodeGenerationAPI(diffusion_model, config)
                    code, metadata = api.generate_code(
                        specification=args.prompt,
                        language=args.language
                    )
                except Exception as e:
                    import traceback
                    logger.error(f"Error in diffusion code generation: {e}")
                    traceback.print_exc()
                    return
            else:  # hybrid approach
                # Use hybrid LLM + diffusion approach
                from agentic_diffusion.api.hybrid_llm_diffusion_api import create_hybrid_llm_diffusion_api
                
                # Create hybrid generation config
                hybrid_config = {
                    "llm_provider": args.llm_provider,
                    "llm_model": args.llm_model,
                    "refinement_iterations": args.refinement_iterations,
                    "temperature": args.temperature,
                    "batch_size": config.get("batch_size", 1),
                    "precision": config.get("precision", "float32"),
                    "device": config.get("device", None)
                }
                
                # Initialize the hybrid API
                api = create_hybrid_llm_diffusion_api(hybrid_config)
                code, metadata = api.generate_code(
                    specification=args.prompt,
                    language=args.language
                )
                
                # Display quality improvement if available
                if "quality" in metadata and "quality_improvement_percentage" in metadata["quality"]:
                    improvement = metadata["quality"]["quality_improvement_percentage"]
                    logger.info(f"Quality improvement: {improvement:.2f}%")
            
            # Output the generated code
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(code)
                logger.info(f"Code written to {args.output}")
            else:
                print("\n" + "=" * 80)
                print(f"Generated {args.language.upper()} Code ({args.approach} approach):")
                print("=" * 80 + "\n")
                print(code)
                print("\n" + "=" * 80)
                
                # Print relevant metrics
                if "quality" in metadata:
                    print("\nQuality Metrics:")
                    for key, value in metadata["quality"].items():
                        if isinstance(value, float):
                            print(f"- {key}: {value:.2f}")
                        else:
                            print(f"- {key}: {value}")
                
                if "timing" in metadata:
                    print("\nTiming Information:")
                    for key, value in metadata["timing"].items():
                        print(f"- {key}: {value:.2f}s")
                
                print("\n")
                
        except ImportError as import_err:
            logger.error(f"Required modules not available: {import_err}")
    
    elif args.command == "adapt":
        logger.info(f"Adapting model to task: {args.task}")
        try:
            from agentic_diffusion.api.adaptation_api import AdaptationAPI
            import json
            
            api = AdaptationAPI(config)
            
            examples = []
            if args.examples:
                with open(args.examples, 'r') as f:
                    examples = json.load(f)
            
            task_id = api.define_task(
                task_description=args.task,
                examples=examples
            )
            
            adaptation_id = api.adapt_model(
                model_id=args.model,
                task_id=task_id
            )
            
            status = api.get_adaptation_status(adaptation_id)
            print(f"Adaptation initiated with ID: {adaptation_id}")
            print(f"Current status: {status}")
        except ImportError:
            logger.error("AdaptationAPI not available. Make sure the package is properly installed.")
    
    elif args.command == "test":
        logger.info("Running tests")
        import subprocess
        
        cmd = ["pytest"]
        
        if args.coverage:
            cmd.append("--cov=agentic_diffusion")
            cmd.append("--cov-report=term")
            cmd.append("--cov-report=html")
        
        if args.unit:
            cmd.append("agentic_diffusion/tests/unit")
        elif args.integration:
            cmd.append("agentic_diffusion/tests/integration")
        elif args.system:
            cmd.append("agentic_diffusion/tests/system")
        elif args.hybrid:
            cmd.append("agentic_diffusion/tests/unit/code_generation/test_hybrid_llm_diffusion_generator.py")
            cmd.append("agentic_diffusion/tests/integration/code_generation/test_hybrid_code_generation.py")
        
        subprocess.run(cmd)
    
    elif args.command == "server":
        logger.info(f"Starting API server on {args.host}:{args.port}")
        try:
            # This is a simple placeholder for server startup
            # In a real implementation, this would use FastAPI or similar
            print(f"Server would start on {args.host}:{args.port}")
            print("This functionality is not yet implemented.")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
    
    elif args.command == "benchmark":
        logger.info(f"Running code generation benchmarks")
        try:
            import json
            import os
            import time
            from datetime import datetime
            
            # Ensure output directory exists
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Load benchmark dataset
            try:
                with open(args.dataset, 'r') as f:
                    dataset = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load benchmark dataset: {e}")
                return
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "dataset": args.dataset,
                "approaches": args.approaches,
                "samples": len(dataset),
                "results": {}
            }
            
            # Run benchmarks for selected approaches
            if args.approaches in ["diffusion", "both"]:
                from agentic_diffusion.api.code_generation_api import CodeGenerationAPI
                from agentic_diffusion.code_generation.code_diffusion import CodeDiffusion
                
                diffusion_model = CodeDiffusion()
                diffusion_api = CodeGenerationAPI(diffusion_model, config)
                
                diffusion_results = []
                logger.info("Running diffusion approach benchmarks...")
                
                for i, sample in enumerate(dataset):
                    logger.info(f"Processing sample {i+1}/{len(dataset)}")
                    start_time = time.time()
                    code, metadata = diffusion_api.generate_code(
                        specification=sample["prompt"],
                        language=sample.get("language", "python")
                    )
                    elapsed = time.time() - start_time
                    
                    # Evaluate code quality
                    quality_metrics = diffusion_api.evaluate_code(code, sample.get("language", "python"))
                    
                    diffusion_results.append({
                        "prompt": sample["prompt"],
                        "language": sample.get("language", "python"),
                        "code": code,
                        "time": elapsed,
                        "quality": quality_metrics
                    })
                
                results["results"]["diffusion"] = {
                    "samples": diffusion_results,
                    "avg_time": sum(r["time"] for r in diffusion_results) / len(diffusion_results),
                    "avg_quality": sum(r["quality"]["overall"] for r in diffusion_results) / len(diffusion_results)
                }
            
            if args.approaches in ["hybrid", "both"]:
                from agentic_diffusion.api.hybrid_llm_diffusion_api import create_hybrid_llm_diffusion_api
                
                hybrid_config = {
                    "llm_provider": "openai",  # Default for benchmark
                    "llm_model": "gpt-4",
                    "refinement_iterations": 3,
                    "temperature": 0.7,
                }
                
                hybrid_api = create_hybrid_llm_diffusion_api(hybrid_config)
                
                hybrid_results = []
                logger.info("Running hybrid approach benchmarks...")
                
                for i, sample in enumerate(dataset):
                    logger.info(f"Processing sample {i+1}/{len(dataset)}")
                    start_time = time.time()
                    code, metadata = hybrid_api.generate_code(
                        specification=sample["prompt"],
                        language=sample.get("language", "python")
                    )
                    elapsed = time.time() - start_time
                    
                    # Evaluate code quality
                    quality_metrics = hybrid_api.evaluate_code(code, sample.get("language", "python"))
                    
                    hybrid_results.append({
                        "prompt": sample["prompt"],
                        "language": sample.get("language", "python"),
                        "code": code,
                        "time": elapsed,
                        "quality": quality_metrics,
                        "improvement": metadata["quality"].get("quality_improvement_percentage", 0)
                    })
                
                results["results"]["hybrid"] = {
                    "samples": hybrid_results,
                    "avg_time": sum(r["time"] for r in hybrid_results) / len(hybrid_results),
                    "avg_quality": sum(r["quality"]["overall"] for r in hybrid_results) / len(hybrid_results),
                    "avg_improvement": sum(r["improvement"] for r in hybrid_results) / len(hybrid_results)
                }
            
            # Compare approaches if both were run
            if args.approaches == "both":
                diff_quality = results["results"]["diffusion"]["avg_quality"]
                hybrid_quality = results["results"]["hybrid"]["avg_quality"]
                quality_improvement = ((hybrid_quality - diff_quality) / diff_quality) * 100
                
                results["comparison"] = {
                    "quality_improvement_percent": quality_improvement,
                    "hybrid_vs_diffusion_time_ratio": results["results"]["hybrid"]["avg_time"] / results["results"]["diffusion"]["avg_time"]
                }
                
                logger.info(f"Overall quality improvement: {quality_improvement:.2f}%")
            
            # Save results
            output_file = os.path.join(args.output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Benchmark results saved to {output_file}")
                
        except ImportError as import_err:
            logger.error(f"Required modules not available: {import_err}")
    
    elif args.command == "adaptdiffuser":
        try:
            from agentic_diffusion.api.adapt_diffuser_api import AdaptDiffuserAPI, create_adapt_diffuser_api
            import json
            import time
            
            # Create AdaptDiffuser API
            api = create_adapt_diffuser_api(config)
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
            
            if args.adaptdiffuser_command == "generate":
                logger.info(f"Generating trajectories for task: {args.task}")
                
                # Start timing
                start_time = time.time()
                
                # Generate trajectories
                trajectories, metadata = api.generate(
                    task=args.task,
                    batch_size=args.batch_size,
                    guidance_scale=args.guidance_scale
                )
                
                # Calculate timing
                elapsed = time.time() - start_time
                
                # Display results
                if trajectories is not None:
                    print(f"\nGenerated {len(trajectories)} trajectories in {elapsed:.2f} seconds")
                    if "rewards" in metadata:
                        print(f"Reward statistics:")
                        print(f"  Mean: {metadata['rewards']['mean']:.4f}")
                        print(f"  Max: {metadata['rewards']['max']:.4f}")
                        print(f"  Min: {metadata['rewards']['min']:.4f}")
                    
                    # Save trajectories if output specified
                    if args.output:
                        # Convert to numpy for JSON serialization
                        trajectory_list = [t.cpu().numpy().tolist() for t in trajectories]
                        output_data = {
                            "trajectories": trajectory_list,
                            "metadata": metadata
                        }
                        
                        with open(args.output, 'w') as f:
                            json.dump(output_data, f)
                        logger.info(f"Saved trajectories to {args.output}")
                
            elif args.adaptdiffuser_command == "adapt":
                logger.info(f"Adapting model to task: {args.task}")
                
                # Load examples if provided
                examples = []
                if args.examples:
                    with open(args.examples, 'r') as f:
                        examples_data = json.load(f)
                        if isinstance(examples_data, list):
                            examples = examples_data
                        elif isinstance(examples_data, dict) and "examples" in examples_data:
                            examples = examples_data["examples"]
                
                # Start timing
                start_time = time.time()
                
                # Run adaptation
                metrics = api.adapt(
                    task=args.task,
                    trajectories=examples if examples else None,
                    iterations=args.iterations,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    quality_threshold=args.quality_threshold,
                    save_checkpoint=args.save_checkpoint,
                    checkpoint_dir="./checkpoints"
                )
                
                # Calculate timing
                elapsed = time.time() - start_time
                
                # Display results
                print(f"\nAdaptation completed in {elapsed:.2f} seconds")
                print(f"Adaptation metrics:")
                for key, value in metrics.items():
                    if isinstance(value, list):
                        if len(value) > 0:
                            print(f"  {key}: {value[-1]:.4f} (final)")
                    else:
                        print(f"  {key}: {value}")
            
            elif args.adaptdiffuser_command == "improve":
                logger.info(f"Self-improving model on task: {args.task}")
                
                # Start timing
                start_time = time.time()
                
                # Run self-improvement
                metrics = api.self_improve(
                    task=args.task,
                    iterations=args.iterations,
                    trajectories_per_iter=args.trajectories,
                    quality_threshold=args.quality_threshold
                )
                
                # Calculate timing
                elapsed = time.time() - start_time
                
                # Display results
                print(f"\nSelf-improvement completed in {elapsed:.2f} seconds")
                print(f"Improvement metrics:")
                if "initial_mean_reward" in metrics and "final_mean_reward" in metrics:
                    initial = metrics["initial_mean_reward"]
                    final = metrics["final_mean_reward"]
                    improvement = ((final - initial) / abs(initial)) * 100 if initial != 0 else 0
                    
                    print(f"  Initial reward: {initial:.4f}")
                    print(f"  Final reward: {final:.4f}")
                    print(f"  Improvement: {improvement:.2f}%")
                
                for key, value in metrics.items():
                    if key not in ["initial_mean_reward", "final_mean_reward"]:
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
                
            else:
                print("Unknown AdaptDiffuser command")
                print("Available commands: generate, adapt, improve")
                
        except ImportError as e:
            logger.error(f"Required modules not available: {e}")
            print(f"Error: {e}")
            print("Make sure the AdaptDiffuser API is properly installed.")
    else:
        # If no command specified, print help
        print("Agentic Diffusion: Advanced Diffusion-based Code Generation Framework")
        print("\nUse one of the following commands:")
        print("  generate       - Generate code from a natural language prompt")
        print("  adapt          - Adapt a model to a specific task")
        print("  test           - Run the test suite")
        print("  server         - Start the API server")
        print("  benchmark      - Run code generation benchmarks")
        print("  adaptdiffuser  - AdaptDiffuser operations")
        print("\nFor more information, use --help")


if __name__ == "__main__":
    main()