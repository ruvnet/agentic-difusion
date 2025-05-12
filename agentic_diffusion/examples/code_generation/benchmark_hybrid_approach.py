#!/usr/bin/env python3
"""
Benchmark script that compares the standard diffusion approach with the hybrid LLM + diffusion approach.

This script runs benchmarks on a set of code generation tasks and measures the quality improvements
achieved by the hybrid approach compared to the standard diffusion-only approach.
"""

import argparse
import json
import os
import time
from datetime import datetime
import logging

from agentic_diffusion.api.code_generation_api import create_code_generation_api
from agentic_diffusion.api.hybrid_llm_diffusion_api import create_hybrid_llm_diffusion_api
from agentic_diffusion.code_generation.code_diffusion import CodeDiffusion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("benchmark")

# Sample benchmark dataset
SAMPLE_DATASET = [
    {
        "id": "1",
        "prompt": "Write a function to calculate the Fibonacci sequence up to n terms",
        "language": "python"
    },
    {
        "id": "2",
        "prompt": "Create a function that checks if a string is a palindrome",
        "language": "python"
    },
    {
        "id": "3",
        "prompt": "Implement a binary search algorithm",
        "language": "python"
    },
    {
        "id": "4",
        "prompt": "Create a simple REST API using Express.js with endpoints for CRUD operations",
        "language": "javascript"
    },
    {
        "id": "5",
        "prompt": "Write a function to find the maximum subarray sum (Kadane's algorithm)",
        "language": "python"
    }
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Hybrid LLM + Diffusion vs Standard Diffusion"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to JSON benchmark dataset (if not provided, uses built-in samples)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        help="LLM provider for hybrid approach (openai, anthropic, etc.)"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4",
        help="LLM model to use for hybrid approach"
    )
    
    parser.add_argument(
        "--refinement-iterations",
        type=int,
        default=3,
        help="Number of diffusion refinement iterations"
    )
    
    return parser.parse_args()


def load_dataset(dataset_path=None):
    """
    Load benchmark dataset from file or use default samples.
    
    Args:
        dataset_path: Path to dataset JSON file
        
    Returns:
        List of benchmark tasks
    """
    if dataset_path:
        try:
            with open(dataset_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.info("Falling back to built-in sample dataset")
    
    return SAMPLE_DATASET


def save_results(results, output_dir):
    """
    Save benchmark results to a JSON file.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def print_summary(results):
    """
    Print a summary of benchmark results.
    
    Args:
        results: Benchmark results dictionary
    """
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    diffusion_quality = results["results"]["diffusion"]["avg_quality"]
    hybrid_quality = results["results"]["hybrid"]["avg_quality"]
    quality_improvement = results["comparison"]["quality_improvement_percent"]
    
    diffusion_time = results["results"]["diffusion"]["avg_time"]
    hybrid_time = results["results"]["hybrid"]["avg_time"]
    time_increase = ((hybrid_time / diffusion_time) - 1) * 100
    
    print(f"Number of samples: {results['samples']}")
    print(f"\nQUALITY METRICS:")
    print(f"  Diffusion approach: {diffusion_quality:.3f}")
    print(f"  Hybrid approach:    {hybrid_quality:.3f}")
    print(f"  Improvement:        {quality_improvement:.2f}%")
    
    print(f"\nTIMING METRICS:")
    print(f"  Diffusion approach: {diffusion_time:.2f}s")
    print(f"  Hybrid approach:    {hybrid_time:.2f}s")
    print(f"  Time increase:      {time_increase:.2f}%")
    
    print(f"\nINDIVIDUAL SAMPLE IMPROVEMENTS:")
    for sample_id, improvement in results["comparison"]["sample_improvements"].items():
        print(f"  Sample {sample_id}: {improvement:.2f}%")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point for benchmarking."""
    args = parse_args()
    
    # Load benchmark dataset
    dataset = load_dataset(args.dataset)
    logger.info(f"Loaded benchmark dataset with {len(dataset)} samples")
    
    # Initialize APIs
    diffusion_model = CodeDiffusion()
    diffusion_api = create_code_generation_api(diffusion_model)
    
    hybrid_config = {
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "refinement_iterations": args.refinement_iterations,
        "temperature": 0.7,
    }
    hybrid_api = create_hybrid_llm_diffusion_api(hybrid_config)
    
    # Prepare results structure
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "refinement_iterations": args.refinement_iterations,
        },
        "samples": len(dataset),
        "results": {
            "diffusion": {"samples": []},
            "hybrid": {"samples": []},
        },
        "comparison": {
            "sample_improvements": {}
        }
    }
    
    # Run diffusion benchmarks
    logger.info("Running standard diffusion benchmarks...")
    diffusion_samples = []
    
    for i, sample in enumerate(dataset):
        sample_id = sample.get("id", str(i+1))
        logger.info(f"Processing sample {sample_id}/{len(dataset)} with diffusion approach")
        
        start_time = time.time()
        code, metadata = diffusion_api.generate_code(
            specification=sample["prompt"],
            language=sample.get("language", "python")
        )
        elapsed = time.time() - start_time
        
        # Evaluate code quality
        quality_metrics = diffusion_api.evaluate_code(
            code,
            language=sample.get("language", "python")
        )
        
        diffusion_samples.append({
            "id": sample_id,
            "prompt": sample["prompt"],
            "language": sample.get("language", "python"),
            "code": code,
            "time": elapsed,
            "quality": quality_metrics
        })
    
    # Calculate diffusion averages
    diffusion_avg_time = sum(s["time"] for s in diffusion_samples) / len(diffusion_samples)
    diffusion_avg_quality = sum(s["quality"]["overall"] for s in diffusion_samples) / len(diffusion_samples)
    
    results["results"]["diffusion"] = {
        "samples": diffusion_samples,
        "avg_time": diffusion_avg_time,
        "avg_quality": diffusion_avg_quality
    }
    
    # Run hybrid benchmarks
    logger.info("Running hybrid LLM + diffusion benchmarks...")
    hybrid_samples = []
    
    for i, sample in enumerate(dataset):
        sample_id = sample.get("id", str(i+1))
        logger.info(f"Processing sample {sample_id}/{len(dataset)} with hybrid approach")
        
        start_time = time.time()
        code, metadata = hybrid_api.generate_code(
            specification=sample["prompt"],
            language=sample.get("language", "python")
        )
        elapsed = time.time() - start_time
        
        # Evaluate code quality
        quality_metrics = hybrid_api.evaluate_code(
            code,
            language=sample.get("language", "python"),
            specification=sample["prompt"]
        )
        
        # Store improvement percentage for this sample
        improvement = metadata["quality"].get("quality_improvement_percentage", 0)
        
        hybrid_samples.append({
            "id": sample_id,
            "prompt": sample["prompt"],
            "language": sample.get("language", "python"),
            "code": code,
            "time": elapsed,
            "quality": quality_metrics,
            "improvement": improvement
        })
        
        # Store individual sample improvements for comparison
        results["comparison"]["sample_improvements"][sample_id] = improvement
    
    # Calculate hybrid averages
    hybrid_avg_time = sum(s["time"] for s in hybrid_samples) / len(hybrid_samples)
    hybrid_avg_quality = sum(s["quality"]["overall"] for s in hybrid_samples) / len(hybrid_samples)
    hybrid_avg_improvement = sum(s["improvement"] for s in hybrid_samples) / len(hybrid_samples)
    
    results["results"]["hybrid"] = {
        "samples": hybrid_samples,
        "avg_time": hybrid_avg_time,
        "avg_quality": hybrid_avg_quality,
        "avg_improvement": hybrid_avg_improvement
    }
    
    # Calculate overall quality improvement
    quality_improvement = ((hybrid_avg_quality - diffusion_avg_quality) / diffusion_avg_quality) * 100
    
    results["comparison"].update({
        "quality_improvement_percent": quality_improvement,
        "hybrid_vs_diffusion_time_ratio": hybrid_avg_time / diffusion_avg_time
    })
    
    # Save and print results
    save_results(results, args.output_dir)
    print_summary(results)


if __name__ == "__main__":
    main()