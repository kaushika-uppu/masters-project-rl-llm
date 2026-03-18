#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation import Evaluator, get_benchmarks
from src import get_inference_function, get_available_functions

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on benchmarks")
    parser.add_argument(
    "--inference-fn",
        type=str, 
        required=True,  # Make it required
        choices=get_available_functions(), 
        help=f"Inference function to use. Available: {get_available_functions()}"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        help=f"Benchmarks to evaluate on. Available: {get_benchmarks()}"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Directory to save results (default: evaluation/results)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with progress bars"
    )

    return parser.parse_args()

def print_summary(results: dict):
    """Print a summary of evaluation results."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    for benchmark_name, benchmark_results in results.get("results", {}).items():
        print(f"\n{benchmark_name.upper()}:")
        print("-" * 70)
        
        metrics = benchmark_results.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name:.<30} {metric_value:.4f}")
        
        num_samples = benchmark_results.get("num_samples", 0)
        print(f"  {'Total samples':.<30} {num_samples}")
    
    print("\n" + "="*70)

def main():
    args = parse_args()

    benchmarks = args.benchmarks
    workers = args.workers
    output_dir = args.output_dir
    inference_fn = get_inference_function(args.inference_fn)

    evaluator = Evaluator(
        inference_fn=inference_fn,
        workers=workers,
        benchmarks=benchmarks,
        output_dir=output_dir,
        verbose=args.verbose
    )

    # Run evaluation
    try:
        results = evaluator.evaluate()
        
        # Print summary
        if not args.verbose:
            print_summary(results)
                
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        sys.exit(1)


if __name__ == "__main__":
    main()
