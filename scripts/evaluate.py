#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation import Evaluator, get_benchmarks
from src import get_inference_function, get_available_functions

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"

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

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of benchmark problems to run"
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index for the dataset (used for parallel array jobs)"
    )

    parser.add_argument(
        "--trimmed",
        action="store_true",
        help="Restrict each benchmark to ids listed in "
             "<trimmed-ids-dir>/<benchmark_name>.txt (one id per line). "
             "If the file is missing the benchmark runs in full."
    )

    parser.add_argument(
        "--trimmed-ids-dir",
        type=str,
        default="evaluation/trimmed_ids",
        help="Directory holding per-benchmark trimmed id files "
             "(default: evaluation/trimmed_ids)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Model id/path for deeptheorem_judge step verification (default: {DEFAULT_JUDGE_MODEL})."
    )
    parser.add_argument(
        "--judge-load-in-4bit",
        action="store_true",
        help="Load the deeptheorem_judge verifier model in 4-bit mode."
    )
    parser.add_argument(
        "--judge-torch-dtype",
        type=str,
        default="auto",
        help="torch_dtype passed to the deeptheorem_judge verifier model."
    )
    parser.add_argument(
        "--judge-device-map",
        type=str,
        default="auto",
        help="device_map passed to the deeptheorem_judge verifier model."
    )
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for each verifier judgement."
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
    if args.judge_model:
        os.environ["DEEPTHEOREM_JUDGE_MODEL"] = args.judge_model
        os.environ["DEEPTHEOREM_JUDGE_LOAD_IN_4BIT"] = str(args.judge_load_in_4bit)
        os.environ["DEEPTHEOREM_JUDGE_TORCH_DTYPE"] = args.judge_torch_dtype
        os.environ["DEEPTHEOREM_JUDGE_DEVICE_MAP"] = args.judge_device_map
        os.environ["DEEPTHEOREM_JUDGE_MAX_NEW_TOKENS"] = str(args.judge_max_new_tokens)

    evaluator = Evaluator(
        inference_fn=inference_fn,
        workers=workers,
        benchmarks=benchmarks,
        output_dir=output_dir,
        verbose=args.verbose,
        limit=args.limit,
        offset=args.offset,
        trimmed=args.trimmed,
        trimmed_ids_dir=args.trimmed_ids_dir,
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
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
