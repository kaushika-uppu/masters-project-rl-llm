#!/usr/bin/env python3
"""
Merge results from SLURM array jobs into a single summary file.
Aggregates metrics across all jobs and generates statistical analysis.
"""
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Merge SLURM array job results")
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="SLURM job ID of the array job"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="evaluation/results",
        help="Base directory containing results"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gsm8k",
        help="Benchmark name to merge results for"
    )
    return parser.parse_args()


def load_job_results(job_dir: Path, benchmark_name: str) -> List[Dict[str, Any]]:
    """Load all JSON result files from a job directory."""
    json_files = sorted(job_dir.glob("eval_results_array_*.json"))
    
    loaded_results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if benchmark_name in data.get('results', {}):
                    loaded_results.append({
                        'filepath': str(json_file),
                        'data': data,
                        'array_task_id': data.get('metadata', {}).get('array_task_id')
                    })
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return loaded_results


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile from list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(len(sorted_values) * percentile)
    return sorted_values[min(idx, len(sorted_values) - 1)]


def aggregate_metrics(job_results: List[Dict], benchmark_name: str) -> Dict[str, Any]:
    """Aggregate metrics from all job results."""
    all_scores = []
    all_latencies = []
    all_prompt_tokens = []
    all_output_tokens = []
    all_tokens_per_second = []
    total_correct = 0
    total_samples = 0
    
    per_job_summary = []
    
    for job in sorted(job_results, key=lambda x: x.get('array_task_id', 0)):
        benchmark_data = job['data']['results'].get(benchmark_name, {})
        job_metrics = benchmark_data.get('metrics', {})
        job_metadata = benchmark_data.get('batch_metadata', {})
        
        # Per-job stats
        job_summary = {
            'array_task_id': job.get('array_task_id'),
            'offset': job_metadata.get('offset'),
            'limit': job_metadata.get('limit'),
            'num_samples': benchmark_data.get('num_samples', 0),
            'accuracy': job_metrics.get('accuracy', 0.0),
            'total_correct': job_metrics.get('total_correct', 0),
            'avg_latency_ms': job_metrics.get('avg_latency_ms'),
            'batch_duration_s': job_metadata.get('batch_duration_s')
        }
        per_job_summary.append(job_summary)
        
        # Aggregate totals
        total_samples += benchmark_data.get('num_samples', 0)
        total_correct += job_metrics.get('total_correct', 0)
        
        # Collect individual results for detailed stats
        for result in benchmark_data.get('results', []):
            score = result.get('score', 0)
            all_scores.append(score)
            
            metrics = result.get('metrics', {})
            if 'latency_ms' in metrics:
                all_latencies.append(metrics['latency_ms'])
            if 'prompt_tokens' in metrics:
                all_prompt_tokens.append(metrics['prompt_tokens'])
            if 'output_tokens' in metrics:
                all_output_tokens.append(metrics['output_tokens'])
            if 'tokens_per_second' in metrics:
                all_tokens_per_second.append(metrics['tokens_per_second'])
    
    # Calculate aggregated statistics
    aggregated = {
        'total_samples': total_samples,
        'total_correct': total_correct,
        'overall_accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
        'num_jobs': len(job_results)
    }
    
    # Latency statistics
    if all_latencies:
        aggregated['latency_stats'] = {
            'mean_ms': statistics.mean(all_latencies),
            'median_ms': statistics.median(all_latencies),
            'min_ms': min(all_latencies),
            'max_ms': max(all_latencies),
            'p50_ms': calculate_percentile(all_latencies, 0.50),
            'p95_ms': calculate_percentile(all_latencies, 0.95),
            'p99_ms': calculate_percentile(all_latencies, 0.99),
            'stdev_ms': statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0
        }
    
    # Token statistics
    if all_prompt_tokens:
        aggregated['token_stats'] = {
            'total_prompt_tokens': sum(all_prompt_tokens),
            'total_output_tokens': sum(all_output_tokens) if all_output_tokens else 0,
            'total_tokens': sum(all_prompt_tokens) + sum(all_output_tokens) if all_output_tokens else sum(all_prompt_tokens),
            'avg_prompt_tokens': statistics.mean(all_prompt_tokens),
            'avg_output_tokens': statistics.mean(all_output_tokens) if all_output_tokens else 0
        }
    
    # Throughput statistics
    if all_tokens_per_second:
        aggregated['throughput_stats'] = {
            'mean_tokens_per_second': statistics.mean(all_tokens_per_second),
            'median_tokens_per_second': statistics.median(all_tokens_per_second)
        }
    
    return aggregated, per_job_summary


def main():
    args = parse_args()

    # Find job directory
    results_base = Path(args.results_dir)
    job_dir = results_base / f"job_{args.job_id}"

    if not job_dir.exists():
        print(f"ERROR: Job directory not found: {job_dir}")
        return

    print(f"Merging results from: {job_dir}")

    # Load all job results
    job_results = load_job_results(job_dir, args.benchmark)

    if not job_results:
        print(f"ERROR: No result files found for benchmark: {args.benchmark}")
        return

    print(f"Found {len(job_results)} result files")

    # Aggregate metrics
    aggregated, per_job_summary = aggregate_metrics(job_results, args.benchmark)

    # Create summary
    summary = {
        'metadata': {
            'job_id': args.job_id,
            'benchmark': args.benchmark,
            'merge_timestamp': datetime.now().isoformat(),
            'num_jobs_merged': len(job_results)
        },
        'aggregated_metrics': aggregated,
        'per_job_breakdown': per_job_summary
    }

    # Save JSON summary
    summary_json_file = job_dir / f"SUMMARY_{args.benchmark}.json"
    with open(summary_json_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nJSON summary saved to: {summary_json_file}")

    # Create text summary
    summary_text_file = job_dir / f"SUMMARY_{args.benchmark}.txt"
    with open(summary_text_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"EVALUATION SUMMARY - {args.benchmark.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Job ID: {args.job_id}\n")
        f.write(f"Benchmark: {args.benchmark}\n")
        f.write(f"Number of array jobs: {len(job_results)}\n")
        f.write(f"Merge timestamp: {summary['metadata']['merge_timestamp']}\n")
        f.write("\n" + "-"*80 + "\n")
        f.write("OVERALL RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples: {aggregated['total_samples']}\n")
        f.write(f"Total correct: {aggregated['total_correct']}\n")
        f.write(f"Overall accuracy: {aggregated['overall_accuracy']:.4f} ({aggregated['overall_accuracy']*100:.2f}%)\n")

        if 'latency_stats' in aggregated:
            f.write("\n" + "-"*80 + "\n")
            f.write("LATENCY STATISTICS (milliseconds)\n")
            f.write("-"*80 + "\n")
            lat = aggregated['latency_stats']
            f.write(f"Mean: {lat['mean_ms']:.2f} ms\n")
            f.write(f"Median: {lat['median_ms']:.2f} ms\n")
            f.write(f"Min: {lat['min_ms']:.2f} ms\n")
            f.write(f"Max: {lat['max_ms']:.2f} ms\n")
            f.write(f"P50: {lat['p50_ms']:.2f} ms\n")
            f.write(f"P95: {lat['p95_ms']:.2f} ms\n")
            f.write(f"P99: {lat['p99_ms']:.2f} ms\n")
            f.write(f"Std Dev: {lat['stdev_ms']:.2f} ms\n")

        if 'token_stats' in aggregated:
            f.write("\n" + "-"*80 + "\n")
            f.write("TOKEN STATISTICS\n")
            f.write("-"*80 + "\n")
            tok = aggregated['token_stats']
            f.write(f"Total prompt tokens: {tok['total_prompt_tokens']:,}\n")
            f.write(f"Total output tokens: {tok['total_output_tokens']:,}\n")
            f.write(f"Total tokens: {tok['total_tokens']:,}\n")
            f.write(f"Avg prompt tokens: {tok['avg_prompt_tokens']:.1f}\n")
            f.write(f"Avg output tokens: {tok['avg_output_tokens']:.1f}\n")

        if 'throughput_stats' in aggregated:
            f.write("\n" + "-"*80 + "\n")
            f.write("THROUGHPUT STATISTICS\n")
            f.write("-"*80 + "\n")
            tp = aggregated['throughput_stats']
            f.write(f"Mean tokens/second: {tp['mean_tokens_per_second']:.2f}\n")
            f.write(f"Median tokens/second: {tp['median_tokens_per_second']:.2f}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("PER-JOB BREAKDOWN\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Task':<6} {'Offset':<8} {'Samples':<8} {'Correct':<8} {'Accuracy':<10} {'Avg Latency':<12} {'Duration':<10}\n")
        f.write("-"*80 + "\n")

        for job in per_job_summary:
            task_id = job['array_task_id'] if job['array_task_id'] is not None else 'N/A'
            offset = job['offset'] if job['offset'] is not None else 'N/A'
            samples = job['num_samples']
            correct = job['total_correct']
            accuracy = f"{job['accuracy']:.4f}"
            latency = f"{job['avg_latency_ms']:.1f} ms" if job['avg_latency_ms'] else 'N/A'
            duration = f"{job['batch_duration_s']:.1f} s" if job['batch_duration_s'] else 'N/A'

            f.write(f"{task_id:<6} {offset!s:<8} {samples:<8} {correct:<8} {accuracy:<10} {latency:<12} {duration:<10}\n")

    print(f"Text summary saved to: {summary_text_file}")

    # Print summary to console
    print("\n" + "="*80)
    print(f"MERGE COMPLETE - {args.benchmark.upper()}")
    print("="*80)
    print(f"Total samples: {aggregated['total_samples']}")
    print(f"Total correct: {aggregated['total_correct']}")
    print(f"Overall accuracy: {aggregated['overall_accuracy']:.4f} ({aggregated['overall_accuracy']*100:.2f}%)")
    if 'latency_stats' in aggregated:
        print(f"Mean latency: {aggregated['latency_stats']['mean_ms']:.2f} ms")
        print(f"P95 latency: {aggregated['latency_stats']['p95_ms']:.2f} ms")
    print("="*80)


if __name__ == "__main__":
    main()
