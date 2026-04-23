#!/usr/bin/env python3
"""
Generate SLURM batch scripts for running benchmarks.

Usage:
    python scripts/generate_slurm_script.py --inference-fn limo --benchmark gsm8k --user kaushika
"""

import argparse
import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import benchmark registry (should always work)
from evaluation.benchmark_registry import get_benchmarks, get_benchmark_size

# Try to import inference function registry (may fail on Windows due to vllm)
try:
    from src.function_registry import get_available_functions
    INFERENCE_REGISTRY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    INFERENCE_REGISTRY_AVAILABLE = False

# User configurations
USER_CONFIG = {
    "kushi": "kaushika.uppu@sjsu.edu",
    "miranda": "miranda.billawala@sjsu.edu",
    "kalyani": "kalyani.chitre@sjsu.edu"
}

# Default SLURM configuration
DEFAULT_CONFIG = {
    "partition": "gpuqm",
    "gpus": 2,
    "memory": "32G",
    "time": "04:00:00",
    "limit_per_job": 100,  # Number of items per job
}


def calculate_job_params(total_items: int, limit_per_job: int) -> tuple[int, int]:
    """
    Calculate the number of jobs needed and limit per job.
    
    Returns:
        tuple: (num_jobs, limit_per_job)
    """
    num_jobs = math.ceil(total_items / limit_per_job)
    return num_jobs, limit_per_job


def generate_script(
    inference_fn: str,
    benchmark: str,
    user: str,
    output_path: str = None,
    limit_per_job: int = None
) -> str:
    """Generate a SLURM batch script for the given parameters."""

    # Validate inputs
    if INFERENCE_REGISTRY_AVAILABLE:
        available_fns = get_available_functions()
        if inference_fn not in available_fns:
            raise ValueError(f"Unknown inference function '{inference_fn}'. Available: {available_fns}")

    available_benchmarks = get_benchmarks()
    if benchmark not in available_benchmarks:
        raise ValueError(f"Unknown benchmark '{benchmark}'. Available: {available_benchmarks}")

    if user not in USER_CONFIG:
        raise ValueError(f"Unknown user '{user}'. Available: {list(USER_CONFIG.keys())}")

    # Get benchmark size
    total_items = get_benchmark_size(benchmark)
    print(f"Benchmark '{benchmark}' has {total_items} items")

    # Calculate job parameters
    limit = limit_per_job or DEFAULT_CONFIG["limit_per_job"]
    num_jobs, actual_limit = calculate_job_params(total_items, limit)
    print(f"Will create {num_jobs} jobs with limit={actual_limit} each")

    # Get user email
    user_email = USER_CONFIG[user]
    
    # Generate script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={inference_fn}_{benchmark}
#SBATCH --output=logs/{inference_fn}_{benchmark}_%A_%a.out
#SBATCH --array=0-{num_jobs - 1}
#SBATCH --partition={DEFAULT_CONFIG['partition']}
#SBATCH --gres=gpu:{DEFAULT_CONFIG['gpus']}
#SBATCH --mem={DEFAULT_CONFIG['memory']}
#SBATCH --time={DEFAULT_CONFIG['time']}
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=END

source v_env/bin/activate
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$(shuf -n 1 -i 30000-65000)
export VLLM_HOST_IP=$(hostname -I | awk '{{print $1}}')
export NCCL_SOCKET_IFNAME=^lo
export VLLM_USE_V1=0

export TMPDIR=/tmp/$USER-vllm-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

export PYTHONPATH="$PWD:$PYTHONPATH"

LIMIT={actual_limit}
OFFSET=$(( SLURM_ARRAY_TASK_ID * LIMIT ))

echo "Job $SLURM_ARRAY_TASK_ID processing from $OFFSET to $((OFFSET + LIMIT))"
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

python scripts/evaluate.py \\
    --inference-fn {inference_fn} \\
    --benchmarks {benchmark} \\
    --limit $LIMIT \\
    --offset $OFFSET
"""
    
    # Determine output path
    if output_path is None:
        output_path = f"sh_scripts/{inference_fn}_{benchmark}.sh"
    
    # Write script to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(script_content)
    print(f"\nGenerated script: {output_path}")
    print(f"  Total items: {total_items}")
    print(f"  Jobs: {num_jobs} (array 0-{num_jobs - 1})")
    print(f"  Limit per job: {actual_limit}")
    print(f"  User: {user} ({user_email})")
    
    return script_content


def main():
    """Parse arguments and generate the script."""
    parser = argparse.ArgumentParser(
        description="Generate SLURM batch scripts for running benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_slurm_script.py --inference-fn limo --benchmark gsm8k --user kaushika
  python scripts/generate_slurm_script.py --inference-fn limo --benchmark math500 --user tanmay --limit 50
  python scripts/generate_slurm_script.py --inference-fn limo --benchmark riddlebench --user kaushika --output my_script.sh
        """
    )

    parser.add_argument(
        "--inference-fn",
        type=str,
        required=True,
        choices=get_available_functions() if INFERENCE_REGISTRY_AVAILABLE else None,
        help=f"Inference function to use. Available: {get_available_functions() if INFERENCE_REGISTRY_AVAILABLE else 'dummy, cot_3shot, limo'}"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=get_benchmarks(),
        help=f"Benchmark to run. Available: {get_benchmarks()}"
    )

    parser.add_argument(
        "--user",
        type=str,
        required=True,
        choices=list(USER_CONFIG.keys()),
        help=f"User running the job. Available: {list(USER_CONFIG.keys())}"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the generated script (default: sh_scripts/<inference_fn>_<benchmark>.sh)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Number of items per job (default: {DEFAULT_CONFIG['limit_per_job']})"
    )

    args = parser.parse_args()

    try:
        generate_script(
            inference_fn=args.inference_fn,
            benchmark=args.benchmark,
            user=args.user,
            output_path=args.output,
            limit_per_job=args.limit
        )
        print("\n✓ Script generated successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

