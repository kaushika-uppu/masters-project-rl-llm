#!/usr/bin/env python3
"""
Generate SLURM batch scripts for running benchmarks.

Usage:
    python scripts/generate_slurm_script.py --inference-fn limo --benchmark gsm8k
"""

import argparse
import math
import subprocess
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

# Default SLURM configuration
DEFAULT_CONFIG = {
    "partition": "gpuqm",
    "gpus": 2,
    "memory": "32G",
    "time": "04:00:00",
    "limit_per_job": 100,  # Number of items per job
}

TRIMMED_IDS_DIR = Path("evaluation/trimmed_ids")


def get_git_email() -> str:
    """Return the email from `git config user.email`."""
    try:
        result = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True,
            text=True,
            check=True,
        )
        email = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "Could not read email from `git config user.email`. "
            "Set it (`git config --global user.email you@example.com`) "
            "or pass --email explicitly."
        ) from e

    if not email:
        raise RuntimeError(
            "`git config user.email` is empty. "
            "Set it or pass --email explicitly."
        )
    return email


def count_trimmed_ids(benchmark: str) -> int:
    """Count usable ids in evaluation/trimmed_ids/<benchmark>.txt."""
    path = TRIMMED_IDS_DIR / f"{benchmark}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"--trimmed requested but {path} does not exist. "
            f"Generate it first (e.g. via scripts/representative_sample.py)."
        )
    with open(path) as f:
        ids = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    if not ids:
        raise ValueError(f"{path} is empty.")
    return len(ids)


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
    output_path: str = None,
    limit_per_job: int = None,
    trimmed: bool = False,
    email: str = None,
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

    user_email = email or get_git_email()

    # When trimmed, chunk over the trimmed id count, not the full benchmark.
    if trimmed:
        total_items = count_trimmed_ids(benchmark)
        print(f"Benchmark '{benchmark}' (trimmed): {total_items} items")
    else:
        total_items = get_benchmark_size(benchmark)
        print(f"Benchmark '{benchmark}' has {total_items} items")

    # Calculate job parameters
    limit = limit_per_job or DEFAULT_CONFIG["limit_per_job"]
    num_jobs, actual_limit = calculate_job_params(total_items, limit)
    print(f"Will create {num_jobs} jobs with limit={actual_limit} each")

    job_name_suffix = "_trimmed" if trimmed else ""
    trimmed_flag_line = " \\\n    --trimmed" if trimmed else ""
    # send results to model specific results
    output_dir = "evaluation/results"
    if output_path:
        output_dir += f"/{output_path}"

    # Generate script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={inference_fn}_{benchmark}{job_name_suffix}
#SBATCH --output=logs/{inference_fn}_{benchmark}{job_name_suffix}_%A_%a.out
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
    --offset $OFFSET{trimmed_flag_line} \\
    --output_dir {output_dir}
"""

    # Determine output path
    if output_path is None:
        output_path = f"sh_scripts/{inference_fn}_{benchmark}{job_name_suffix}.sh"

    # Write script to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(script_content)
    print(f"\nGenerated script: {output_path}")
    print(f"  Total items: {total_items}{' (trimmed)' if trimmed else ''}")
    print(f"  Jobs: {num_jobs} (array 0-{num_jobs - 1})")
    print(f"  Limit per job: {actual_limit}")
    print(f"  Email: {user_email}")

    return script_content


def main():
    """Parse arguments and generate the script."""
    parser = argparse.ArgumentParser(
        description="Generate SLURM batch scripts for running benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_slurm_script.py --inference-fn limo --benchmark gsm8k
  python scripts/generate_slurm_script.py --inference-fn limo --benchmark math500 --limit 50
  python scripts/generate_slurm_script.py --inference-fn limo --benchmark riddlebench --trimmed
  python scripts/generate_slurm_script.py --inference-fn limo --benchmark riddlebench --output my_script.sh
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
        "--email",
        type=str,
        default=None,
        help="Email for SLURM notifications. Defaults to `git config user.email`."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the generated script (default: sh_scripts/<inference_fn>_<benchmark>[_trimmed].sh)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Number of items per job (default: {DEFAULT_CONFIG['limit_per_job']})"
    )

    parser.add_argument(
        "--trimmed",
        action="store_true",
        help="Pass --trimmed to evaluate.py and chunk jobs over the trimmed id "
             "count from evaluation/trimmed_ids/<benchmark>.txt instead of the full dataset."
    )

    args = parser.parse_args()

    try:
        generate_script(
            inference_fn=args.inference_fn,
            benchmark=args.benchmark,
            output_path=args.output,
            limit_per_job=args.limit,
            trimmed=args.trimmed,
            email=args.email,
        )
        print("\n✓ Script generated successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

