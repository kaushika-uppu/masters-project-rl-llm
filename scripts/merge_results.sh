#!/bin/bash
#SBATCH --job-name=merge_results
#SBATCH --output=results/merge_%j.out
#SBATCH --partition=gpuqs
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# This script merges all array job results into a single summary
# Usage: sbatch --dependency=afterok:<JOB_ID> scripts/merge_results.sh <JOB_ID>

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"
conda activate v_env

export PYTHONPATH="$PWD:$PYTHONPATH"

# Get the array job ID from the first argument
ARRAY_JOB_ID=$1

if [ -z "$ARRAY_JOB_ID" ]; then
    echo "ERROR: No job ID provided"
    echo "Usage: sbatch --dependency=afterok:<JOB_ID> scripts/merge_results.sh <JOB_ID>"
    exit 1
fi

echo "Merging results from array job $ARRAY_JOB_ID"
echo "Timestamp: $(date)"

# Run Python script to merge results
python scripts/merge_results.py \
    --job-id $ARRAY_JOB_ID \
    --results-dir evaluation/results \
    --benchmark gsm8k

echo "Merge complete!"
echo "Results available in: evaluation/results/job_${ARRAY_JOB_ID}/"
