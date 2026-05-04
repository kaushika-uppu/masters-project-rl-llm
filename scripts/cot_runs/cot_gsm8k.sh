#!/bin/bash
#SBATCH --job-name=cot_gsm8k
#SBATCH --output=logs/cot_gsm8k/cot_gsm8k_%A_%a.out
#SBATCH --array=0-26%2
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --nodelist=cs001,cs002,cs003,cs004
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-user=kalyani.chitre@sjsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ../.venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

RUN_ID="${RUN_ID:?Please provide RUN_ID when submitting with sbatch --export=ALL,RUN_ID=...}"
CHUNK_DIR="evaluation/results/gsm8k/gsm8k_chunks_${RUN_ID}"
FINAL_DIR="evaluation/results/gsm8k"
SUMMARY_DIR="evaluation/results/gsm8k/gsm8k_merged_summary"

mkdir -p "$CHUNK_DIR"

LIMIT=50
OFFSET=$(( SLURM_ARRAY_TASK_ID * LIMIT ))

echo "Job $SLURM_ARRAY_TASK_ID processing from $OFFSET to $((OFFSET + LIMIT))"

python scripts/evaluate.py \
    --inference-fn cot_3shot \
    --benchmarks gsm8k \
    --limit $LIMIT \
    --offset $OFFSET \
    --output-dir "$CHUNK_DIR" \
    --verbose
