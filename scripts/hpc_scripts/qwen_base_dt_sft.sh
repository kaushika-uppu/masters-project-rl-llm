#!/bin/bash
#SBATCH --job-name=eval_base_sft
#SBATCH --output=logs/eval_base_sft/eval_base_%A_%a.out
#SBATCH --error=logs/eval_base_sft/eval_base_%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --nodelist=cs[001,003-004]
#SBATCH --time=04:00:00

source v_env/bin/activate
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$(shuf -n 1 -i 30000-65000)
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export NCCL_SOCKET_IFNAME=^lo
export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1

export TMPDIR=/tmp/$USER-vllm-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

export PYTHONPATH="$PWD:$PYTHONPATH"
export DEEPTHEOREM_EVAL_PATH="data/sft_dt_eval_dataset.jsonl"

LIMIT=300
OFFSET=$((SLURM_ARRAY_TASK_ID * LIMIT))

echo "============================================================"
echo "Starting Array Task $SLURM_ARRAY_TASK_ID"
echo "Processing $LIMIT questions starting at index $OFFSET"
echo "============================================================"

# Run the evaluation chunk
python scripts/evaluate.py \
    --benchmarks deeptheorem \
    --inference-fn qwen_base \
    --limit $LIMIT \
    --offset $OFFSET \
    --output-dir evaluation/results/eval_base/sft_subset \
    --verbose

echo "Evaluation complete!"