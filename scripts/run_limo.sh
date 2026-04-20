#!/bin/bash
#SBATCH --job-name=limo_gsm8k
#SBATCH --output=results/limo_gsm8k_%A_%a.out
#SBATCH --array=0-9
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source v_env/bin/activate
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$(shuf -n 1 -i 30000-65000)
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export NCCL_SOCKET_IFNAME=^lo
export VLLM_USE_V1=0

export TMPDIR=/tmp/$USER-vllm-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

export PYTHONPATH="$PWD:$PYTHONPATH"

LIMIT=50
OFFSET=$(( SLURM_ARRAY_TASK_ID * LIMIT ))

echo "Job $SLURM_ARRAY_TASK_ID processing from $OFFSET to $((OFFSET + LIMIT))"
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

python scripts/evaluate.py \
    --inference-fn limo \
    --benchmarks gsm8k \
    --limit $LIMIT \
    --offset $OFFSET \
    --job-id $SLURM_ARRAY_JOB_ID \
    --array-task-id $SLURM_ARRAY_TASK_ID