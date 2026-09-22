#!/bin/bash
#SBATCH --job-name=rl_dataset_rollouts
#SBATCH --output=logs/rl_dataset_rollouts/rollouts_%A_%a.out
#SBATCH --error=logs/rl_dataset_rollouts/rollouts_%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --nodelist=cs[001,003-004]
#SBATCH --time=04:00:00
#SBATCH --mail-user=miranda.billawala@sigmacomputing.com
#SBATCH --mail-type=END

# Runs scripts/rl_training_dataset/generate_rollouts.py as a chunked array job,
# same pattern as scripts/hpc_scripts/limo_deeptheorem.sh /
# qwen_base_deeptheorem.sh but calling the rollout script directly (it already
# takes --chunk-id/--chunk-size, same convention as deeptheorem_phase1.py /
# deeptheorem_phase2.py) instead of going through scripts/evaluate.py.
#
# --array must cover ceil(rows in candidates.jsonl / CHUNK_SIZE) - 1. Default
# --n-candidates in prepare_candidates.py is 3000 theorems -> 6000 variant
# rows, so with CHUNK_SIZE=500 that's 12 chunks -> array=0-11 (set above).
# Recompute and edit --array if you change --n-candidates or CHUNK_SIZE:
#   python -c "import math,sys; print(math.ceil(sum(1 for _ in open(sys.argv[1]))/int(sys.argv[2]))-1)" \
#       scripts/rl_training_dataset/candidates.jsonl 500

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

MODEL="Qwen/Qwen2.5-7B-Instruct"
CHUNK_SIZE=500

echo "============================================================"
echo "Starting Array Task $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Chunk size: $CHUNK_SIZE rows (chunk id = $SLURM_ARRAY_TASK_ID)"
echo "============================================================"

python scripts/rl_training_dataset/generate_rollouts.py \
    --input scripts/rl_training_dataset/candidates.jsonl \
    --model "$MODEL" \
    --chunk-id $SLURM_ARRAY_TASK_ID \
    --chunk-size $CHUNK_SIZE \
    --output-dir scripts/rl_training_dataset/rollouts

echo "Rollout chunk $SLURM_ARRAY_TASK_ID complete!"
