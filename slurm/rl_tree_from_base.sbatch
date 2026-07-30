#!/bin/bash
# Train a model with RL Tree(GRPO) starting from the BASE instruct model (no SFT).
# Submit with:  sbatch slurm/rl_tree_from_base.sbatch
#SBATCH --job-name=rl_tree_from_base
#SBATCH --output=logs/rl_tree_from_base_%j.out
#SBATCH --partition=gpuql
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=14-01:00:0
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=cs002  # Exclude cs002 because GPU 0 has ECC errors

set -euo pipefail

CONFIG="src/training/configs/rl_tree_from_base.yaml"
source /home/018230446/miniconda3/etc/profile.d/conda.sh
conda activate v_env
mkdir -p logs

# --- runtime env (vLLM + single-node) ---
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$(shuf -n 1 -i 30000-65000)
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export NCCL_SOCKET_IFNAME=^lo
export VLLM_USE_V1=0
export TMPDIR="/tmp/$USER-rl-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Reduce CUDA allocator fragmentation (the OOM report showed ~2GB reserved-but-unallocated).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] Starting RL-Tree-from-base on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
python -m src.training.run "$CONFIG"
echo "[$(date)] Done. Checkpoint in checkpoints/rl_tree_from_base"
