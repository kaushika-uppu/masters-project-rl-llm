#!/bin/bash
# Train a model with RL (GRPO) starting from the BASE instruct model (no SFT).
# Submit with:  sbatch slurm/rl_from_base.sbatch
#SBATCH --job-name=rl_from_base
#SBATCH --output=logs/rl_from_base_%j.out
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

CONFIG="src/training/configs/rl_from_base.yaml"
VENV="v_env"                 # cluster virtualenv (matches the eval scripts); change if yours differs

mkdir -p logs
source "${VENV}/bin/activate"

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

echo "[$(date)] Starting RL-from-base on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
python -m src.training.run "$CONFIG"
echo "[$(date)] Done. Checkpoint in checkpoints/rl_from_base"
