#!/bin/bash
# Full pipeline in one job: SFT first, then RL (GRPO) on the resulting checkpoint.
# Submit with:  sbatch slurm/sft_then_rl.sbatch
#SBATCH --job-name=sft_then_rl
#SBATCH --output=logs/sft_then_rl_%j.out
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

CONFIG="src/training/configs/sft_then_rl.yaml"
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
export TMPDIR="/tmp/$USER-sftrl-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "[$(date)] Starting SFT->RL on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
python -m src.training.run "$CONFIG"
echo "[$(date)] Done. SFT -> checkpoints/pipeline_sft, RL -> checkpoints/pipeline_rl"
