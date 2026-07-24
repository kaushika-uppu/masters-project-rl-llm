#!/bin/bash
# Train a model with Tree-GRPO starting from the pushed SFT checkpoint.
# Set the checkpoint in src/training/configs/rl_tree_from_sft.yaml first.
# Submit with:  sbatch slurm/rl_tree_from_sft.sbatch
#SBATCH --job-name=rl_tree_from_sft
#SBATCH --output=logs/rl_tree_from_sft_%j.out
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=cs002  # Exclude cs002 because GPU 0 has ECC errors

set -euo pipefail

CONFIG="src/training/configs/rl_tree_from_sft.yaml"

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

# GPU 0 on cs002 has ECC errors (199 uncorrectable), so exclude it
# Force use of GPUs 1,2,3 if we're on cs002
if [ "$(hostname)" == "cs002.hpc.coe" ] || [ "$(hostname)" == "cs002" ]; then
    echo "WARNING: cs002 GPU 0 has ECC errors, restricting to GPUs 1-3"
    export CUDA_VISIBLE_DEVICES="1,2,3"
fi

# Verify GPU access and show memory
echo "=== GPU Status ==="
nvidia-smi || echo "WARNING: nvidia-smi failed"

# Check for zombie processes
echo "=== Checking for zombie processes ==="
for gpu in $(echo ${CUDA_VISIBLE_DEVICES} | tr ',' ' '); do
    echo "Checking GPU $gpu for processes..."
    nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader | while read pid; do
        if [ ! -z "$pid" ]; then
            echo "WARNING: Found process $pid on GPU $gpu"
            ps -p $pid -o user,cmd || true
        fi
    done
done

echo "[$(date)] Starting Tree-GRPO from SFT on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
python -m src.training.run "$CONFIG"
echo "[$(date)] Done. Checkpoint in checkpoints/rl_tree_from_sft"
