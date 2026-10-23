#!/bin/bash
# Measure RL step-judge reliability and compare candidate judges on a held-out problem set.
# Judges (run sequentially on ONE GPU, each loaded 4-bit then freed):
#   - Qwen/Qwen2.5-7B-Instruct   : the judge used in training (~6 GB at 4-bit)
#   - Qwen/Qwen2.5-32B-Instruct  : candidate stronger judge (~20 GB at 4-bit; fits a 24 GB+ card,
#                                  leave headroom for KV cache with batch-size 16 -> lower --batch-size if OOM)
# Submit with:  sbatch sh_scripts/templates/eval_judge.sbatch
#SBATCH --job-name=eval_judge
#SBATCH --output=logs/eval_judge_%j.out
#SBATCH --partition=gpuql
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=cs002  # Exclude cs002 because GPU 0 has ECC errors

set -euo pipefail

PROBLEMS="${PROBLEMS:-data/dt_eval_dataset.jsonl}"   # held-out set, NOT the RL training file
JUDGES="${JUDGES:-Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-32B-Instruct}"

source /home/018230446/miniconda3/etc/profile.d/conda.sh
conda activate v_env

mkdir -p logs

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TMPDIR="/tmp/$USER-evaljudge-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$(hostname)" == "cs002.hpc.coe" ] || [ "$(hostname)" == "cs002" ]; then
    echo "WARNING: cs002 GPU 0 has ECC errors, restricting to GPUs 1-3"
    export CUDA_VISIBLE_DEVICES="1"
fi

echo "[$(date)] Judge eval on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
python scripts/eval_judge.py \
    --problems "$PROBLEMS" \
    --max-problems 200 \
    --n-per-problem 2 \
    --judge-model "$JUDGES" \
    --load-in-4bit \
    --device cuda:0 \
    --batch-size 16
echo "[$(date)] Done. Results in results/judge_eval/"
