#!/bin/bash
#SBATCH --job-name=vias-concat-check
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
set -euo pipefail

export HF_HOME=/data5/$USER/vias/hf
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

cd /data5/$USER/vias/code/vias_pilot
PY=/data5/$USER/vias/envs/vias-gpu/bin/python
OUT=/data5/$USER/vias/runs/concat_check/seed20261127
mkdir -p "$OUT"

$PY src/pan18_joint_fusion.py concat \
  --data data/processed/pan18_joint \
  --out "$OUT" \
  --model-dir "$OUT/ckpt" \
  --chunks 4 \
  --max-len 512 \
  --epochs 4 \
  --batch-size 8 \
  --lr-encoder 1e-5 \
  --lr-head 1e-3 \
  --seed 20261127
