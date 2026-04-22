#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"

RUNPOD_ENV_PREFIX="${RUNPOD_ENV_PREFIX:-/root/miniconda3/envs/runpod}"
PYTHON_BIN="${PYTHON_BIN:-$RUNPOD_ENV_PREFIX/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Set RUNPOD_ENV_PREFIX or PYTHON_BIN before running this script." >&2
  exit 1
fi

export CONDA_PREFIX="${CONDA_PREFIX:-$RUNPOD_ENV_PREFIX}"
export LD_LIBRARY_PATH="$RUNPOD_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export MODEL="${MODEL:-/root/PaddlePaddle/ERNIE-4.5-0.3B-Paddle}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export ENABLE_V1_KVCACHE_SCHEDULER="${ENABLE_V1_KVCACHE_SCHEDULER:-1}"

cd "$SRC_DIR"
exec "$PYTHON_BIN" handler.py "$@"
