#!/bin/bash

# FineWeb-EDU entrypoint for the shared SemDeDup quality A/B runner.

set -euo pipefail

export DATASET_TAG="${DATASET_TAG:-fineweb_edu}"
export NANOCHAT_DATASET_NAME="${NANOCHAT_DATASET_NAME:-fineweb_edu}"
export NANOCHAT_DATASET_URL="${NANOCHAT_DATASET_URL:-https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main}"
export NANOCHAT_DATASET_MAX_SHARD="${NANOCHAT_DATASET_MAX_SHARD:-1822}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_fineweb_edu}"
export INPUT_DATA_DIR="${INPUT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_fineweb_edu}"
export RUN_ROOT="${RUN_ROOT:-$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality}"

exec bash runs/semdedup_quality_b200.sh "$@"
