#!/bin/bash

# B200 SemDeDup-reduced ClimbMix path.
# This script keeps the tokenizer/eval/model path aligned with runs/speedrun_b200.sh,
# but trains from an explicit SemDeDup-reduced parquet directory.
#
# Typical pilot:
#   NUM_TRAIN_SHARDS=8 SEMD_EPS=0.07 NUM_ITERATIONS=400 \
#     bash runs/semdedup_reduced_b200.sh
#
# For a dependency-free plumbing smoke test only, use SEMD_BACKEND=exact-smoke.

set -euo pipefail

export OMP_NUM_THREADS=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_climbmix}"
mkdir -p "$NANOCHAT_BASE_DIR"

NUM_GPUS="${NUM_GPUS:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
DEPTH="${DEPTH:-24}"
PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-9.5}"
NUM_ITERATIONS="${NUM_ITERATIONS:-}"
WANDB_RUN="${WANDB_RUN:-dummy}"

INPUT_DATA_DIR="${INPUT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_climbmix}"
NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-8}"
SEMD_EPS="${SEMD_EPS:-0.07}"
SEMD_N_CLUSTERS="${SEMD_N_CLUSTERS:-100}"
SEMD_MODEL="${SEMD_MODEL:-google/embeddinggemma-300m}"
SEMD_BACKEND="${SEMD_BACKEND:-curator}"
SEMD_OVERWRITE="${SEMD_OVERWRITE:-0}"
SEMD_EPS_SLUG="${SEMD_EPS//./p}"
SEMD_OUTPUT_DIR="${SEMD_OUTPUT_DIR:-$NANOCHAT_BASE_DIR/base_data_climbmix_semdedup_eps${SEMD_EPS_SLUG}_n${NUM_TRAIN_SHARDS}}"
SEMD_CACHE_DIR="${SEMD_CACHE_DIR:-$NANOCHAT_BASE_DIR/semdedup_cache/eps${SEMD_EPS_SLUG}_n${NUM_TRAIN_SHARDS}}"
RUN_TAG="${RUN_TAG:-d${DEPTH}-climbmix-semdedup-eps${SEMD_EPS_SLUG}-n${NUM_TRAIN_SHARDS}}"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
if [ "${SKIP_UV_SYNC:-0}" != "1" ]; then
    uv sync --extra gpu
fi
source .venv/bin/activate
if [ "${INSTALL_CURATOR:-0}" = "1" ]; then
    uv pip install --extra-index-url https://pypi.nvidia.com "nemo-curator[text_cuda12]"
fi

python -m nanochat.report reset

# Ensure the baseline ClimbMix shards/tokenizer exist. The SemDeDup output reuses
# the original final validation shard and the same tokenizer for A/B comparison.
python -m nanochat.dataset -n "$NUM_TRAIN_SHARDS"
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] && [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer_kind.txt" ]; then
    python -m scripts.tok_train --data-dir "$INPUT_DATA_DIR"
fi

SEMD_ARGS=(
    --input-data-dir "$INPUT_DATA_DIR"
    --output-data-dir "$SEMD_OUTPUT_DIR"
    --cache-dir "$SEMD_CACHE_DIR"
    --num-train-shards "$NUM_TRAIN_SHARDS"
    --backend "$SEMD_BACKEND"
    --model-identifier "$SEMD_MODEL"
    --eps "$SEMD_EPS"
    --n-clusters "$SEMD_N_CLUSTERS"
)
if [ "$SEMD_OVERWRITE" = "1" ]; then
    SEMD_ARGS+=(--overwrite)
fi
python -m scripts.build_semdedup_dataset "${SEMD_ARGS[@]}"

TRAIN_HORIZON_ARGS=(--target-param-data-ratio="$PARAM_DATA_RATIO")
if [ -n "$NUM_ITERATIONS" ]; then
    TRAIN_HORIZON_ARGS=(--num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1)
fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_train -- \
    --depth="$DEPTH" \
    "${TRAIN_HORIZON_ARGS[@]}" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --window-pattern=L \
    --fp8 \
    --run="$WANDB_RUN" \
    --model-tag="$RUN_TAG" \
    --data-source=parquet \
    --data-dir="$SEMD_OUTPUT_DIR"

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_eval -- \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --model-tag="$RUN_TAG" \
    --data-source=parquet \
    --data-dir="$SEMD_OUTPUT_DIR"

python -m nanochat.report generate
