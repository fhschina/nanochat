#!/bin/bash

# B200-friendly reproduction script for the ClimbMix leaderboard run (#4-style).
# Compared with runs/speedrun.sh, this keeps the ClimbMix pipeline but adds the
# B200 settings Sarah used: NCCL_NVLS_ENABLE=0 and full-context attention via
# --window-pattern=L because Blackwell currently uses the SDPA fallback here.

set -euo pipefail

# Override knobs:
#   WANDB_RUN=d24-climbmix-b200 bash runs/speedrun_b200.sh
#   PARAM_DATA_RATIO=8 bash runs/speedrun_b200.sh     # current master speedrun-style
#   PARAM_DATA_RATIO=9.5 bash runs/speedrun_b200.sh   # leaderboard #4-style default
#   NANOCHAT_BASE_DIR=/raid/$USER/nanochat/climbmix bash runs/speedrun_b200.sh
export OMP_NUM_THREADS=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_climbmix}"
mkdir -p "$NANOCHAT_BASE_DIR"

NUM_GPUS="${NUM_GPUS:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
DEPTH="${DEPTH:-24}"
PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-9.5}"
RUN_TAG="${RUN_TAG:-d${DEPTH}-climbmix-b200-r${PARAM_DATA_RATIO}}"
WANDB_RUN="${WANDB_RUN:-dummy}"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.report reset

# Tokenizer and ClimbMix shards. The current nanochat dataset.py points at
# karpathy/climbmix-400b-shuffle and writes base_data_climbmix/.
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --target-param-data-ratio="$PARAM_DATA_RATIO" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --window-pattern=L \
    --fp8 \
    --run="$WANDB_RUN" \
    --model-tag="$RUN_TAG"

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_eval -- \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --model-tag="$RUN_TAG"

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.chat_sft -- \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN" \
    --model-tag="$RUN_TAG"

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.chat_eval -- \
    -i sft \
    --model-tag="$RUN_TAG"

python -m nanochat.report generate
