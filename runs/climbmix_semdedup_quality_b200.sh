#!/bin/bash

# ClimbMix SemDeDup quality A/B runner for an 8x B200-style NanoChat setup.
#
# Modes:
#   bash runs/climbmix_semdedup_quality_b200.sh baseline
#   bash runs/climbmix_semdedup_quality_b200.sh semdedup
#   bash runs/climbmix_semdedup_quality_b200.sh randomdrop
#   bash runs/climbmix_semdedup_quality_b200.sh both
#
# This script only runs base pretraining + base evaluation. It intentionally
# does not run SFT/RL because this experiment is about semantic dedup quality.

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "baseline" && "$MODE" != "semdedup" && "$MODE" != "randomdrop" && "$MODE" != "both" ]]; then
    echo "Usage: $0 baseline|semdedup|randomdrop|both" >&2
    exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
DATASET_TAG="${DATASET_TAG:-climbmix}"
export DATASET_TAG
export NANOCHAT_DATASET_NAME="${NANOCHAT_DATASET_NAME:-$DATASET_TAG}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_${DATASET_TAG}}"
mkdir -p "$NANOCHAT_BASE_DIR"

NUM_GPUS="${NUM_GPUS:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
DEPTH="${DEPTH:-24}"
PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-9.5}"
NUM_ITERATIONS="${NUM_ITERATIONS:-}"
NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-8}"
MAX_DOCS="${MAX_DOCS:--1}"
STATS_MAX_DOCS="${STATS_MAX_DOCS:-$MAX_DOCS}"

if [[ -z "${DATASET_SHARDS:-}" ]]; then
    if [[ "$NUM_TRAIN_SHARDS" -gt 0 ]]; then
        DATASET_SHARDS=$((NUM_TRAIN_SHARDS + 1))
    else
        DATASET_SHARDS=170
    fi
fi

INPUT_DATA_DIR="${INPUT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_${DATASET_TAG}}"
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$INPUT_DATA_DIR}"
RUN_ROOT="${RUN_ROOT:-$NANOCHAT_BASE_DIR/experiments/${DATASET_TAG}_semdedup_quality}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"
DO_SEMDEDUP="${DO_SEMDEDUP:-1}"
DO_RANDOM_DROP="${DO_RANDOM_DROP:-1}"
PREP_DATASET="${PREP_DATASET:-1}"
SKIP_UV_SYNC="${SKIP_UV_SYNC:-0}"
INSTALL_CURATOR="${INSTALL_CURATOR:-0}"

SEMD_EPS="${SEMD_EPS:-0.07}"
SEMD_EPS_SLUG="${SEMD_EPS//./p}"
SEMD_N_CLUSTERS="${SEMD_N_CLUSTERS:-100}"
SEMD_MODEL="${SEMD_MODEL:-google/embeddinggemma-300m}"
SEMD_BACKEND="${SEMD_BACKEND:-curator}"
SEMD_OVERWRITE="${SEMD_OVERWRITE:-0}"
SEMD_DISTANCE_METRIC="${SEMD_DISTANCE_METRIC:-cosine}"
SEMD_WHICH_TO_KEEP="${SEMD_WHICH_TO_KEEP:-hard}"
SEMD_PAIRWISE_BATCH_SIZE="${SEMD_PAIRWISE_BATCH_SIZE:-1024}"
SEMD_EMBEDDING_MAX_CHARS="${SEMD_EMBEDDING_MAX_CHARS:-}"
SEMD_MODEL_CACHE_DIR="${SEMD_MODEL_CACHE_DIR:-}"
SEMD_VLLM_INIT_KWARGS_JSON="${SEMD_VLLM_INIT_KWARGS_JSON:-}"
SEMD_VLLM_ATTENTION_BACKEND="${SEMD_VLLM_ATTENTION_BACKEND:-}"
SEMD_VLLM_ENFORCE_EAGER="${SEMD_VLLM_ENFORCE_EAGER:-0}"
SEMD_RAY_TEMP_DIR="${SEMD_RAY_TEMP_DIR:-}"
SEMD_NO_RAY_PREINIT="${SEMD_NO_RAY_PREINIT:-0}"
SEMD_RESUME_FROM_EMBEDDINGS="${SEMD_RESUME_FROM_EMBEDDINGS:-0}"
AUDIT_SAMPLE_SIZE="${AUDIT_SAMPLE_SIZE:-100}"
AUDIT_SEED="${AUDIT_SEED:-1337}"
SKIP_TOKEN_STATS="${SKIP_TOKEN_STATS:-0}"

RANDOM_DROP_REMOVED_DOCS="${RANDOM_DROP_REMOVED_DOCS:-291374}"
RANDOM_DROP_SEED="${RANDOM_DROP_SEED:-9001}"
RANDOM_DROP_OVERWRITE="${RANDOM_DROP_OVERWRITE:-0}"

SEED="${SEED:-42}"
CORE_EVAL_SEED="${CORE_EVAL_SEED:-1337}"
EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_TOKENS="${EVAL_TOKENS:-41943040}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-2000}"
CORE_METRIC_MAX_PER_TASK="${CORE_METRIC_MAX_PER_TASK:-500}"
FINAL_CORE_MAX_PER_TASK="${FINAL_CORE_MAX_PER_TASK:--1}"
BASE_EVAL_MODES="${BASE_EVAL_MODES:-core,bpb,sample}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"
SAVE_EVERY="${SAVE_EVERY:--1}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-128}"
TOKENIZER_THREADS="${TOKENIZER_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export NUM_GPUS DEVICE_BATCH_SIZE DEPTH PARAM_DATA_RATIO NUM_ITERATIONS NUM_TRAIN_SHARDS DATASET_SHARDS MAX_DOCS
export NANOCHAT_DATA_DIR NANOCHAT_DATASET_NAME NANOCHAT_DATASET_URL NANOCHAT_DATASET_BASE_URL NANOCHAT_DATASET_MAX_SHARD
export SEMD_BACKEND SEMD_MODEL SEMD_EPS SEMD_N_CLUSTERS SEMD_DISTANCE_METRIC SEMD_WHICH_TO_KEEP
export SEMD_PAIRWISE_BATCH_SIZE SEMD_EMBEDDING_MAX_CHARS SEMD_VLLM_INIT_KWARGS_JSON
export SEMD_VLLM_ATTENTION_BACKEND SEMD_VLLM_ENFORCE_EAGER SEMD_RAY_TEMP_DIR SEMD_NO_RAY_PREINIT
export SEMD_RESUME_FROM_EMBEDDINGS
export RANDOM_DROP_REMOVED_DOCS RANDOM_DROP_SEED RANDOM_DROP_OVERWRITE
export SEED CORE_EVAL_SEED EVAL_EVERY EVAL_TOKENS CORE_METRIC_EVERY CORE_METRIC_MAX_PER_TASK FINAL_CORE_MAX_PER_TASK BASE_EVAL_MODES
export DO_TRAIN DO_EVAL DO_SEMDEDUP DO_RANDOM_DROP PYTHON_BIN

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
if [[ "$SKIP_UV_SYNC" != "1" ]]; then
    uv sync --extra gpu
fi
source .venv/bin/activate
if [[ "$INSTALL_CURATOR" == "1" ]]; then
    uv pip install --extra-index-url https://pypi.nvidia.com "nemo-curator[text_cuda12]"
fi

if [[ "$PREP_DATASET" == "1" ]]; then
    "$PYTHON_BIN" -m nanochat.dataset -n "$DATASET_SHARDS"
fi
if [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" && ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer_kind.txt" ]]; then
    "$PYTHON_BIN" -m scripts.tok_train --data-dir "$INPUT_DATA_DIR"
fi

mkdir -p "$RUN_ROOT"

write_run_config() {
    local run_kind="$1"
    local run_id="$2"
    local run_dir="$3"
    local data_dir="$4"
    local model_tag="$5"
    RUN_KIND="$run_kind" RUN_ID_VALUE="$run_id" RUN_DIR_VALUE="$run_dir" DATA_DIR_VALUE="$data_dir" MODEL_TAG_VALUE="$model_tag" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

keys = [
    "RUN_KIND", "RUN_ID_VALUE", "RUN_DIR_VALUE", "DATA_DIR_VALUE", "MODEL_TAG_VALUE",
    "DATASET_TAG", "NANOCHAT_BASE_DIR", "NANOCHAT_DATA_DIR", "NANOCHAT_DATASET_NAME",
    "NANOCHAT_DATASET_URL", "NANOCHAT_DATASET_BASE_URL", "NANOCHAT_DATASET_MAX_SHARD",
    "WANDB_MODE", "NUM_GPUS", "DEVICE_BATCH_SIZE", "DEPTH", "PARAM_DATA_RATIO",
    "NUM_ITERATIONS", "NUM_TRAIN_SHARDS", "DATASET_SHARDS", "MAX_DOCS",
    "SEMD_BACKEND", "SEMD_MODEL", "SEMD_EPS", "SEMD_N_CLUSTERS",
    "SEMD_DISTANCE_METRIC", "SEMD_WHICH_TO_KEEP", "SEMD_PAIRWISE_BATCH_SIZE",
    "SEMD_EMBEDDING_MAX_CHARS", "SEMD_VLLM_INIT_KWARGS_JSON",
    "SEMD_VLLM_ATTENTION_BACKEND", "SEMD_VLLM_ENFORCE_EAGER",
    "SEMD_RAY_TEMP_DIR", "SEMD_NO_RAY_PREINIT", "SEMD_RESUME_FROM_EMBEDDINGS",
    "RANDOM_DROP_REMOVED_DOCS",
    "RANDOM_DROP_SEED", "RANDOM_DROP_OVERWRITE", "SEED", "CORE_EVAL_SEED",
    "EVAL_EVERY", "EVAL_TOKENS", "CORE_METRIC_EVERY", "CORE_METRIC_MAX_PER_TASK",
    "FINAL_CORE_MAX_PER_TASK", "BASE_EVAL_MODES", "DO_TRAIN", "DO_EVAL",
    "DO_SEMDEDUP", "DO_RANDOM_DROP",
]
payload = {key: os.environ.get(key) for key in keys}
payload["run_kind"] = payload.pop("RUN_KIND")
payload["run_id"] = payload.pop("RUN_ID_VALUE")
payload["run_dir"] = payload.pop("RUN_DIR_VALUE")
payload["data_dir"] = payload.pop("DATA_DIR_VALUE")
payload["model_tag"] = payload.pop("MODEL_TAG_VALUE")
path = Path(payload["run_dir"]) / "run_config.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True))
print(f"Wrote run config: {path}")
PY
}

write_run_summary() {
    local run_kind="$1"
    local run_id="$2"
    local run_dir="$3"
    local data_dir="$4"
    local model_tag="$5"
    RUN_KIND="$run_kind" RUN_ID_VALUE="$run_id" RUN_DIR_VALUE="$run_dir" DATA_DIR_VALUE="$data_dir" MODEL_TAG_VALUE="$model_tag" \
    "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
import re
from pathlib import Path


def read(path):
    return path.read_text(errors="replace") if path.exists() else ""


def parse_int(value):
    return int(value.replace(",", ""))


def last_float(pattern, text):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def last_int(pattern, text):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return parse_int(matches[-1]) if matches else None


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def load_core_csv(path):
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            task = row[0].strip()
            acc = row[1].strip() if len(row) > 1 else ""
            centered = row[2].strip() if len(row) > 2 else ""
            out[task] = {
                "accuracy": float(acc) if acc else None,
                "centered": float(centered) if centered else None,
            }
    return out


run_dir = Path(os.environ["RUN_DIR_VALUE"])
train_log = read(run_dir / "train.log")
eval_log = read(run_dir / "base_eval.log")
val_curve = [
    {"step": int(step), "bpb": float(bpb)}
    for step, bpb in re.findall(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([0-9.]+)", train_log)
]
core_curve = [
    {"step": int(step), "core": float(core)}
    for step, core in re.findall(r"Step\s+(\d+)\s+\|\s+CORE metric:\s+([0-9.]+)", train_log)
]
num_iterations = (
    last_int(r"Calculated number of iterations.*:\s+([0-9,]+)", train_log)
    or last_int(r"Using user-provided number of iterations:\s+([0-9,]+)", train_log)
)
total_training_tokens = last_int(r"Total number of training tokens:\s+([0-9,]+)", train_log)
total_training_time_min = last_float(r"Total training time:\s+([0-9.]+)m", train_log)
final_core = last_float(r"CORE metric:\s+([0-9.]+)", eval_log) or (core_curve[-1]["core"] if core_curve else None)

summary = {
    "run_kind": os.environ["RUN_KIND"],
    "run_id": os.environ["RUN_ID_VALUE"],
    "run_dir": str(run_dir),
    "data_dir": os.environ["DATA_DIR_VALUE"],
    "model_tag": os.environ["MODEL_TAG_VALUE"],
    "paths": {
        "run_config": str(run_dir / "run_config.json"),
        "data_stats": str(run_dir / "data_stats.json"),
        "train_log": str(run_dir / "train.log"),
        "base_eval_log": str(run_dir / "base_eval.log"),
        "base_eval_core_csv": str(run_dir / "base_eval_core.csv"),
        "report": str(run_dir / "report.md"),
        "semdedup_manifest": str(run_dir / "semdedup_manifest.json"),
        "random_drop_manifest": str(run_dir / "random_drop_manifest.json"),
    },
    "data_stats": load_json(run_dir / "data_stats.json"),
    "semdedup_manifest": load_json(run_dir / "semdedup_manifest.json"),
    "random_drop_manifest": load_json(run_dir / "random_drop_manifest.json"),
    "metrics": {
        "num_iterations": num_iterations,
        "total_training_tokens": total_training_tokens,
        "total_training_time_sec": total_training_time_min * 60 if total_training_time_min is not None else None,
        "min_val_bpb": last_float(r"Minimum validation bpb:\s+([0-9.]+)", train_log),
        "final_train_val_bpb": val_curve[-1]["bpb"] if val_curve else None,
        "base_eval_train_bpb": last_float(r"^train bpb:\s+([0-9.]+)", eval_log),
        "base_eval_val_bpb": last_float(r"^val bpb:\s+([0-9.]+)", eval_log),
        "final_core": final_core,
    },
    "curves": {
        "val_bpb": val_curve,
        "core": core_curve,
    },
    "per_task_core": load_core_csv(run_dir / "base_eval_core.csv"),
}
(run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Wrote run summary: {run_dir / 'run_summary.json'}")
PY
}

run_one() {
    local run_kind="$1"
    local default_run_id
    if [[ "$run_kind" == "baseline" ]]; then
        default_run_id="baseline_${RUN_TIMESTAMP}_d${DEPTH}_n${NUM_TRAIN_SHARDS}_seed${SEED}"
    elif [[ "$run_kind" == "semdedup" ]]; then
        default_run_id="semdedup_eps${SEMD_EPS_SLUG}_${RUN_TIMESTAMP}_d${DEPTH}_n${NUM_TRAIN_SHARDS}_seed${SEED}"
    else
        default_run_id="randomdrop_drop${RANDOM_DROP_REMOVED_DOCS}_rdseed${RANDOM_DROP_SEED}_${RUN_TIMESTAMP}_d${DEPTH}_n${NUM_TRAIN_SHARDS}_seed${SEED}"
    fi
    local run_id
    if [[ -n "${RUN_ID:-}" ]]; then
        if [[ "$MODE" == "both" ]]; then
            run_id="${RUN_ID}_${run_kind}"
        else
            run_id="$RUN_ID"
        fi
    else
        run_id="$default_run_id"
    fi

    local run_dir="$RUN_ROOT/$run_id"
    mkdir -p "$run_dir"

    local data_dir="$INPUT_DATA_DIR"
    if [[ "$run_kind" == "semdedup" ]]; then
        data_dir="${SEMD_OUTPUT_DIR:-$run_dir/base_data_${DATASET_TAG}_semdedup_eps${SEMD_EPS_SLUG}_n${NUM_TRAIN_SHARDS}}"
    elif [[ "$run_kind" == "randomdrop" ]]; then
        data_dir="${RANDOM_DROP_OUTPUT_DIR:-$run_dir/base_data_${DATASET_TAG}_randomdrop_drop${RANDOM_DROP_REMOVED_DOCS}_seed${RANDOM_DROP_SEED}_n${NUM_TRAIN_SHARDS}}"
    fi

    local model_tag
    if [[ "$run_kind" == "baseline" ]]; then
        model_tag="${RUN_TAG_BASELINE:-d${DEPTH}-${DATASET_TAG}-nosd-n${NUM_TRAIN_SHARDS}-seed${SEED}-${RUN_TIMESTAMP}}"
    elif [[ "$run_kind" == "semdedup" ]]; then
        model_tag="${RUN_TAG_SEMDEDUP:-d${DEPTH}-${DATASET_TAG}-semdedup-eps${SEMD_EPS_SLUG}-n${NUM_TRAIN_SHARDS}-seed${SEED}-${RUN_TIMESTAMP}}"
    else
        model_tag="${RUN_TAG_RANDOMDROP:-d${DEPTH}-${DATASET_TAG}-randomdrop-drop${RANDOM_DROP_REMOVED_DOCS}-rdseed${RANDOM_DROP_SEED}-n${NUM_TRAIN_SHARDS}-seed${SEED}-${RUN_TIMESTAMP}}"
    fi

    write_run_config "$run_kind" "$run_id" "$run_dir" "$data_dir" "$model_tag"

    if [[ "$run_kind" == "semdedup" ]]; then
        if [[ "$DO_SEMDEDUP" == "1" ]]; then
            SEMD_ARGS=(
                --input-data-dir "$INPUT_DATA_DIR"
                --output-data-dir "$data_dir"
                --cache-dir "$run_dir/semdedup_cache"
                --analysis-output-dir "$run_dir"
                --num-train-shards "$NUM_TRAIN_SHARDS"
                --max-docs "$MAX_DOCS"
                --backend "$SEMD_BACKEND"
                --model-identifier "$SEMD_MODEL"
                --eps "$SEMD_EPS"
                --n-clusters "$SEMD_N_CLUSTERS"
                --distance-metric "$SEMD_DISTANCE_METRIC"
                --which-to-keep "$SEMD_WHICH_TO_KEEP"
                --pairwise-batch-size "$SEMD_PAIRWISE_BATCH_SIZE"
                --audit-sample-size "$AUDIT_SAMPLE_SIZE"
                --audit-seed "$AUDIT_SEED"
                --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE"
                --tokenizer-threads "$TOKENIZER_THREADS"
            )
            if [[ -n "$SEMD_EMBEDDING_MAX_CHARS" ]]; then
                SEMD_ARGS+=(--embedding-max-chars "$SEMD_EMBEDDING_MAX_CHARS")
            fi
            if [[ -n "$SEMD_MODEL_CACHE_DIR" ]]; then
                SEMD_ARGS+=(--model-cache-dir "$SEMD_MODEL_CACHE_DIR")
            fi
            if [[ -n "$SEMD_VLLM_INIT_KWARGS_JSON" ]]; then
                SEMD_ARGS+=(--embedding-vllm-init-kwargs-json "$SEMD_VLLM_INIT_KWARGS_JSON")
            fi
            if [[ -n "$SEMD_VLLM_ATTENTION_BACKEND" ]]; then
                SEMD_ARGS+=(--embedding-attention-backend "$SEMD_VLLM_ATTENTION_BACKEND")
            fi
            if [[ "$SEMD_VLLM_ENFORCE_EAGER" == "1" ]]; then
                SEMD_ARGS+=(--embedding-enforce-eager)
            fi
            if [[ -n "$SEMD_RAY_TEMP_DIR" ]]; then
                SEMD_ARGS+=(--ray-temp-dir "$SEMD_RAY_TEMP_DIR")
            fi
            if [[ "$SEMD_NO_RAY_PREINIT" == "1" ]]; then
                SEMD_ARGS+=(--no-ray-preinit)
            fi
            if [[ "$SEMD_RESUME_FROM_EMBEDDINGS" == "1" ]]; then
                SEMD_ARGS+=(--resume-from-embeddings)
            fi
            if [[ "$SEMD_OVERWRITE" == "1" ]]; then
                SEMD_ARGS+=(--overwrite)
            fi
            if [[ "$SKIP_TOKEN_STATS" == "1" ]]; then
                SEMD_ARGS+=(--skip-token-stats)
            fi
            "$PYTHON_BIN" -m scripts.build_semdedup_climbmix "${SEMD_ARGS[@]}" 2>&1 | tee "$run_dir/semdedup.log"
        else
            if [[ ! -d "$data_dir" ]]; then
                echo "DO_SEMDEDUP=0 but SemDeDup data dir does not exist: $data_dir" >&2
                exit 1
            fi
            if [[ -f "$data_dir/semdedup_manifest.json" ]]; then
                cp "$data_dir/semdedup_manifest.json" "$run_dir/semdedup_manifest.json"
            fi
            echo "Skipped SemDeDup build; using existing data dir: $data_dir" | tee "$run_dir/semdedup.skipped"
        fi
    fi

    if [[ "$run_kind" == "randomdrop" ]]; then
        if [[ "$DO_RANDOM_DROP" == "1" ]]; then
            RANDOM_DROP_ARGS=(
                --input-data-dir "$INPUT_DATA_DIR"
                --output-data-dir "$data_dir"
                --analysis-output-dir "$run_dir"
                --num-train-shards "$NUM_TRAIN_SHARDS"
                --max-docs "$MAX_DOCS"
                --target-removed-docs "$RANDOM_DROP_REMOVED_DOCS"
                --random-seed "$RANDOM_DROP_SEED"
                --audit-sample-size "$AUDIT_SAMPLE_SIZE"
                --audit-seed "$AUDIT_SEED"
                --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE"
                --tokenizer-threads "$TOKENIZER_THREADS"
            )
            if [[ "$RANDOM_DROP_OVERWRITE" == "1" || "$SEMD_OVERWRITE" == "1" ]]; then
                RANDOM_DROP_ARGS+=(--overwrite)
            fi
            if [[ "$SKIP_TOKEN_STATS" == "1" ]]; then
                RANDOM_DROP_ARGS+=(--skip-token-stats)
            fi
            "$PYTHON_BIN" -m scripts.build_random_drop_climbmix "${RANDOM_DROP_ARGS[@]}" 2>&1 | tee "$run_dir/randomdrop.log"
        else
            if [[ ! -d "$data_dir" ]]; then
                echo "DO_RANDOM_DROP=0 but random-drop data dir does not exist: $data_dir" >&2
                exit 1
            fi
            if [[ -f "$data_dir/random_drop_manifest.json" ]]; then
                cp "$data_dir/random_drop_manifest.json" "$run_dir/random_drop_manifest.json"
            fi
            echo "Skipped random-drop build; using existing data dir: $data_dir" | tee "$run_dir/randomdrop.skipped"
        fi
    fi

    STATS_ARGS=(
        --data-dir "$data_dir"
        --split train
        --output "$run_dir/data_stats.json"
        --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE"
        --tokenizer-threads "$TOKENIZER_THREADS"
    )
    if [[ "$run_kind" == "baseline" ]]; then
        STATS_ARGS+=(--num-train-shards "$NUM_TRAIN_SHARDS")
    else
        STATS_ARGS+=(--num-train-shards -1)
    fi
    if [[ "$STATS_MAX_DOCS" -ge 0 ]]; then
        STATS_ARGS+=(--max-docs "$STATS_MAX_DOCS")
    fi
    "$PYTHON_BIN" -m scripts.climbmix_data_stats "${STATS_ARGS[@]}"

    "$PYTHON_BIN" -m nanochat.report reset

    local wandb_run="${WANDB_RUN:-dummy}"
    if [[ "$run_kind" == "baseline" && -n "${WANDB_RUN_BASELINE:-}" ]]; then
        wandb_run="$WANDB_RUN_BASELINE"
    elif [[ "$run_kind" == "semdedup" && -n "${WANDB_RUN_SEMDEDUP:-}" ]]; then
        wandb_run="$WANDB_RUN_SEMDEDUP"
    elif [[ "$run_kind" == "randomdrop" && -n "${WANDB_RUN_RANDOMDROP:-}" ]]; then
        wandb_run="$WANDB_RUN_RANDOMDROP"
    fi

    TRAIN_HORIZON_ARGS=(--target-param-data-ratio="$PARAM_DATA_RATIO")
    if [[ -n "$NUM_ITERATIONS" ]]; then
        TRAIN_HORIZON_ARGS=(--num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1)
    fi

    if [[ "$DO_TRAIN" == "1" ]]; then
        torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_train -- \
            --depth="$DEPTH" \
            "${TRAIN_HORIZON_ARGS[@]}" \
            --device-batch-size="$DEVICE_BATCH_SIZE" \
            --window-pattern=L \
            --fp8 \
            --eval-every="$EVAL_EVERY" \
            --eval-tokens="$EVAL_TOKENS" \
            --core-metric-every="$CORE_METRIC_EVERY" \
            --core-metric-max-per-task="$CORE_METRIC_MAX_PER_TASK" \
            --sample-every="$SAMPLE_EVERY" \
            --save-every="$SAVE_EVERY" \
            --run="$wandb_run" \
            --seed="$SEED" \
            --core-eval-seed="$CORE_EVAL_SEED" \
            --model-tag="$model_tag" \
            --data-source=parquet \
            --data-dir="$data_dir" 2>&1 | tee "$run_dir/train.log"
    else
        echo "Skipped training because DO_TRAIN=0" | tee "$run_dir/train.log"
    fi

    if [[ "$DO_EVAL" == "1" ]]; then
        torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_eval -- \
            --eval="$BASE_EVAL_MODES" \
            --max-per-task="$FINAL_CORE_MAX_PER_TASK" \
            --device-batch-size="$DEVICE_BATCH_SIZE" \
            --seed="$SEED" \
            --core-eval-seed="$CORE_EVAL_SEED" \
            --model-tag="$model_tag" \
            --data-source=parquet \
            --data-dir="$data_dir" 2>&1 | tee "$run_dir/base_eval.log"
        if compgen -G "$NANOCHAT_BASE_DIR/base_eval/base_model_*.csv" > /dev/null; then
            latest_core_csv="$(ls -t "$NANOCHAT_BASE_DIR"/base_eval/base_model_*.csv | head -n 1)"
            cp "$latest_core_csv" "$run_dir/base_eval_core.csv"
        fi
    else
        echo "Skipped base eval because DO_EVAL=0" | tee "$run_dir/base_eval.log"
    fi

    "$PYTHON_BIN" -m nanochat.report generate 2>&1 | tee "$run_dir/report_generate.log"
    if [[ -f report.md ]]; then
        cp report.md "$run_dir/report.md"
    elif [[ -f "$NANOCHAT_BASE_DIR/report/report.md" ]]; then
        cp "$NANOCHAT_BASE_DIR/report/report.md" "$run_dir/report.md"
    fi

    write_run_summary "$run_kind" "$run_id" "$run_dir" "$data_dir" "$model_tag"
    echo "Finished $run_kind run: $run_dir"
}

if [[ "$MODE" == "baseline" || "$MODE" == "both" ]]; then
    run_one baseline
fi
if [[ "$MODE" == "semdedup" || "$MODE" == "both" ]]; then
    run_one semdedup
fi
if [[ "$MODE" == "randomdrop" ]]; then
    run_one randomdrop
fi
