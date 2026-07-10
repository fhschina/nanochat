#!/bin/bash

# Robust FineWeb-EDU Qwen3 SemDeDup orchestrator.
# This keeps the formal run IDs from fineweb_edu_qwen3_semdedup_nohup_b200.sh,
# but runs the seed42 dedup build directly before handing training/repeats/report
# back to the shared runners.

set -euo pipefail

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_fineweb_edu}"
export DATASET_TAG="${DATASET_TAG:-fineweb_edu}"
export INPUT_DATA_DIR="${INPUT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_fineweb_edu}"
export RUN_ROOT="${RUN_ROOT:-$NANOCHAT_BASE_DIR/experiments/fineweb_edu_qwen3_semdedup_quality}"
export OLD_RUN_ROOT="${OLD_RUN_ROOT:-$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality}"
export REPORT_DIR="${REPORT_DIR:-reports/semdedup_quality/fineweb_edu}"

QWEN_BATCH_ID="${QWEN_BATCH_ID:-qwen3-n170-r9p5-eps0p07-$(date -u +%Y%m%dT%H%M%SZ)}"
SEED42_RUN_ID="${QWEN_SEED42_RUN_ID:-${QWEN_BATCH_ID}-seed42_semdedup}"
SEED42_RUN_DIR="${QWEN_SEED42_RUN:-$RUN_ROOT/$SEED42_RUN_ID}"
SEED42_DATA_DIR="${QWEN_SEMD_OUTPUT_DIR:-$SEED42_RUN_DIR/base_data_fineweb_edu_semdedup_eps0p07_n170}"
SEED42_CACHE_DIR="$SEED42_RUN_DIR/semdedup_cache"
QWEN_FORCE_REBUILD="${QWEN_FORCE_REBUILD:-0}"
QWEN_FORCE_TRAIN_SEED42="${QWEN_FORCE_TRAIN_SEED42:-0}"
QWEN_RESUME_FROM_EMBEDDINGS="${QWEN_RESUME_FROM_EMBEDDINGS:-0}"
QWEN_REUSE_STAGED_INPUTS="${QWEN_REUSE_STAGED_INPUTS:-0}"
QWEN_INPUT_STATS_CACHE="${QWEN_INPUT_STATS_CACHE:-$SEED42_CACHE_DIR/input_stats.json}"
QWEN_STAGED_DOCS_PER_FILE="${QWEN_STAGED_DOCS_PER_FILE:-4096}"
QWEN_INPUT_FILES_PER_PARTITION="${QWEN_INPUT_FILES_PER_PARTITION:-1}"
QWEN_EMBEDDING_MAX_CHARS="${QWEN_EMBEDDING_MAX_CHARS:-65536}"
QWEN_KMEANS_FILES_PER_GROUP="${QWEN_KMEANS_FILES_PER_GROUP:-128}"
QWEN_VLLM_INIT_KWARGS_JSON="${QWEN_VLLM_INIT_KWARGS_JSON:-}"
if [[ -z "$QWEN_VLLM_INIT_KWARGS_JSON" ]]; then
    QWEN_VLLM_INIT_KWARGS_JSON='{"max_model_len":32768,"max_num_seqs":64,"max_num_batched_tokens":32768,"gpu_memory_utilization":0.82}'
fi

mkdir -p "$RUN_ROOT" "$SEED42_RUN_DIR" "$REPORT_DIR"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

build_seed42_data() {
    if [[ "$QWEN_FORCE_REBUILD" != "1" && -f "$SEED42_RUN_DIR/semdedup_manifest.json" && -d "$SEED42_CACHE_DIR/curator_cache/semantic_dedup/pairwise_results" ]]; then
        log "Seed42 Qwen3 dedup artifacts already exist; skipping rebuild"
        return
    fi

    log "Building seed42 Qwen3 SemDeDup data directly"
    args=(
        --input-data-dir "$INPUT_DATA_DIR"
        --output-data-dir "$SEED42_DATA_DIR"
        --cache-dir "$SEED42_CACHE_DIR"
        --analysis-output-dir "$SEED42_RUN_DIR"
        --num-train-shards 170
        --max-docs -1
        --staged-docs-per-file "$QWEN_STAGED_DOCS_PER_FILE"
        --backend curator
        --model-identifier qwen3-embedding-8b
        --embedding-max-chars "$QWEN_EMBEDDING_MAX_CHARS"
        --embedding-vllm-init-kwargs-json "$QWEN_VLLM_INIT_KWARGS_JSON"
        --input-files-per-partition "$QWEN_INPUT_FILES_PER_PARTITION"
        --kmeans-files-per-group "$QWEN_KMEANS_FILES_PER_GROUP"
        --eps 0.07
        --n-clusters 100
        --distance-metric cosine
        --which-to-keep hard
        --pairwise-batch-size 1024
        --audit-sample-size 100
        --audit-seed 1337
        --tokenizer-batch-size 128
        --tokenizer-threads 4
        --ray-temp-dir /tmp/fwe_qwen3_ray_seed42_direct
    )
    if [[ "$QWEN_RESUME_FROM_EMBEDDINGS" == "1" ]]; then
        args+=(--resume-from-embeddings)
    fi
    if [[ "$QWEN_REUSE_STAGED_INPUTS" == "1" ]]; then
        args+=(--reuse-staged-inputs --input-stats-cache "$QWEN_INPUT_STATS_CACHE")
    fi
    if [[ "$QWEN_FORCE_REBUILD" == "1" ]]; then
        args+=(--overwrite)
    fi
    PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.build_semdedup_dataset "${args[@]}" 2>&1 | tee "$SEED42_RUN_DIR/semdedup.log"
}

train_seed42() {
    if [[ "$QWEN_FORCE_TRAIN_SEED42" != "1" && -f "$SEED42_RUN_DIR/run_summary.json" ]]; then
        log "Seed42 training summary already exists; skipping seed42 training"
        return
    fi

    log "Training/evaluating seed42 Qwen3 SemDeDup run"
    SEED=42 \
    RUN_ID="$SEED42_RUN_ID" \
    RUN_TIMESTAMP="${QWEN_BATCH_ID}_seed42" \
    WANDB_RUN_SEMDEDUP="fineweb-edu-qwen3-semdedup-eps0p07-n170-seed42" \
    RUN_TAG_SEMDEDUP="d24-fineweb_edu-qwen3-semdedup-eps0p07-n170-seed42-${QWEN_BATCH_ID}" \
    NUM_TRAIN_SHARDS=170 \
    PARAM_DATA_RATIO=9.5 \
    SEMD_EPS=0.07 \
    SEMD_N_CLUSTERS=100 \
    SEMD_MODEL_IDENTIFIER=qwen3-embedding-8b \
    SEMD_OUTPUT_DIR="$SEED42_DATA_DIR" \
    DO_SEMDEDUP=0 \
    PREP_DATASET=0 \
    SKIP_UV_SYNC=1 \
    INSTALL_CURATOR=0 \
    SEMD_OVERWRITE=0 \
    CORE_METRIC_EVERY=2000 \
    CORE_METRIC_MAX_PER_TASK=500 \
    FINAL_CORE_MAX_PER_TASK=-1 \
    bash runs/fineweb_edu_semdedup_quality_b200.sh semdedup
}

run_repeats_and_report() {
    log "Running Qwen3 repeat seeds 43/44"
    QWEN_BATCH_ID="$QWEN_BATCH_ID" \
    QWEN_SEED42_RUN="$SEED42_RUN_DIR" \
    QWEN_SEMD_OUTPUT_DIR="$SEED42_DATA_DIR" \
    QWEN_DATA_STATS_SOURCE="$SEED42_RUN_DIR/data_stats.json" \
    SKIP_UV_SYNC=1 \
    QWEN_INSTALL_CURATOR=0 \
    QWEN_PREP_DATASET=0 \
    bash runs/fineweb_edu_qwen3_semdedup_nohup_b200.sh repeats

    log "Generating Qwen3 ECDF and final report"
    QWEN_BATCH_ID="$QWEN_BATCH_ID" \
    QWEN_SEED42_RUN="$SEED42_RUN_DIR" \
    QWEN_SEMD_OUTPUT_DIR="$SEED42_DATA_DIR" \
    QWEN_DATA_STATS_SOURCE="$SEED42_RUN_DIR/data_stats.json" \
    SKIP_UV_SYNC=1 \
    QWEN_INSTALL_CURATOR=0 \
    QWEN_PREP_DATASET=0 \
    bash runs/fineweb_edu_qwen3_semdedup_nohup_b200.sh report
}

log "Qwen3 direct orchestrator batch: $QWEN_BATCH_ID"
log "Seed42 run dir: $SEED42_RUN_DIR"
log "Qwen3 conservative embedding config: staged_docs_per_file=$QWEN_STAGED_DOCS_PER_FILE input_files_per_partition=$QWEN_INPUT_FILES_PER_PARTITION embedding_max_chars=$QWEN_EMBEDDING_MAX_CHARS kmeans_files_per_group=$QWEN_KMEANS_FILES_PER_GROUP resume_from_embeddings=$QWEN_RESUME_FROM_EMBEDDINGS reuse_staged_inputs=$QWEN_REUSE_STAGED_INPUTS vllm_kwargs=$QWEN_VLLM_INIT_KWARGS_JSON"
build_seed42_data
train_seed42
run_repeats_and_report
log "Qwen3 direct orchestrator complete"
