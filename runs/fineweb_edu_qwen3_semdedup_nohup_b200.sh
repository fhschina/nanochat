#!/bin/bash

# Sequential FineWeb-EDU Qwen3-Embedding-8B SemDeDup runner.
# Intended usage:
#   nohup bash runs/fineweb_edu_qwen3_semdedup_nohup_b200.sh all > qwen3_fwe.log 2>&1 &

set -uo pipefail

MODE="${1:-all}"
if [[ "$MODE" != "all" && "$MODE" != "dry-run" && "$MODE" != "seed42" && "$MODE" != "repeats" && "$MODE" != "report" ]]; then
    echo "Usage: $0 all|dry-run|seed42|repeats|report" >&2
    exit 2
fi

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_fineweb_edu}"
export DATASET_TAG="${DATASET_TAG:-fineweb_edu}"
export INPUT_DATA_DIR="${INPUT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_fineweb_edu}"
export OLD_RUN_ROOT="${OLD_RUN_ROOT:-$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality}"
export RUN_ROOT="${RUN_ROOT:-$NANOCHAT_BASE_DIR/experiments/fineweb_edu_qwen3_semdedup_quality}"
export REPORT_DIR="${REPORT_DIR:-reports/semdedup_quality/fineweb_edu}"
export LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/nohup_logs}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$REPORT_DIR"

QWEN_BATCH_ID="${QWEN_BATCH_ID:-qwen3-n170-r9p5-eps0p07-$(date -u +%Y%m%dT%H%M%SZ)}"
QWEN_SEED42_RUN_ID="${QWEN_SEED42_RUN_ID:-${QWEN_BATCH_ID}-seed42_semdedup}"
QWEN_SEED43_RUN_ID="${QWEN_SEED43_RUN_ID:-${QWEN_BATCH_ID}-seed43_semdedup}"
QWEN_SEED44_RUN_ID="${QWEN_SEED44_RUN_ID:-${QWEN_BATCH_ID}-seed44_semdedup}"
QWEN_SEED42_RUN="${QWEN_SEED42_RUN:-$RUN_ROOT/$QWEN_SEED42_RUN_ID}"
QWEN_SEED43_RUN="${QWEN_SEED43_RUN:-$RUN_ROOT/$QWEN_SEED43_RUN_ID}"
QWEN_SEED44_RUN="${QWEN_SEED44_RUN:-$RUN_ROOT/$QWEN_SEED44_RUN_ID}"
QWEN_SEMD_OUTPUT_DIR="${QWEN_SEMD_OUTPUT_DIR:-$QWEN_SEED42_RUN/base_data_fineweb_edu_semdedup_eps0p07_n170}"
QWEN_DATA_STATS_SOURCE="${QWEN_DATA_STATS_SOURCE:-$QWEN_SEED42_RUN/data_stats.json}"
QWEN_DRY_RUN_CONFIG="${QWEN_DRY_RUN_CONFIG:-$RUN_ROOT/qwen3_curator_dry_run_config.json}"
QWEN_STAGED_DOCS_PER_FILE="${QWEN_STAGED_DOCS_PER_FILE:-512}"
QWEN_INPUT_FILES_PER_PARTITION="${QWEN_INPUT_FILES_PER_PARTITION:-1}"
QWEN_EMBEDDING_MAX_CHARS="${QWEN_EMBEDDING_MAX_CHARS:-16384}"
QWEN_KMEANS_FILES_PER_GROUP="${QWEN_KMEANS_FILES_PER_GROUP:-128}"
QWEN_VLLM_INIT_KWARGS_JSON="${QWEN_VLLM_INIT_KWARGS_JSON:-}"
if [[ -z "$QWEN_VLLM_INIT_KWARGS_JSON" ]]; then
    QWEN_VLLM_INIT_KWARGS_JSON='{"max_model_len":32768,"max_num_seqs":64,"max_num_batched_tokens":32768,"gpu_memory_utilization":0.82}'
fi

BASELINE_SEED42="${BASELINE_SEED42:-$OLD_RUN_ROOT/full-n170-r9p5-eps0p07-20260622T192524Z_baseline}"
BASELINE_SEED43="${BASELINE_SEED43:-$OLD_RUN_ROOT/repeat-20260624T000845Z-seed43-n170-r9p5-eps0p07_baseline}"
BASELINE_SEED44="${BASELINE_SEED44:-$OLD_RUN_ROOT/repeat-20260624T000845Z-seed44-n170-r9p5-eps0p07_baseline}"

EMBEDDINGGEMMA_SEED42="${EMBEDDINGGEMMA_SEED42:-$OLD_RUN_ROOT/full-n170-r9p5-eps0p07-20260622T192524Z_semdedup}"
EMBEDDINGGEMMA_SEED43="${EMBEDDINGGEMMA_SEED43:-$OLD_RUN_ROOT/repeat-20260624T000845Z-seed43-n170-r9p5-eps0p07_semdedup}"
EMBEDDINGGEMMA_SEED44="${EMBEDDINGGEMMA_SEED44:-$OLD_RUN_ROOT/repeat-20260624T000845Z-seed44-n170-r9p5-eps0p07_semdedup}"

QWEN_ECDF_SVG="${QWEN_ECDF_SVG:-$REPORT_DIR/qwen3_semdedup_similarity_ecdf.svg}"
QWEN_ECDF_STATS="${QWEN_ECDF_STATS:-$REPORT_DIR/qwen3_semdedup_similarity_ecdf.json}"
QWEN_FINAL_REPORT="${QWEN_FINAL_REPORT:-$REPORT_DIR/qwen3_final_report.md}"
QWEN_REPORT_JSON="${QWEN_REPORT_JSON:-$REPORT_DIR/qwen3_final_report_summary.json}"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

check_path() {
    local path="$1"
    local label="$2"
    if [[ ! -e "$path" ]]; then
        echo "Missing $label: $path" >&2
        return 1
    fi
    return 0
}

check_reference_inputs() {
    local missing=0
    for run_dir in "$BASELINE_SEED42" "$BASELINE_SEED43" "$BASELINE_SEED44"; do
        check_path "$run_dir/run_summary.json" "baseline run summary" || missing=1
        check_path "$run_dir/base_eval_core.csv" "baseline CORE CSV" || missing=1
        check_path "$run_dir/data_stats.json" "baseline data stats" || missing=1
    done
    for run_dir in "$EMBEDDINGGEMMA_SEED42" "$EMBEDDINGGEMMA_SEED43" "$EMBEDDINGGEMMA_SEED44"; do
        check_path "$run_dir/run_summary.json" "EmbeddingGemma run summary" || missing=1
        check_path "$run_dir/base_eval_core.csv" "EmbeddingGemma CORE CSV" || missing=1
        check_path "$run_dir/data_stats.json" "EmbeddingGemma data stats" || missing=1
        check_path "$run_dir/semdedup_manifest.json" "EmbeddingGemma SemDeDup manifest" || missing=1
    done
    return "$missing"
}

check_qwen_seed42_outputs() {
    local missing=0
    check_path "$QWEN_SEED42_RUN/semdedup_manifest.json" "Qwen3 seed42 SemDeDup manifest" || missing=1
    check_path "$QWEN_SEED42_RUN/data_stats.json" "Qwen3 seed42 data stats" || missing=1
    check_path "$QWEN_SEMD_OUTPUT_DIR" "Qwen3 seed42 deduped data dir" || missing=1
    check_path "$QWEN_SEED42_RUN/semdedup_cache/curator_cache/semantic_dedup/pairwise_results" "Qwen3 seed42 pairwise results" || missing=1
    return "$missing"
}

dry_run_qwen_curator_config() {
    log "Dry-running Qwen3 Curator config: $QWEN_DRY_RUN_CONFIG"
    .venv/bin/python -m scripts.build_semdedup_dataset \
        --model-identifier qwen3-embedding-8b \
        --num-train-shards 170 \
        --staged-docs-per-file "$QWEN_STAGED_DOCS_PER_FILE" \
        --embedding-max-chars "$QWEN_EMBEDDING_MAX_CHARS" \
        --embedding-vllm-init-kwargs-json "$QWEN_VLLM_INIT_KWARGS_JSON" \
        --input-files-per-partition "$QWEN_INPUT_FILES_PER_PARTITION" \
        --kmeans-files-per-group "$QWEN_KMEANS_FILES_PER_GROUP" \
        --eps 0.07 \
        --n-clusters 100 \
        --dry-run-curator-config \
        > "$QWEN_DRY_RUN_CONFIG"
    local status=$?
    if [[ "$status" -ne 0 ]]; then
        log "Qwen3 Curator config dry-run failed with status $status"
        return "$status"
    fi
    if ! grep -q '"resolved_model_identifier": "Qwen/Qwen3-Embedding-8B"' "$QWEN_DRY_RUN_CONFIG"; then
        echo "Dry-run did not resolve to Qwen/Qwen3-Embedding-8B" >&2
        return 1
    fi
    for needle in '"runner": "pooling"' '"convert": "embed"' '"dtype": "bfloat16"' '"enforce_eager": true' '"backend": "TRITON_ATTN"' '"resolved_embedding_dim": 4096' '"kmeans_files_per_group": 128'; do
        if ! grep -q "$needle" "$QWEN_DRY_RUN_CONFIG"; then
            echo "Dry-run config is missing expected Qwen3 setting: $needle" >&2
            return 1
        fi
    done
    log "Qwen3 Curator config dry-run passed"
}

run_seed42() {
    log "Starting Qwen3 seed 42 run: $QWEN_SEED42_RUN_ID"
    SEED=42 \
    RUN_ID="$QWEN_SEED42_RUN_ID" \
    RUN_TIMESTAMP="${QWEN_BATCH_ID}_seed42" \
    WANDB_RUN_SEMDEDUP="fineweb-edu-qwen3-semdedup-eps0p07-n170-seed42" \
    RUN_TAG_SEMDEDUP="d24-fineweb_edu-qwen3-semdedup-eps0p07-n170-seed42-${QWEN_BATCH_ID}" \
    NUM_TRAIN_SHARDS=170 \
    PARAM_DATA_RATIO=9.5 \
    SEMD_EPS=0.07 \
    SEMD_N_CLUSTERS=100 \
    SEMD_MODEL_IDENTIFIER=qwen3-embedding-8b \
    SEMD_OUTPUT_DIR="$QWEN_SEMD_OUTPUT_DIR" \
    SEMD_STAGED_DOCS_PER_FILE="$QWEN_STAGED_DOCS_PER_FILE" \
    SEMD_INPUT_FILES_PER_PARTITION="$QWEN_INPUT_FILES_PER_PARTITION" \
    SEMD_EMBEDDING_MAX_CHARS="$QWEN_EMBEDDING_MAX_CHARS" \
    SEMD_KMEANS_FILES_PER_GROUP="$QWEN_KMEANS_FILES_PER_GROUP" \
    SEMD_VLLM_INIT_KWARGS_JSON="$QWEN_VLLM_INIT_KWARGS_JSON" \
    SEMD_RAY_TEMP_DIR="${QWEN_SEMD_RAY_TEMP_DIR:-/tmp/fwe_qwen3_ray_seed42}" \
    SEMD_OVERWRITE="${QWEN_SEMD_OVERWRITE:-0}" \
    DO_SEMDEDUP=1 \
    PREP_DATASET="${QWEN_PREP_DATASET:-1}" \
    INSTALL_CURATOR="${QWEN_INSTALL_CURATOR:-1}" \
    CORE_METRIC_EVERY=2000 \
    CORE_METRIC_MAX_PER_TASK=500 \
    FINAL_CORE_MAX_PER_TASK=-1 \
    bash runs/fineweb_edu_semdedup_quality_b200.sh semdedup
    local status=$?
    if [[ "$status" -ne 0 ]]; then
        log "Qwen3 seed 42 run failed with status $status"
        return "$status"
    fi
    check_qwen_seed42_outputs
}

run_repeat_seed() {
    local seed="$1"
    local run_id
    if [[ "$seed" == "43" ]]; then
        run_id="$QWEN_SEED43_RUN_ID"
    elif [[ "$seed" == "44" ]]; then
        run_id="$QWEN_SEED44_RUN_ID"
    else
        echo "Unsupported repeat seed: $seed" >&2
        return 2
    fi

    if ! check_qwen_seed42_outputs; then
        return 1
    fi

    log "Starting Qwen3 repeat seed $seed using seed42 deduped data: $run_id"
    SEED="$seed" \
    RUN_ID="$run_id" \
    RUN_TIMESTAMP="${QWEN_BATCH_ID}_seed${seed}" \
    WANDB_RUN_SEMDEDUP="fineweb-edu-qwen3-semdedup-eps0p07-n170-seed${seed}" \
    RUN_TAG_SEMDEDUP="d24-fineweb_edu-qwen3-semdedup-eps0p07-n170-seed${seed}-${QWEN_BATCH_ID}" \
    NUM_TRAIN_SHARDS=170 \
    PARAM_DATA_RATIO=9.5 \
    SEMD_EPS=0.07 \
    SEMD_N_CLUSTERS=100 \
    SEMD_MODEL_IDENTIFIER=qwen3-embedding-8b \
    SEMD_OUTPUT_DIR="$QWEN_SEMD_OUTPUT_DIR" \
    DO_SEMDEDUP=0 \
    PREP_DATASET=0 \
    SKIP_UV_SYNC=1 \
    INSTALL_CURATOR=0 \
    SEMD_OVERWRITE=0 \
    DATA_STATS_SOURCE_SEMDEDUP="$QWEN_DATA_STATS_SOURCE" \
    CORE_METRIC_EVERY=2000 \
    CORE_METRIC_MAX_PER_TASK=500 \
    FINAL_CORE_MAX_PER_TASK=-1 \
    bash runs/fineweb_edu_semdedup_quality_b200.sh semdedup
    local status=$?
    if [[ "$status" -ne 0 ]]; then
        log "Qwen3 repeat seed $seed failed with status $status"
        return "$status"
    fi
    log "Finished Qwen3 repeat seed $seed"
}

generate_reports() {
    local status=0
    if ! check_qwen_seed42_outputs; then
        return 1
    fi

    log "Generating Qwen3 SemDeDup similarity ECDF"
    .venv/bin/python -m scripts.plot_semdedup_similarity_ecdf \
        --pairwise-dir "$QWEN_SEED42_RUN/semdedup_cache/curator_cache/semantic_dedup/pairwise_results" \
        --manifest "$QWEN_SEED42_RUN/semdedup_manifest.json" \
        --output "$QWEN_ECDF_SVG" \
        --stats-output "$QWEN_ECDF_STATS" \
        --dataset-label fineweb-edu-qwen3
    status=$?
    if [[ "$status" -ne 0 ]]; then
        log "Qwen3 ECDF generation failed with status $status"
        return "$status"
    fi

    log "Generating Qwen3 final report"
    .venv/bin/python -m scripts.compare_fineweb_edu_qwen3_repeats \
        --baseline-run "$BASELINE_SEED42" \
        --baseline-run "$BASELINE_SEED43" \
        --baseline-run "$BASELINE_SEED44" \
        --qwen-run "$QWEN_SEED42_RUN" \
        --qwen-run "$QWEN_SEED43_RUN" \
        --qwen-run "$QWEN_SEED44_RUN" \
        --embeddinggemma-run "$EMBEDDINGGEMMA_SEED42" \
        --embeddinggemma-run "$EMBEDDINGGEMMA_SEED43" \
        --embeddinggemma-run "$EMBEDDINGGEMMA_SEED44" \
        --ecdf-svg "$QWEN_ECDF_SVG" \
        --ecdf-stats "$QWEN_ECDF_STATS" \
        --curator-dry-run-config "$QWEN_DRY_RUN_CONFIG" \
        --output "$QWEN_FINAL_REPORT" \
        --json-output "$QWEN_REPORT_JSON" \
        --require-complete
    status=$?
    if [[ "$status" -ne 0 ]]; then
        log "Qwen3 final report generation failed with status $status"
        return "$status"
    fi
    log "Wrote Qwen3 final report: $QWEN_FINAL_REPORT"
}

main() {
    local status=0
    echo "$QWEN_BATCH_ID" > "$LOG_ROOT/latest_qwen3_batch_id.txt"
    log "Qwen3 batch: $QWEN_BATCH_ID"
    log "Qwen3 run root: $RUN_ROOT"
    log "Qwen3 seed42 deduped data: $QWEN_SEMD_OUTPUT_DIR"

    if ! check_reference_inputs; then
        exit 1
    fi

    if [[ "$MODE" == "all" || "$MODE" == "dry-run" || "$MODE" == "seed42" ]]; then
        if ! dry_run_qwen_curator_config; then
            status=1
        fi
    fi

    if [[ "$status" -eq 0 && ( "$MODE" == "all" || "$MODE" == "seed42" ) ]]; then
        if ! run_seed42; then
            status=1
        fi
    fi

    if [[ "$status" -eq 0 && ( "$MODE" == "all" || "$MODE" == "repeats" ) ]]; then
        for seed in 43 44; do
            if ! run_repeat_seed "$seed"; then
                status=1
                break
            fi
        done
    fi

    if [[ "$status" -eq 0 && ( "$MODE" == "all" || "$MODE" == "report" ) ]]; then
        if ! generate_reports; then
            status=1
        fi
    fi

    log "Qwen3 runner exit status: $status"
    exit "$status"
}

main
