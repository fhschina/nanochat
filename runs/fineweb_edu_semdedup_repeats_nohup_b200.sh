#!/bin/bash

# Sequential FineWeb-EDU SemDeDup repeat runner.
# Intended usage:
#   nohup bash runs/fineweb_edu_semdedup_repeats_nohup_b200.sh all > repeat.log 2>&1 &

set -uo pipefail

MODE="${1:-all}"
if [[ "$MODE" != "all" && "$MODE" != "report" ]]; then
    echo "Usage: $0 all|report" >&2
    exit 2
fi

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_fineweb_edu}"
export RUN_ROOT="${RUN_ROOT:-$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality}"
export LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/nohup_logs}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

REPEAT_SEEDS="${REPEAT_SEEDS:-43 44}"
REPEAT_BATCH_ID="${REPEAT_BATCH_ID:-repeat-$(date -u +%Y%m%dT%H%M%SZ)}"
REFERENCE_RUN_ID="${REFERENCE_RUN_ID:-full-n170-r9p5-eps0p07-20260622T192524Z}"
REFERENCE_BASELINE_RUN="$RUN_ROOT/${REFERENCE_RUN_ID}_baseline"
REFERENCE_SEMDEDUP_RUN="$RUN_ROOT/${REFERENCE_RUN_ID}_semdedup"
SHARED_SEMD_OUTPUT_DIR="${SHARED_SEMD_OUTPUT_DIR:-$REFERENCE_SEMDEDUP_RUN/base_data_fineweb_edu_semdedup_eps0p07_n170}"
BASELINE_STATS_SOURCE="${BASELINE_STATS_SOURCE:-$REFERENCE_BASELINE_RUN/data_stats.json}"
SEMDEDUP_STATS_SOURCE="${SEMDEDUP_STATS_SOURCE:-$REFERENCE_SEMDEDUP_RUN/data_stats.json}"
ECDF_SVG="${ECDF_SVG:-$RUN_ROOT/fineweb_edu_semdedup_similarity_ecdf.svg}"
ECDF_STATS="${ECDF_STATS:-$RUN_ROOT/fineweb_edu_semdedup_similarity_ecdf.json}"
MULTI_REPORT="${MULTI_REPORT:-$RUN_ROOT/fineweb_edu_repeats_report.md}"
MULTI_REPORT_JSON="${MULTI_REPORT_JSON:-$RUN_ROOT/fineweb_edu_repeats_report_summary.json}"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

check_reference_inputs() {
    local missing=0
    for path in "$REFERENCE_BASELINE_RUN/run_summary.json" "$REFERENCE_SEMDEDUP_RUN/run_summary.json" "$SHARED_SEMD_OUTPUT_DIR" "$BASELINE_STATS_SOURCE" "$SEMDEDUP_STATS_SOURCE"; do
        if [[ ! -e "$path" ]]; then
            echo "Missing required reference artifact: $path" >&2
            missing=1
        fi
    done
    return "$missing"
}

run_seed_pair() {
    local seed="$1"
    local run_id="${REPEAT_BATCH_ID}-seed${seed}-n170-r9p5-eps0p07"
    local baseline_run="$RUN_ROOT/${run_id}_baseline"
    local semdedup_run="$RUN_ROOT/${run_id}_semdedup"
    local pair_report="$RUN_ROOT/${run_id}_comparison.md"

    log "Starting repeat seed $seed: $run_id"
    SEED="$seed" \
    RUN_ID="$run_id" \
    RUN_TIMESTAMP="${REPEAT_BATCH_ID}_seed${seed}" \
    WANDB_RUN_BASELINE="fineweb-edu-nosd-n170-seed${seed}" \
    WANDB_RUN_SEMDEDUP="fineweb-edu-semdedup-eps0p07-n170-seed${seed}" \
    NUM_TRAIN_SHARDS=170 \
    PARAM_DATA_RATIO=9.5 \
    SEMD_EPS=0.07 \
    SEMD_N_CLUSTERS=100 \
    SEMD_MODEL=google/embeddinggemma-300m \
    SEMD_OUTPUT_DIR="$SHARED_SEMD_OUTPUT_DIR" \
    DO_SEMDEDUP=0 \
    SEMD_OVERWRITE=0 \
    PREP_DATASET=0 \
    SKIP_UV_SYNC=1 \
    INSTALL_CURATOR=0 \
    DATA_STATS_SOURCE_BASELINE="$BASELINE_STATS_SOURCE" \
    DATA_STATS_SOURCE_SEMDEDUP="$SEMDEDUP_STATS_SOURCE" \
    CORE_METRIC_EVERY=2000 \
    CORE_METRIC_MAX_PER_TASK=500 \
    FINAL_CORE_MAX_PER_TASK=-1 \
    bash runs/fineweb_edu_semdedup_quality_b200.sh both
    local train_status=$?
    if [[ "$train_status" -ne 0 ]]; then
        log "Repeat seed $seed failed with status $train_status"
        return "$train_status"
    fi

    if [[ -f "$baseline_run/run_summary.json" && -f "$semdedup_run/run_summary.json" ]]; then
        log "Generating per-seed comparison for seed $seed"
        .venv/bin/python -m scripts.compare_climbmix_experiments \
            --baseline "$baseline_run" \
            --semdedup "$semdedup_run" \
            --output "$pair_report"
    fi
    log "Finished repeat seed $seed"
}

generate_reports() {
    log "Generating SemDeDup ECDF SVG"
    .venv/bin/python -m scripts.plot_semdedup_similarity_ecdf \
        --pairwise-dir "$REFERENCE_SEMDEDUP_RUN/semdedup_cache/curator_cache/semantic_dedup/pairwise_results" \
        --manifest "$REFERENCE_SEMDEDUP_RUN/semdedup_manifest.json" \
        --output "$ECDF_SVG" \
        --stats-output "$ECDF_STATS" \
        --dataset-label fineweb-edu

    log "Generating multi-seed repeat report"
    .venv/bin/python -m scripts.compare_fineweb_edu_repeats \
        --run-root "$RUN_ROOT" \
        --output "$MULTI_REPORT" \
        --ecdf-svg "$ECDF_SVG" \
        --ecdf-stats "$ECDF_STATS" \
        --json-output "$MULTI_REPORT_JSON"
    log "Wrote multi-seed report: $MULTI_REPORT"
}

main() {
    local status=0
    echo "$REPEAT_BATCH_ID" > "$LOG_ROOT/latest_repeat_batch_id.txt"
    log "repeat batch: $REPEAT_BATCH_ID"
    log "repeat seeds: $REPEAT_SEEDS"

    if ! check_reference_inputs; then
        exit 1
    fi

    if [[ "$MODE" == "all" ]]; then
        for seed in $REPEAT_SEEDS; do
            if ! run_seed_pair "$seed"; then
                status=1
            fi
        done
    fi

    if ! generate_reports; then
        status=1
    fi
    log "repeat runner exit status: $status"
    exit "$status"
}

main
