#!/bin/bash

# Sequential FineWeb-EDU SemDeDup A/B orchestrator.
# Intended to be launched with nohup:
#
#   nohup bash runs/fineweb_edu_semdedup_ab_nohup_b200.sh all > all.log 2>&1 &

set -euo pipefail

MODE="${1:-all}"
if [[ "$MODE" != "smoke" && "$MODE" != "pilot-baseline" && "$MODE" != "pilot-semdedup" && "$MODE" != "pilot" && "$MODE" != "formal" && "$MODE" != "formal-semdedup" && "$MODE" != "formal-semdedup-report" && "$MODE" != "report" && "$MODE" != "resume" && "$MODE" != "all" ]]; then
    echo "Usage: $0 smoke|pilot-baseline|pilot-semdedup|pilot|formal|formal-semdedup|formal-semdedup-report|report|resume|all" >&2
    exit 2
fi

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_fineweb_edu}"
export RUN_ROOT="${RUN_ROOT:-$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality}"
export LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/nohup_logs}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

run_smoke() {
    log "Starting FineWeb-EDU SemDeDup smoke"
    RUN_ID="${SMOKE_RUN_ID:-smoke-fineweb-semdedup-exact}" \
    NUM_TRAIN_SHARDS="${SMOKE_NUM_TRAIN_SHARDS:-1}" \
    MAX_DOCS="${SMOKE_MAX_DOCS:-2000}" \
    STATS_MAX_DOCS="${SMOKE_STATS_MAX_DOCS:-2000}" \
    SEMD_BACKEND="${SMOKE_SEMD_BACKEND:-exact-smoke}" \
    DO_TRAIN=0 \
    DO_EVAL=0 \
    SKIP_TOKEN_STATS="${SMOKE_SKIP_TOKEN_STATS:-1}" \
    bash runs/fineweb_edu_semdedup_quality_b200.sh semdedup
    log "Finished FineWeb-EDU SemDeDup smoke"
}

run_pilot_baseline() {
    log "Starting FineWeb-EDU pilot baseline"
    RUN_ID="${PILOT_BASELINE_RUN_ID:-pilot-n8-i400-baseline}" \
    WANDB_RUN="${PILOT_BASELINE_WANDB_RUN:-fineweb-edu-nosd-n8-i400}" \
    NUM_TRAIN_SHARDS="${PILOT_NUM_TRAIN_SHARDS:-8}" \
    NUM_ITERATIONS="${PILOT_NUM_ITERATIONS:-400}" \
    FINAL_CORE_MAX_PER_TASK="${PILOT_FINAL_CORE_MAX_PER_TASK:-500}" \
    bash runs/fineweb_edu_semdedup_quality_b200.sh baseline
    log "Finished FineWeb-EDU pilot baseline"
}

run_pilot_semdedup() {
    log "Starting FineWeb-EDU pilot SemDeDup"
    RUN_ID="${PILOT_SEMDEDUP_RUN_ID:-pilot-n8-i400-semdedup-eps0p07}" \
    WANDB_RUN="${PILOT_SEMDEDUP_WANDB_RUN:-fineweb-edu-semdedup-eps0p07-n8-i400}" \
    NUM_TRAIN_SHARDS="${PILOT_NUM_TRAIN_SHARDS:-8}" \
    NUM_ITERATIONS="${PILOT_NUM_ITERATIONS:-400}" \
    SEMD_EPS="${SEMD_EPS:-0.07}" \
    SEMD_N_CLUSTERS="${SEMD_N_CLUSTERS:-100}" \
    SEMD_MODEL="${SEMD_MODEL:-google/embeddinggemma-300m}" \
    SEMD_RAY_TEMP_DIR="${PILOT_SEMD_RAY_TEMP_DIR:-/tmp/fwe_ray_pilot}" \
    SEMD_VLLM_ATTENTION_BACKEND="${SEMD_VLLM_ATTENTION_BACKEND:-TRITON_ATTN}" \
    SEMD_VLLM_ENFORCE_EAGER="${SEMD_VLLM_ENFORCE_EAGER:-1}" \
    SEMD_OVERWRITE="${PILOT_SEMD_OVERWRITE:-1}" \
    INSTALL_CURATOR="${INSTALL_CURATOR:-1}" \
    FINAL_CORE_MAX_PER_TASK="${PILOT_FINAL_CORE_MAX_PER_TASK:-500}" \
    bash runs/fineweb_edu_semdedup_quality_b200.sh semdedup
    log "Finished FineWeb-EDU pilot SemDeDup"
}

run_formal() {
    local run_id
    run_id="${FORMAL_RUN_ID:-full-n170-r9p5-eps0p07-$(date -u +%Y%m%dT%H%M%SZ)}"
    echo "$run_id" > "$LOG_ROOT/latest_formal_run_id.txt"
    log "Starting FineWeb-EDU formal A/B: $run_id"
    RUN_ID="$run_id" \
    NUM_TRAIN_SHARDS="${FORMAL_NUM_TRAIN_SHARDS:-170}" \
    PARAM_DATA_RATIO="${FORMAL_PARAM_DATA_RATIO:-9.5}" \
    SEMD_EPS="${SEMD_EPS:-0.07}" \
    SEMD_N_CLUSTERS="${SEMD_N_CLUSTERS:-100}" \
    SEMD_MODEL="${SEMD_MODEL:-google/embeddinggemma-300m}" \
    SEMD_RAY_TEMP_DIR="${FORMAL_SEMD_RAY_TEMP_DIR:-/tmp/fwe_ray_formal}" \
    SEMD_VLLM_ATTENTION_BACKEND="${SEMD_VLLM_ATTENTION_BACKEND:-TRITON_ATTN}" \
    SEMD_VLLM_ENFORCE_EAGER="${SEMD_VLLM_ENFORCE_EAGER:-1}" \
    SEMD_OVERWRITE="${FORMAL_SEMD_OVERWRITE:-0}" \
    CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-2000}" \
    CORE_METRIC_MAX_PER_TASK="${CORE_METRIC_MAX_PER_TASK:-500}" \
    FINAL_CORE_MAX_PER_TASK="${FINAL_CORE_MAX_PER_TASK:--1}" \
    INSTALL_CURATOR="${INSTALL_CURATOR:-1}" \
    bash runs/fineweb_edu_semdedup_quality_b200.sh both
    log "Finished FineWeb-EDU formal A/B: $run_id"
}

latest_run_dir() {
    local pattern="$1"
    find "$RUN_ROOT" -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n' \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

latest_formal_run_id() {
    if [[ -n "${FORMAL_RUN_ID:-}" ]]; then
        echo "$FORMAL_RUN_ID"
        return
    fi
    if [[ -s "$LOG_ROOT/latest_formal_run_id.txt" ]]; then
        cat "$LOG_ROOT/latest_formal_run_id.txt"
        return
    fi
    local latest_baseline base
    latest_baseline="$(latest_run_dir 'full-n170-r9p5-eps0p07-*baseline')"
    if [[ -z "$latest_baseline" ]]; then
        echo "Could not infer formal run id; set FORMAL_RUN_ID" >&2
        exit 1
    fi
    base="$(basename "$latest_baseline")"
    echo "${base%_baseline}"
}

run_formal_semdedup() {
    local run_id semdedup_run_id
    run_id="$(latest_formal_run_id)"
    semdedup_run_id="$run_id"
    if [[ "$semdedup_run_id" != *_semdedup ]]; then
        semdedup_run_id="${semdedup_run_id}_semdedup"
    fi
    echo "$run_id" > "$LOG_ROOT/latest_formal_run_id.txt"
    log "Starting FineWeb-EDU formal SemDeDup only: $semdedup_run_id"
    RUN_ID="$semdedup_run_id" \
    NUM_TRAIN_SHARDS="${FORMAL_NUM_TRAIN_SHARDS:-170}" \
    PARAM_DATA_RATIO="${FORMAL_PARAM_DATA_RATIO:-9.5}" \
    SEMD_EPS="${SEMD_EPS:-0.07}" \
    SEMD_N_CLUSTERS="${SEMD_N_CLUSTERS:-100}" \
    SEMD_MODEL="${SEMD_MODEL:-google/embeddinggemma-300m}" \
    SEMD_RAY_TEMP_DIR="${FORMAL_SEMD_RAY_TEMP_DIR:-/tmp/fwe_ray_formal}" \
    SEMD_VLLM_ATTENTION_BACKEND="${SEMD_VLLM_ATTENTION_BACKEND:-TRITON_ATTN}" \
    SEMD_VLLM_ENFORCE_EAGER="${SEMD_VLLM_ENFORCE_EAGER:-1}" \
    SEMD_OVERWRITE="${FORMAL_SEMD_OVERWRITE:-1}" \
    CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-2000}" \
    CORE_METRIC_MAX_PER_TASK="${CORE_METRIC_MAX_PER_TASK:-500}" \
    FINAL_CORE_MAX_PER_TASK="${FINAL_CORE_MAX_PER_TASK:--1}" \
    INSTALL_CURATOR="${INSTALL_CURATOR:-1}" \
    bash runs/fineweb_edu_semdedup_quality_b200.sh semdedup
    log "Finished FineWeb-EDU formal SemDeDup only: $semdedup_run_id"
}

run_report() {
    local baseline_run semdedup_run report_path
    baseline_run="${BASELINE_RUN:-$(latest_run_dir 'full-n170-r9p5-eps0p07-*baseline')}"
    semdedup_run="${SEMDEDUP_RUN:-$(latest_run_dir 'full-n170-r9p5-eps0p07-*semdedup')}"
    report_path="${REPORT_PATH:-$RUN_ROOT/fineweb_edu_full_n170_r9p5_eps0p07_comparison.md}"
    if [[ -z "$baseline_run" || -z "$semdedup_run" ]]; then
        echo "Could not find formal baseline/semdedup run dirs under $RUN_ROOT" >&2
        exit 1
    fi
    log "Generating comparison report"
    python -m scripts.compare_climbmix_experiments \
        --baseline "$baseline_run" \
        --semdedup "$semdedup_run" \
        --output "$report_path"
    log "Wrote report: $report_path"
}

case "$MODE" in
    smoke)
        run_smoke
        ;;
    pilot-baseline)
        run_pilot_baseline
        ;;
    pilot-semdedup)
        run_pilot_semdedup
        ;;
    pilot)
        run_pilot_baseline
        run_pilot_semdedup
        ;;
    formal)
        run_formal
        ;;
    formal-semdedup)
        run_formal_semdedup
        ;;
    formal-semdedup-report)
        run_formal_semdedup
        run_report
        ;;
    report)
        run_report
        ;;
    resume)
        run_pilot_semdedup
        run_formal
        run_report
        ;;
    all)
        run_smoke
        run_pilot_baseline
        run_pilot_semdedup
        run_formal
        run_report
        ;;
esac
