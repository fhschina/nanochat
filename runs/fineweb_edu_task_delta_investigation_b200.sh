#!/bin/bash

# FineWeb-EDU SemDeDup follow-up matrix for 8x B200.
# Heavy stages are intended to be launched by wait_and_run_followup.sh only after
# all GPUs are idle. Smoke mode performs CPU/plumbing checks only.

set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: $0 smoke|eps-sweep|randomdrop-control|removed-audit|boolq-analysis|seed-expansion|runtime-profile|report|all" >&2
    exit 2
fi

case "$MODE" in
    smoke|eps-sweep|randomdrop-control|removed-audit|boolq-analysis|seed-expansion|runtime-profile|report|all) ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac

export DATASET_TAG="${DATASET_TAG:-fineweb_edu}"
export NANOCHAT_DATASET_NAME="${NANOCHAT_DATASET_NAME:-fineweb_edu}"
export NANOCHAT_DATASET_URL="${NANOCHAT_DATASET_URL:-https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main}"
export NANOCHAT_DATASET_MAX_SHARD="${NANOCHAT_DATASET_MAX_SHARD:-1822}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_fineweb_edu}"
export QUALITY_ROOT="${QUALITY_ROOT:-$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality}"
export FOLLOWUP_ROOT="${FOLLOWUP_ROOT:-$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_followup}"
export INPUT_DATA_DIR="${INPUT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_fineweb_edu}"
export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

LOG_DIR="$FOLLOWUP_ROOT/logs"
PID_DIR="$FOLLOWUP_ROOT/pids"
DATA_DIR="$FOLLOWUP_ROOT/data"
mkdir -p "$LOG_DIR" "$PID_DIR" "$DATA_DIR"
echo "$$" > "$PID_DIR/$MODE.pid"
exec > >(tee -a "$LOG_DIR/$MODE.$RUN_TIMESTAMP.log") 2>&1

cd "$(dirname "$0")/.."
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

export NUM_GPUS="${NUM_GPUS:-8}"
export DEPTH="${DEPTH:-24}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
export NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-170}"
export PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-9.5}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-2000}"
export CORE_METRIC_MAX_PER_TASK="${CORE_METRIC_MAX_PER_TASK:-500}"
export FINAL_CORE_MAX_PER_TASK="${FINAL_CORE_MAX_PER_TASK:--1}"
export BASE_EVAL_MODES="${BASE_EVAL_MODES:-core,bpb,sample}"
export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"
export PREP_DATASET="${PREP_DATASET:-0}"
export INSTALL_CURATOR="${INSTALL_CURATOR:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export AUDIT_SAMPLE_SIZE="${AUDIT_SAMPLE_SIZE:-1000}"
export AUDIT_SEED="${AUDIT_SEED:-1337}"
export TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-128}"
export TOKENIZER_THREADS="${TOKENIZER_THREADS:-4}"
export SEMD_MODEL="${SEMD_MODEL_IDENTIFIER:-${SEMD_MODEL:-google/embeddinggemma-300m}}"
SEMD_MODEL_IDENTIFIER="$SEMD_MODEL"
export SEMD_N_CLUSTERS="${SEMD_N_CLUSTERS:-100}"
export SEMD_DISTANCE_METRIC="${SEMD_DISTANCE_METRIC:-cosine}"
export SEMD_WHICH_TO_KEEP="${SEMD_WHICH_TO_KEEP:-hard}"
export SEMD_PAIRWISE_BATCH_SIZE="${SEMD_PAIRWISE_BATCH_SIZE:-1024}"
export SEMD_VLLM_ATTENTION_BACKEND="${SEMD_VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export SEMD_VLLM_ENFORCE_EAGER="${SEMD_VLLM_ENFORCE_EAGER:-1}"
export SEMD_NO_RAY_PREINIT="${SEMD_NO_RAY_PREINIT:-0}"
export SEMD_RAY_TEMP_DIR="${SEMD_RAY_TEMP_DIR:-/tmp/ray_hfang_fineweb_followup}"
export SEMD_OVERWRITE="${SEMD_OVERWRITE:-0}"

REFERENCE_RUN_ID="${REFERENCE_RUN_ID:-full-n170-r9p5-eps0p07-20260622T192524Z}"
EXISTING_BASELINE_RUN_DIR="${EXISTING_BASELINE_RUN_DIR:-$QUALITY_ROOT/${REFERENCE_RUN_ID}_baseline}"
EXISTING_SEMDEDUP_RUN_DIR="${EXISTING_SEMDEDUP_RUN_DIR:-$QUALITY_ROOT/${REFERENCE_RUN_ID}_semdedup}"
EXISTING_SEMDEDUP_DATA_DIR="${EXISTING_SEMDEDUP_DATA_DIR:-$EXISTING_SEMDEDUP_RUN_DIR/base_data_fineweb_edu_semdedup_eps0p07_n170}"
BASELINE_STATS_SOURCE="${BASELINE_STATS_SOURCE:-$EXISTING_BASELINE_RUN_DIR/data_stats.json}"
SEMDEDUP_STATS_SOURCE="${SEMDEDUP_STATS_SOURCE:-$EXISTING_SEMDEDUP_RUN_DIR/data_stats.json}"
EPS_SWEEP_VALUES="${EPS_SWEEP_VALUES:-0.03 0.05 0.09 0.10}"
ALL_EPS_VALUES="${ALL_EPS_VALUES:-0.03 0.05 0.07 0.09 0.10}"
PROMOTED_EPS_FILE="$FOLLOWUP_ROOT/promoted_eps.txt"
DMON_STARTED=0

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

eps_slug() {
    printf '%s' "$1" | tr '.' 'p'
}

start_dmon_once() {
    if [[ "$DMON_STARTED" == "1" ]]; then
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "nvidia-smi not found; dmon unavailable"
        return
    fi
    local dmon_log="$LOG_DIR/dmon.$MODE.$RUN_TIMESTAMP.log"
    log "Starting bounded nvidia-smi dmon monitor: $dmon_log"
    nvidia-smi dmon -s u -d "${DMON_INTERVAL_SEC:-60}" -o DT -c "${DMON_SAMPLES:-4320}" > "$dmon_log" 2>&1 &
    echo "$!" > "$PID_DIR/dmon.$MODE.pid"
    DMON_STARTED=1
}

run_quality() (
    export RUN_ROOT RUN_ID RUN_TAG_BASELINE RUN_TAG_SEMDEDUP RUN_TAG_RANDOMDROP
    export WANDB_MODE WANDB_RUN WANDB_RUN_BASELINE WANDB_RUN_SEMDEDUP WANDB_RUN_RANDOMDROP
    export SEED CORE_EVAL_SEED PYTHON_BIN DO_TRAIN DO_EVAL PREP_DATASET SKIP_UV_SYNC INSTALL_CURATOR
    export NUM_GPUS DEPTH DEVICE_BATCH_SIZE NUM_TRAIN_SHARDS NUM_ITERATIONS PARAM_DATA_RATIO
    export INPUT_DATA_DIR STATS_MAX_DOCS MAX_DOCS AUDIT_SAMPLE_SIZE AUDIT_SEED SKIP_TOKEN_STATS
    export SEMD_EPS SEMD_MODEL SEMD_MODEL_IDENTIFIER SEMD_N_CLUSTERS SEMD_DISTANCE_METRIC SEMD_WHICH_TO_KEEP
    export SEMD_PAIRWISE_BATCH_SIZE SEMD_OUTPUT_DIR SEMD_OVERWRITE DO_SEMDEDUP
    export SEMD_VLLM_ATTENTION_BACKEND SEMD_VLLM_ENFORCE_EAGER SEMD_NO_RAY_PREINIT SEMD_RAY_TEMP_DIR
    export RANDOM_DROP_OUTPUT_DIR RANDOM_DROP_REMOVED_DOCS RANDOM_DROP_REMOVED_TOKENS RANDOM_DROP_SEED RANDOM_DROP_OVERWRITE DO_RANDOM_DROP
    export DATA_STATS_SOURCE DATA_STATS_SOURCE_BASELINE DATA_STATS_SOURCE_SEMDEDUP DATA_STATS_SOURCE_RANDOMDROP
    bash runs/fineweb_edu_semdedup_quality_b200.sh "$@"
)

semdedup_data_dir_for_eps() {
    local eps="$1"
    if [[ "$eps" == "0.07" || "$eps" == "0.070000" ]]; then
        printf '%s\n' "$EXISTING_SEMDEDUP_DATA_DIR"
    else
        printf '%s/base_data_fineweb_edu_semdedup_eps%s_n%s\n' "$DATA_DIR" "$(eps_slug "$eps")" "$NUM_TRAIN_SHARDS"
    fi
}

semdedup_seed42_run_dir_for_eps() {
    local eps="$1"
    if [[ "$eps" == "0.07" || "$eps" == "0.070000" ]]; then
        printf '%s\n' "$EXISTING_SEMDEDUP_RUN_DIR"
    else
        printf '%s/eps_sweep/full-semdedup-eps%s-seed42-n%s-r9p5\n' "$FOLLOWUP_ROOT" "$(eps_slug "$eps")" "$NUM_TRAIN_SHARDS"
    fi
}

semdedup_stats_source_for_eps() {
    local eps="$1"
    if [[ "$eps" == "0.07" || "$eps" == "0.070000" ]]; then
        printf '%s\n' "$SEMDEDUP_STATS_SOURCE"
    else
        local run_dir
        run_dir="$(semdedup_seed42_run_dir_for_eps "$eps")"
        if [[ -f "$run_dir/data_stats.json" ]]; then
            printf '%s\n' "$run_dir/data_stats.json"
        fi
    fi
}

run_baseline_seed() {
    local seed="$1"
    local run_root="$FOLLOWUP_ROOT/seed_expansion"
    local run_id="full-baseline-seed${seed}-n${NUM_TRAIN_SHARDS}-r9p5"
    local run_dir="$run_root/$run_id"
    if [[ -f "$run_dir/run_summary.json" ]]; then
        log "Skipping completed baseline seed=$seed: $run_dir"
        return
    fi
    start_dmon_once
    log "Training baseline seed=$seed"
    RUN_ROOT="$run_root" \
    RUN_ID="$run_id" \
    RUN_TAG_BASELINE="d${DEPTH}-fineweb_edu-nosd-n${NUM_TRAIN_SHARDS}-seed${seed}-followup" \
    WANDB_RUN="fineweb-edu-nosd-n${NUM_TRAIN_SHARDS}-seed${seed}-followup" \
    SEED="$seed" \
    DATA_STATS_SOURCE_BASELINE="$BASELINE_STATS_SOURCE" \
    DO_TRAIN=1 DO_EVAL=1 PREP_DATASET=0 SKIP_UV_SYNC=1 \
    run_quality baseline
}

run_semdedup_eps_seed() {
    local eps="$1"
    local seed="$2"
    local stage="$3"
    local data_dir="$4"
    local stats_source="${5:-}"
    local slug
    slug="$(eps_slug "$eps")"
    local run_root="$FOLLOWUP_ROOT/$stage"
    local run_id="full-semdedup-eps${slug}-seed${seed}-n${NUM_TRAIN_SHARDS}-r9p5"
    local run_dir="$run_root/$run_id"
    if [[ -f "$run_dir/run_summary.json" ]]; then
        log "Skipping completed SemDeDup eps=$eps seed=$seed: $run_dir"
        return
    fi
    local do_semdedup=1
    if [[ -f "$data_dir/semdedup_manifest.json" ]]; then
        do_semdedup=0
    fi
    local reuse_staged=0
    if [[ "$do_semdedup" == "1" && -f "$run_dir/semdedup_cache/input_stats.json" ]] && compgen -G "$run_dir/semdedup_cache/input_with_ids/*.parquet" > /dev/null; then
        reuse_staged=1
    fi
    local resume_from_embeddings=0
    if [[ "$do_semdedup" == "1" && -d "$run_dir/semdedup_cache/curator_cache/embeddings" ]] && compgen -G "$run_dir/semdedup_cache/curator_cache/embeddings/*.parquet" > /dev/null; then
        resume_from_embeddings=1
    fi
    local run_semd_overwrite="$SEMD_OVERWRITE"
    if [[ "$reuse_staged" == "1" && -d "$run_dir/semdedup_cache/curator_output" && ! -f "$data_dir/semdedup_manifest.json" ]]; then
        run_semd_overwrite=1
    fi
    start_dmon_once
    log "SemDeDup eps=$eps seed=$seed stage=$stage build=$do_semdedup reuse_staged=$reuse_staged resume_embeddings=$resume_from_embeddings overwrite=$run_semd_overwrite data=$data_dir"
    RUN_ROOT="$run_root" \
    RUN_ID="$run_id" \
    RUN_TAG_SEMDEDUP="d${DEPTH}-fineweb_edu-semdedup-eps${slug}-n${NUM_TRAIN_SHARDS}-seed${seed}-followup" \
    WANDB_RUN="fineweb-edu-semdedup-eps${slug}-n${NUM_TRAIN_SHARDS}-seed${seed}-followup" \
    SEED="$seed" \
    SEMD_EPS="$eps" \
    SEMD_OUTPUT_DIR="$data_dir" \
    SEMD_REUSE_STAGED_INPUTS="$reuse_staged" \
    SEMD_RESUME_FROM_EMBEDDINGS="$resume_from_embeddings" \
    SEMD_OVERWRITE="$run_semd_overwrite" \
    DO_SEMDEDUP="$do_semdedup" \
    DATA_STATS_SOURCE_SEMDEDUP="$stats_source" \
    DO_TRAIN=1 DO_EVAL=1 PREP_DATASET=0 SKIP_UV_SYNC=1 \
    run_quality semdedup
}

select_promoted_eps() {
    "$PYTHON_BIN" - "$QUALITY_ROOT" "$FOLLOWUP_ROOT" <<'PYSELECT'
import sys
from pathlib import Path
from scripts.aggregate_fineweb_edu_followup import _discover_runs, _select_promoted_eps
quality = Path(sys.argv[1])
followup = Path(sys.argv[2])
promotion = _select_promoted_eps(_discover_runs([quality, followup]))
print(promotion.get("promoted_eps") or "0.07")
PYSELECT
}

stage_eps_sweep() {
    for eps in $EPS_SWEEP_VALUES; do
        run_semdedup_eps_seed "$eps" 42 eps_sweep "$(semdedup_data_dir_for_eps "$eps")" ""
    done
    local promoted
    promoted="$(select_promoted_eps)"
    printf '%s\n' "$promoted" > "$PROMOTED_EPS_FILE"
    log "Promoted eps after sweep: $promoted"
}

stage_seed_expansion() {
    for seed in 45 46; do
        run_baseline_seed "$seed"
    done
    local promoted="0.07"
    if [[ -f "$PROMOTED_EPS_FILE" ]]; then
        promoted="$(tr -d '[:space:]' < "$PROMOTED_EPS_FILE")"
    else
        promoted="$(select_promoted_eps)"
        printf '%s\n' "$promoted" > "$PROMOTED_EPS_FILE"
    fi
    local data_dir stats_source
    data_dir="$(semdedup_data_dir_for_eps "$promoted")"
    stats_source="$(semdedup_stats_source_for_eps "$promoted")"
    if [[ "$promoted" == "0.07" || "$promoted" == "0.070000" ]]; then
        for seed in 45 46; do
            run_semdedup_eps_seed "$promoted" "$seed" seed_expansion "$data_dir" "$stats_source"
        done
    else
        for seed in 43 44 45 46; do
            run_semdedup_eps_seed "$promoted" "$seed" seed_expansion "$data_dir" "$stats_source"
        done
    fi
}

run_randomdrop_train() {
    local target_kind="$1"
    local target_value="$2"
    local rd_seed="$3"
    local train_seed="$4"
    local suffix="$5"
    local target_label="drop${target_value}"
    local removed_docs="$target_value"
    local removed_tokens=""
    if [[ "$target_kind" == "tokens" ]]; then
        target_label="droptok${target_value}"
        removed_docs=""
        removed_tokens="$target_value"
    fi
    local data_dir="$DATA_DIR/base_data_fineweb_edu_randomdrop_${target_label}_rdseed${rd_seed}_n${NUM_TRAIN_SHARDS}"
    local run_root="$FOLLOWUP_ROOT/randomdrop_control"
    local run_id="full-randomdrop-${suffix}-${target_label}-rdseed${rd_seed}-seed${train_seed}-n${NUM_TRAIN_SHARDS}-r9p5"
    local run_dir="$run_root/$run_id"
    if [[ -f "$run_dir/run_summary.json" ]]; then
        log "Skipping completed random-drop run: $run_dir"
        return
    fi
    local do_build=1
    if [[ -f "$data_dir/random_drop_manifest.json" ]]; then
        do_build=0
    fi
    local stats_source=""
    local seed42_stats="$run_root/full-randomdrop-${suffix}-${target_label}-rdseed${rd_seed}-seed42-n${NUM_TRAIN_SHARDS}-r9p5/data_stats.json"
    if [[ "$train_seed" != "42" && -f "$seed42_stats" ]]; then
        stats_source="$seed42_stats"
    fi
    start_dmon_once
    log "Random-drop $suffix target_kind=$target_kind target=$target_value rd_seed=$rd_seed train_seed=$train_seed build=$do_build"
    RUN_ROOT="$run_root" \
    RUN_ID="$run_id" \
    RUN_TAG_RANDOMDROP="d${DEPTH}-fineweb_edu-randomdrop-${suffix}-${target_label}-rdseed${rd_seed}-seed${train_seed}-followup" \
    WANDB_RUN="fineweb-edu-randomdrop-${suffix}-${target_label}-rdseed${rd_seed}-seed${train_seed}-followup" \
    SEED="$train_seed" \
    RANDOM_DROP_OUTPUT_DIR="$data_dir" \
    RANDOM_DROP_REMOVED_DOCS="$removed_docs" \
    RANDOM_DROP_REMOVED_TOKENS="$removed_tokens" \
    RANDOM_DROP_SEED="$rd_seed" \
    DO_RANDOM_DROP="$do_build" \
    DATA_STATS_SOURCE_RANDOMDROP="$stats_source" \
    DO_TRAIN=1 DO_EVAL=1 PREP_DATASET=0 SKIP_UV_SYNC=1 \
    run_quality randomdrop
}

should_expand_token_randomdrop() {
    "$PYTHON_BIN" - "$FOLLOWUP_ROOT" <<'PYCOMPARE'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]) / "randomdrop_control"
doc = root / "full-randomdrop-docmatched-drop633070-rdseed9001-seed42-n170-r9p5/run_summary.json"
tok = root / "full-randomdrop-tokenmatched-droptok775605001-rdseed9001-seed42-n170-r9p5/run_summary.json"
def load(path):
    return json.loads(path.read_text()) if path.exists() else {}
def metric(payload, key):
    value = (payload.get("metrics") or {}).get(key)
    return float(value) if value is not None else None
d = load(doc)
t = load(tok)
if not d or not t:
    print("0")
    raise SystemExit
bpb_diff = abs((metric(d, "base_eval_val_bpb") or 0) - (metric(t, "base_eval_val_bpb") or 0))
core_diff = abs((metric(d, "final_core") or 0) - (metric(t, "final_core") or 0))
print("1" if bpb_diff > 0.0003 or core_diff > 0.005 else "0")
PYCOMPARE
}

promoted_removed_docs() {
    local eps="$1"
    "$PYTHON_BIN" - "$QUALITY_ROOT" "$FOLLOWUP_ROOT" "$eps" <<'PYDOCS'
import sys
from pathlib import Path
from scripts.aggregate_fineweb_edu_followup import _discover_runs
quality, followup, eps = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
for run in _discover_runs([quality, followup]):
    if run.get("arm") == "semdedup" and run.get("eps") == eps and run.get("seed") == 42 and run.get("removed_docs"):
        print(run["removed_docs"])
        break
PYDOCS
}

stage_randomdrop_control() {
    for seed in 42 43 44 45 46; do
        run_randomdrop_train docs 633070 9001 "$seed" docmatched
    done
    run_randomdrop_train docs 633070 9002 42 docmatched-sensitivity
    run_randomdrop_train tokens 775605001 9001 42 tokenmatched
    if [[ "$(should_expand_token_randomdrop)" == "1" ]]; then
        run_randomdrop_train tokens 775605001 9001 43 tokenmatched
        run_randomdrop_train tokens 775605001 9001 44 tokenmatched
    else
        log "Token-matched random-drop seed expansion not triggered by seed42 BPB/CORE delta."
    fi
    local promoted="0.07"
    if [[ -f "$PROMOTED_EPS_FILE" ]]; then
        promoted="$(tr -d '[:space:]' < "$PROMOTED_EPS_FILE")"
    else
        promoted="$(select_promoted_eps)"
    fi
    if [[ "$promoted" != "0.07" && "$promoted" != "0.070000" ]]; then
        local docs
        docs="$(promoted_removed_docs "$promoted")"
        if [[ -n "$docs" ]]; then
            run_randomdrop_train docs "$docs" 9001 42 "promoted-eps${promoted//./p}-docmatched"
        else
            log "Could not infer removed_docs for promoted eps=$promoted; skipping promoted-eps random-drop."
        fi
    fi
}

run_removed_audit_for_eps() {
    local eps="$1"
    local sample_size="$2"
    local slug
    slug="$(eps_slug "$eps")"
    local run_dir
    run_dir="$(semdedup_seed42_run_dir_for_eps "$eps")"
    if [[ ! -f "$run_dir/run_summary.json" && ! -f "$run_dir/semdedup_manifest.json" ]]; then
        log "Skipping audit for eps=$eps; run artifacts not found at $run_dir"
        return
    fi
    local audit_root="$FOLLOWUP_ROOT/audits/eps${slug}"
    log "Heuristic removed/kept audit eps=$eps sample=1000"
    "$PYTHON_BIN" -m scripts.analyze_removed_samples \
        --run-dir "$run_dir" \
        --input-data-dir "$INPUT_DATA_DIR" \
        --output-dir "$audit_root/removed_heuristic" \
        --dataset-label FineWeb-EDU \
        --sample-size 1000 \
        --seed "$AUDIT_SEED" \
        --num-train-shards "$NUM_TRAIN_SHARDS"
    log "Pair audit eps=$eps sample=$sample_size"
    "$PYTHON_BIN" -m scripts.extract_semdedup_pair_audit \
        --run-dir "$run_dir" \
        --input-data-dir "$INPUT_DATA_DIR" \
        --output-dir "$audit_root/pair_audit" \
        --sample-size "$sample_size" \
        --seed "$AUDIT_SEED" \
        --num-train-shards "$NUM_TRAIN_SHARDS"
}

stage_removed_audit() {
    local promoted="0.07"
    if [[ -f "$PROMOTED_EPS_FILE" ]]; then
        promoted="$(tr -d '[:space:]' < "$PROMOTED_EPS_FILE")"
    else
        promoted="$(select_promoted_eps)"
    fi
    for eps in $ALL_EPS_VALUES; do
        local pair_sample=100
        if [[ "$eps" == "$promoted" ]]; then
            pair_sample=300
        fi
        run_removed_audit_for_eps "$eps" "$pair_sample"
    done
}

stage_report() {
    log "Aggregating FineWeb-EDU follow-up report"
    "$PYTHON_BIN" -m scripts.aggregate_fineweb_edu_followup \
        --quality-root "$QUALITY_ROOT" \
        --followup-root "$FOLLOWUP_ROOT" \
        --output "$FOLLOWUP_ROOT/FINEWEB_EDU_SEMDEDUP_FOLLOWUP_REPORT.md" \
        --json-output "$FOLLOWUP_ROOT/fineweb_edu_semdedup_followup_report.json"
}

stage_smoke() {
    log "Running static checks inside smoke"
    bash -n runs/fineweb_edu_task_delta_investigation_b200.sh
    "$PYTHON_BIN" -m py_compile \
        scripts/build_random_drop_dataset.py \
        scripts/analyze_removed_samples.py \
        scripts/extract_semdedup_pair_audit.py \
        scripts/aggregate_fineweb_edu_followup.py

    log "Running doc-matched random-drop smoke"
    RUN_ROOT="$FOLLOWUP_ROOT/smoke" \
    RUN_ID="smoke-randomdrop-doc" \
    NUM_TRAIN_SHARDS=1 MAX_DOCS=2000 STATS_MAX_DOCS=2000 \
    RANDOM_DROP_REMOVED_DOCS=100 RANDOM_DROP_REMOVED_TOKENS="" RANDOM_DROP_SEED=9001 RANDOM_DROP_OVERWRITE=1 \
    SKIP_TOKEN_STATS=1 DO_TRAIN=0 DO_EVAL=0 PREP_DATASET=0 SKIP_UV_SYNC=1 \
    run_quality randomdrop

    log "Running token-matched random-drop smoke"
    RUN_ROOT="$FOLLOWUP_ROOT/smoke" \
    RUN_ID="smoke-randomdrop-token" \
    NUM_TRAIN_SHARDS=1 MAX_DOCS=500 STATS_MAX_DOCS=500 \
    RANDOM_DROP_REMOVED_DOCS="" RANDOM_DROP_REMOVED_TOKENS=20000 RANDOM_DROP_SEED=9001 RANDOM_DROP_OVERWRITE=1 \
    SKIP_TOKEN_STATS=0 DO_TRAIN=0 DO_EVAL=0 PREP_DATASET=0 SKIP_UV_SYNC=1 \
    run_quality randomdrop

    log "Running exact-smoke SemDeDup plumbing smoke"
    RUN_ROOT="$FOLLOWUP_ROOT/smoke" \
    RUN_ID="smoke-semdedup-exact" \
    NUM_TRAIN_SHARDS=1 MAX_DOCS=2000 STATS_MAX_DOCS=2000 \
    SEMD_BACKEND=exact-smoke SEMD_OVERWRITE=1 SKIP_TOKEN_STATS=1 \
    DO_TRAIN=0 DO_EVAL=0 PREP_DATASET=0 SKIP_UV_SYNC=1 \
    run_quality semdedup

    stage_report
}

run_mode() {
    case "$1" in
        smoke) stage_smoke ;;
        eps-sweep) stage_eps_sweep ;;
        randomdrop-control) stage_randomdrop_control ;;
        removed-audit) stage_removed_audit ;;
        boolq-analysis) stage_report ;;
        seed-expansion) stage_seed_expansion ;;
        runtime-profile) stage_report ;;
        report) stage_report ;;
        all)
            stage_eps_sweep
            stage_seed_expansion
            stage_randomdrop_control
            stage_removed_audit
            stage_report
            ;;
    esac
}

log "Starting FineWeb-EDU follow-up mode=$MODE pid=$$ root=$FOLLOWUP_ROOT"
run_mode "$MODE"
log "Finished FineWeb-EDU follow-up mode=$MODE"
