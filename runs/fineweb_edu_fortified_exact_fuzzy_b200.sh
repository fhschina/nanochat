#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

MODE="${1:-all}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CACHE_BASE="${XDG_CACHE_HOME:-${HOME}/.cache}"
RUN_ROOT="${RUN_ROOT:-$CACHE_BASE/nanochat_b200_fineweb_edu_fortified_exact_fuzzy}"
RAW_DATA_DIR="${RAW_DATA_DIR:-$RUN_ROOT/raw}"
WORK_DIR="${WORK_DIR:-$RUN_ROOT/work}"
OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-$RUN_ROOT/post_fuzzy}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$CACHE_BASE/nanochat_b200_fineweb_edu/tokenizer}"
REPORT_PATH="${REPORT_PATH:-$ROOT_DIR/reports/semdedup_quality/fineweb_edu_fortified/exact_fuzzy/final_report.md}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
DOCS_PER_SHARD="${DOCS_PER_SHARD:-100000}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-128}"
TOKENIZER_THREADS="${TOKENIZER_THREADS:-8}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-12}"
INPUT_BLOCKSIZE="${INPUT_BLOCKSIZE:-512MiB}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
REMOVAL_WORKERS="${REMOVAL_WORKERS:-32}"
REMOVAL_BATCH_SIZE="${REMOVAL_BATCH_SIZE:-8192}"
REMOVAL_CHECKPOINT_FILES="${REMOVAL_CHECKPOINT_FILES:-25}"
CLEANUP_INTERMEDIATES="${CLEANUP_INTERMEDIATES:-1}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$LOG_DIR" "$(dirname "$REPORT_PATH")"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ]]; then
    echo "NanoChat tokenizer not found: $TOKENIZER_DIR/tokenizer.pkl" >&2
    exit 1
fi

run_materialize() {
    local args=(
        --output-dir "$RAW_DATA_DIR"
        --docs-per-shard "$DOCS_PER_SHARD"
        --tokenizer-dir "$TOKENIZER_DIR"
        --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE"
        --tokenizer-threads "$TOKENIZER_THREADS"
        --workers "$MATERIALIZE_WORKERS"
        --fast-exit
    )
    if [[ "$OVERWRITE" == "1" ]]; then
        args+=(--overwrite)
    elif [[ -f "$RAW_DATA_DIR/materialize_manifest.json" ]]; then
        args+=(--resume)
    fi
    "$PYTHON_BIN" -m scripts.materialize_fineweb_edu_fortified_parallel "${args[@]}" \
        2>&1 | tee "$LOG_DIR/materialize.log"
}

dedup_args() {
    DEDUP_ARGS=(
        --input-data-dir "$RAW_DATA_DIR"
        --output-data-dir "$OUTPUT_DATA_DIR"
        --work-dir "$WORK_DIR"
        --report-path "$REPORT_PATH"
        --input-blocksize "$INPUT_BLOCKSIZE"
        --tokenizer-dir "$TOKENIZER_DIR"
        --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE"
        --tokenizer-threads "$TOKENIZER_THREADS"
        --ray-num-cpus "$RAY_NUM_CPUS"
        --removal-workers "$REMOVAL_WORKERS"
        --removal-batch-size "$REMOVAL_BATCH_SIZE"
        --removal-checkpoint-files "$REMOVAL_CHECKPOINT_FILES"
    )
    if [[ "$CLEANUP_INTERMEDIATES" == "1" ]]; then
        DEDUP_ARGS+=(--cleanup-intermediates)
    fi
}

run_dedup() {
    dedup_args
    if [[ "$OVERWRITE" == "1" ]]; then
        DEDUP_ARGS+=(--overwrite)
    elif [[ -f "$WORK_DIR/exact_fuzzy_manifest.json" ]]; then
        DEDUP_ARGS+=(--resume)
    fi
    "$PYTHON_BIN" -m scripts.build_exact_fuzzy_dedup_dataset "${DEDUP_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/dedup.log"
}

run_one_stage() {
    local stage="$1"
    dedup_args
    DEDUP_ARGS+=(--stage "$stage")
    if [[ "$stage" == "exact" && "$OVERWRITE" == "1" ]]; then
        DEDUP_ARGS+=(--overwrite)
    fi
    "$PYTHON_BIN" -m scripts.build_exact_fuzzy_dedup_dataset "${DEDUP_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/${stage}.log"
}

run_smoke() {
    local smoke_root="${SMOKE_ROOT:-/tmp/nanochat_fineweb_edu_fortified_exact_fuzzy_smoke}"
    local smoke_input="$smoke_root/raw"
    local smoke_work="$smoke_root/work"
    local smoke_output="$smoke_root/post_fuzzy"
    local smoke_report="$smoke_root/final_report.md"

    "$PYTHON_BIN" -m scripts.make_exact_fuzzy_smoke_fixture \
        --output-dir "$smoke_input" \
        --tokenizer-dir "$TOKENIZER_DIR" \
        --overwrite
    "$PYTHON_BIN" -m scripts.build_exact_fuzzy_dedup_dataset \
        --input-data-dir "$smoke_input" \
        --output-data-dir "$smoke_output" \
        --work-dir "$smoke_work" \
        --report-path "$smoke_report" \
        --input-blocksize 64MiB \
        --tokenizer-dir "$TOKENIZER_DIR" \
        --tokenizer-batch-size 16 \
        --tokenizer-threads 4 \
        --ray-num-cpus 8 \
        --overwrite \
        2>&1 | tee "$LOG_DIR/smoke.log"

    "$PYTHON_BIN" -c '
import json, pathlib, sys
manifest = json.loads((pathlib.Path(sys.argv[1]) / "exact_fuzzy_manifest.json").read_text())
exact = manifest["stages"]["exact"]["num_duplicates"]
fuzzy = manifest["stages"]["fuzzy"]["num_duplicates"]
if exact != 2:
    raise SystemExit(f"expected 2 exact removals, got {exact}")
if fuzzy < 1:
    raise SystemExit(f"expected at least 1 fuzzy removal, got {fuzzy}")
print(f"Smoke assertions passed: exact_removed={exact}, fuzzy_removed={fuzzy}")
' "$smoke_work"
    echo "Smoke report: $smoke_report"
}

case "$MODE" in
    smoke)
        run_smoke
        ;;
    materialize)
        run_materialize
        ;;
    dedup)
        run_dedup
        ;;
    exact|fuzzy|stats|report)
        run_one_stage "$MODE"
        ;;
    all)
        run_materialize
        run_dedup
        ;;
    *)
        echo "Usage: $0 {smoke|materialize|exact|fuzzy|stats|report|dedup|all}" >&2
        exit 2
        ;;
esac
