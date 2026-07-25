#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$REPO_ROOT/.venv/bin/torchrun}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu}"

SOURCE_ROOT="${SOURCE_ROOT:-/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu_fortified_exact_fuzzy}"
FUZZY_SOURCE="${FUZZY_SOURCE:-$SOURCE_ROOT/post_fuzzy}"
RAW_SOURCE="${RAW_SOURCE:-$SOURCE_ROOT/raw}"
VALIDATION_FILE="${VALIDATION_FILE:-$NANOCHAT_BASE_DIR/base_data_fineweb_edu/shard_01822.parquet}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu_fortified_fuzzy_ab}"
FUZZY_VIEW="${FUZZY_VIEW:-$EXPERIMENT_ROOT/data/fuzzy_hash_0_08}"
RAW_VIEW="${RAW_VIEW:-$EXPERIMENT_ROOT/data/raw_hash_0_08}"
RUN_ROOT="${RUN_ROOT:-$EXPERIMENT_ROOT/runs}"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/reports/semdedup_quality/fineweb_edu_fortified/fuzzy_ab}"
PIONEER_REPORT_DIR="${PIONEER_REPORT_DIR:-$REPORT_DIR/pioneer_seed42}"
AUDIT_ROOT="$REPO_ROOT/reports/semdedup_quality/fineweb_edu_fortified/exact_fuzzy/fuzzy_audit"

NUM_GPUS="${NUM_GPUS:-8}"
SEEDS=(42 43 44)
FULL_STEPS=6612
TOTAL_BATCH_SIZE=1048576
DEVICE_BATCH_SIZE=16
MAX_SEQ_LEN=2048
EVAL_TOKENS=41943040
OVERWRITE_VIEW="${OVERWRITE_VIEW:-0}"
VIEW_WORKERS="${VIEW_WORKERS:-32}"

cd "$REPO_ROOT"

require_file() {
    [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}

preflight() {
    require_file "$PYTHON_BIN"
    require_file "$TORCHRUN_BIN"
    require_file "$VALIDATION_FILE"
    require_file "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
    require_file "$SOURCE_ROOT/work/exact_fuzzy_manifest.json"
    [[ -d "$FUZZY_SOURCE" ]] || { echo "Missing fuzzy source: $FUZZY_SOURCE" >&2; exit 1; }
    [[ -d "$RAW_SOURCE" ]] || { echo "Missing raw source: $RAW_SOURCE" >&2; exit 1; }
    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    [[ "$gpu_count" -eq "$NUM_GPUS" ]] || { echo "Expected $NUM_GPUS GPUs, found $gpu_count" >&2; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
}

prepare_view() {
    local arm="$1"
    local source view
    if [[ "$arm" == "fuzzy" ]]; then
        source="$FUZZY_SOURCE"; view="$FUZZY_VIEW"
    else
        source="$RAW_SOURCE"; view="$RAW_VIEW"
    fi
    local args=(
        -m scripts.build_fortified_hash_view
        --arm "$arm"
        --source-root "$source"
        --output-dir "$view"
        --validation-file "$VALIDATION_FILE"
        --shuffle-seed 20260722
        --hash-start 0
        --hash-end 0.08
        --num-buckets 512
        --buffer-rows 2000000
        --workers "$VIEW_WORKERS"
    )
    if [[ "$arm" == "fuzzy" ]]; then
        args+=(--min-source-tokens 14000000000)
    else
        require_file "$FUZZY_VIEW/view_manifest.json"
        args+=(--fuzzy-view-dir "$FUZZY_VIEW")
    fi
    if [[ "$OVERWRITE_VIEW" == "1" ]]; then args+=(--overwrite); fi
    "$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$EXPERIMENT_ROOT/prepare_${arm}.log"
}

run_one() {
    local kind="$1" arm="$2" seed="$3"
    local steps eval_every eval_tokens core_every core_max
    if [[ "$kind" == "smoke" ]]; then
        steps=100; eval_every=50; eval_tokens=4194304; core_every=-1; core_max=100
    else
        steps="$FULL_STEPS"; eval_every=250; eval_tokens="$EVAL_TOKENS"; core_every=1000; core_max=500
    fi
    local view
    if [[ "$arm" == "fuzzy" ]]; then view="$FUZZY_VIEW"; else view="$RAW_VIEW"; fi
    require_file "$view/view_manifest.json"
    local run_dir="$RUN_ROOT/$kind/$arm/seed_${seed}"
    local model_tag="d24-fwef-fuzzy-ab-${kind}-${arm}-seed${seed}"
    if [[ -f "$run_dir/run_summary.json" ]]; then
        echo "Skipping completed run: $run_dir"
        return
    fi
    if [[ -e "$run_dir/train.log" ]]; then
        echo "Incomplete run exists at $run_dir; inspect it instead of silently overwriting" >&2
        exit 1
    fi
    mkdir -p "$run_dir"
    "$PYTHON_BIN" -m scripts.compare_fortified_fuzzy_ab record-config         --run-dir "$run_dir" --arm "$arm" --seed "$seed" --kind "$kind"         --model-tag "$model_tag" --train-manifest "$view/view_manifest.json"         --validation-file "$VALIDATION_FILE" --depth 24 --num-iterations "$steps"         --total-batch-size "$TOTAL_BATCH_SIZE" --max-seq-len "$MAX_SEQ_LEN"         --device-batch-size "$DEVICE_BATCH_SIZE" --eval-every "$eval_every"         --eval-tokens "$eval_tokens" --core-metric-every "$core_every"         --core-metric-max-per-task "$core_max" --core-eval-seed 1337         --window-pattern L --fp8

    "$TORCHRUN_BIN" --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_train --         --depth=24 --num-iterations="$steps" --target-param-data-ratio=-1         --total-batch-size="$TOTAL_BATCH_SIZE" --max-seq-len="$MAX_SEQ_LEN"         --device-batch-size="$DEVICE_BATCH_SIZE" --window-pattern=L --fp8         --eval-every="$eval_every" --eval-tokens="$eval_tokens"         --core-metric-every="$core_every" --core-metric-max-per-task="$core_max"         --core-eval-seed=1337 --sample-every=-1 --save-every=-1         --run="fwef-fuzzy-ab-${kind}-${arm}-seed${seed}" --seed="$seed"         --model-tag="$model_tag" --data-source=parquet         --train-data-dir="$view" --val-data-dir="$VALIDATION_FILE"         --fail-on-data-epoch-rollover 2>&1 | tee "$run_dir/train.log"

    "$TORCHRUN_BIN" --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_eval --         --eval=core,bpb --max-per-task=-1 --split-tokens="$EVAL_TOKENS"         --device-batch-size="$DEVICE_BATCH_SIZE" --seed="$seed" --core-eval-seed=1337         --model-tag="$model_tag" --data-source=parquet         --train-data-dir="$view" --val-data-dir="$VALIDATION_FILE"         2>&1 | tee "$run_dir/base_eval.log"

    local core_csv="$NANOCHAT_BASE_DIR/base_eval/base_model_$(printf '%06d' "$steps").csv"
    require_file "$core_csv"
    cp "$core_csv" "$run_dir/base_eval_core.csv"
    "$PYTHON_BIN" -m scripts.compare_fortified_fuzzy_ab summarize-run --run-dir "$run_dir"
}

require_fuzzy_full() {
    for seed in "${SEEDS[@]}"; do
        require_file "$RUN_ROOT/full/fuzzy/seed_${seed}/run_summary.json"
    done
}

make_pioneer_report() {
    require_file "$RUN_ROOT/full/fuzzy/seed_42/run_summary.json"
    "$PYTHON_BIN" -m scripts.generate_fortified_fuzzy_pioneer         --run-root "$RUN_ROOT" --output-dir "$PIONEER_REPORT_DIR"         --fuzzy-manifest "$FUZZY_VIEW/view_manifest.json"         --dedup-manifest "$SOURCE_ROOT/work/exact_fuzzy_manifest.json"         --component-audit "$AUDIT_ROOT/component_audit.json"         --pair-audit "$AUDIT_ROOT/removed_to_keeper_pairs.jsonl" --seed 42
}

run_arm() {
    local arm="$1"
    for seed in "${SEEDS[@]}"; do
        run_one full "$arm" "$seed"
        if [[ "$arm" == "fuzzy" && "$seed" == "42" ]]; then
            make_pioneer_report
        fi
    done
}

make_report() {
    require_file "$FUZZY_VIEW/view_manifest.json"
    require_file "$RAW_VIEW/view_manifest.json"
    "$PYTHON_BIN" -m scripts.compare_fortified_fuzzy_ab report         --run-root "$RUN_ROOT" --output-dir "$REPORT_DIR"         --fuzzy-manifest "$FUZZY_VIEW/view_manifest.json"         --raw-manifest "$RAW_VIEW/view_manifest.json"         --dedup-manifest "$SOURCE_ROOT/work/exact_fuzzy_manifest.json"         --component-audit "$AUDIT_ROOT/component_audit.json"         --pair-audit "$AUDIT_ROOT/removed_to_keeper_pairs.jsonl"
}

status() {
    for arm in fuzzy raw; do
        local view="$FUZZY_VIEW"
        [[ "$arm" == "raw" ]] && view="$RAW_VIEW"
        if [[ -f "$view/view_manifest.json" ]]; then
            "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(sys.argv[2], p.get("status"), p.get("selection",{}).get("docs"), p.get("selection",{}).get("tokens"))' "$view/view_manifest.json" "$arm-view"
        else
            echo "$arm-view missing"
        fi
        for seed in "${SEEDS[@]}"; do
            if [[ -f "$RUN_ROOT/full/$arm/seed_${seed}/run_summary.json" ]]; then echo "$arm seed=$seed complete"; else echo "$arm seed=$seed pending"; fi
        done
    done
    [[ -f "$PIONEER_REPORT_DIR/pioneer_report.md" ]] && echo "pioneer report complete: $PIONEER_REPORT_DIR/pioneer_report.md" || echo "pioneer report pending"
    [[ -f "$REPORT_DIR/final_report.md" ]] && echo "report complete: $REPORT_DIR/final_report.md" || echo "report pending"
}

mkdir -p "$EXPERIMENT_ROOT" "$RUN_ROOT"
case "$MODE" in
    prepare-fuzzy) preflight; prepare_view fuzzy ;;
    smoke-fuzzy) preflight; run_one smoke fuzzy 42 ;;
    run-fuzzy) preflight; run_arm fuzzy ;;
    pioneer-report) make_pioneer_report ;;
    prepare-raw) preflight; require_fuzzy_full; prepare_view raw ;;
    smoke-raw) preflight; require_fuzzy_full; run_one smoke raw 42 ;;
    run-raw) preflight; require_fuzzy_full; run_arm raw ;;
    report) make_report ;;
    status) status ;;
    all)
        preflight
        prepare_view fuzzy
        run_one smoke fuzzy 42
        run_arm fuzzy
        require_fuzzy_full
        prepare_view raw
        run_one smoke raw 42
        run_arm raw
        make_report
        ;;
    *)
        echo "Usage: $0 {prepare-fuzzy|smoke-fuzzy|run-fuzzy|pioneer-report|prepare-raw|smoke-raw|run-raw|report|status|all}" >&2
        exit 2
        ;;
esac
