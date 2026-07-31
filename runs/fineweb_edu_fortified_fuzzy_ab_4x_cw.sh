#!/usr/bin/env bash
set -euo pipefail

# d24 long-horizon 4x-token, one-seed paired raw/fuzzy A/B.
# Data preparation runs beside the canonical corpus; GPU modes run on CW H100 nodes.

MODE="${1:-status}"
ARM="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$REPO_ROOT/.venv/bin/torchrun}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

SOURCE_ROOT="${SOURCE_ROOT:-/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu_fortified_exact_fuzzy}"
RAW_SOURCE="${RAW_SOURCE:-$SOURCE_ROOT/raw}"
FUZZY_SOURCE="${FUZZY_SOURCE:-$SOURCE_ROOT/post_fuzzy}"
SOURCE_DEDUP_MANIFEST="${SOURCE_DEDUP_MANIFEST:-$SOURCE_ROOT/work/exact_fuzzy_manifest.json}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu}"
export NANOCHAT_BASE_DIR
VALIDATION_FILE="${VALIDATION_FILE:-$NANOCHAT_BASE_DIR/base_data_fineweb_edu/shard_01822.parquet}"
VALIDATION_SHA256="${VALIDATION_SHA256:-a97e7f2f4680561c1fdf0461295b0cfc38d85f497ba03851ab267ac1b8165990}"
TOKENIZER_FILE="${TOKENIZER_FILE:-$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$NANOCHAT_BASE_DIR/tokenizer}"
CORE_BUNDLE="${CORE_BUNDLE:-$NANOCHAT_BASE_DIR/eval_bundle}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/home/nfs/hfang/.cache/nanochat_cw_fwef_fuzzy_ab_4x}"
DATA_ROOT="${DATA_ROOT:-$EXPERIMENT_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$EXPERIMENT_ROOT/runs}"
REPORT_DIR="${REPORT_DIR:-$EXPERIMENT_ROOT/report}"
LOG_ROOT="${LOG_ROOT:-$EXPERIMENT_ROOT/logs}"
EVAL_ROOT="$DATA_ROOT/eval_queries"
CONTAMINATION_ROOT="$DATA_ROOT/contamination"
CONTAMINATION_MANIFEST="$CONTAMINATION_ROOT/manifest/contamination_manifest.json"
CONTAMINATION_ARTIFACT_ROOT="${CONTAMINATION_ARTIFACT_ROOT:-/tmp/${USER}_nanochat_fwef_4x_contamination_artifacts}"
ACTIVE_HASH_FILE="$DATA_ROOT/active_hash_end.txt"
PREFLIGHT_MANIFEST="${PREFLIGHT_MANIFEST:-$EXPERIMENT_ROOT/preflight.json}"
FROZEN_DEDUP_MANIFEST="$DATA_ROOT/source/exact_fuzzy_manifest.json"

NUM_GPUS=8
MODEL_SEED=42
FULL_STEPS=26448
TOTAL_BATCH_SIZE=1048576
TRAINING_TOKENS=27732738048
TARGET_CAPACITY=27733786624
MAX_SEQ_LEN=2048
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
EVAL_EVERY=1102
EVAL_TOKENS=41943040
SMOKE_EVAL_TOKENS=4194304
CORE_EVERY=6612
SAVE_EVERY=1102
PRESERVE_STEPS="6612,13224,19836,26448"
SELECTION_SEED=20260722
ORDER_SEED=20260723
CORE_SEED=1337
VIEW_WORKERS="${VIEW_WORKERS:-32}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"

cd "$REPO_ROOT"

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "Required file is missing: $1"; }
require_dir() { [[ -d "$1" ]] || die "Required directory is missing: $1"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
with_log() { local log="$1"; shift; mkdir -p "$(dirname "$log")"; "$@" 2>&1 | tee "$log"; }
validate_arm() { [[ "$1" == raw || "$1" == fuzzy ]] || die "arm must be raw or fuzzy"; }
hash_label() { printf '%s' "$1" | tr '.' '_'; }

validate_static_inputs() {
    require_file "$PYTHON_BIN"
    require_file "$TORCHRUN_BIN"
    require_file "$VALIDATION_FILE"
    require_file "$TOKENIZER_FILE"
    require_dir "$CORE_BUNDLE"
    local actual
    actual="$(sha256_file "$VALIDATION_FILE")"
    [[ "$actual" == "$VALIDATION_SHA256" ]] || die "Frozen validation checksum changed: $actual"
}

active_view_root() {
    if [[ -n "${VIEW_ROOT:-}" ]]; then printf '%s\n' "$VIEW_ROOT"; return; fi
    require_file "$ACTIVE_HASH_FILE"
    local hash_end
    hash_end="$(tr -d '[:space:]' < "$ACTIVE_HASH_FILE")"
    printf '%s/views/hash_%s\n' "$DATA_ROOT" "$(hash_label "$hash_end")"
}

prepare_eval_and_contamination() {
    require_dir "$RAW_SOURCE"
    require_dir "$FUZZY_SOURCE"
    require_file "$SOURCE_DEDUP_MANIFEST"
    mkdir -p "$DATA_ROOT/source" "$LOG_ROOT"
    cp -p "$SOURCE_DEDUP_MANIFEST" "$FROZEN_DEDUP_MANIFEST"
    "$PYTHON_BIN" -m scripts.export_eval_contamination_corpus \
        --validation-file "$VALIDATION_FILE" --core-bundle "$CORE_BUNDLE" --output-dir "$EVAL_ROOT"
    "$PYTHON_BIN" -m scripts.prepare_contamination_union \
        --training-root "$RAW_SOURCE" --eval-root "$EVAL_ROOT" \
        --output-dir "$CONTAMINATION_ROOT/union" --link-mode hardlink
    "$PYTHON_BIN" -m scripts.build_fuzzy_audit_artifacts \
        --input-data-dir "$CONTAMINATION_ROOT/union" \
        --work-dir "$CONTAMINATION_ARTIFACT_ROOT" \
        --seed 42 --char-ngrams 24 --num-bands 20 --minhashes-per-band 13 \
        --ray-num-cpus "$RAY_NUM_CPUS"
    with_log "$LOG_ROOT/contamination.log" "$PYTHON_BIN" -m scripts.build_contamination_manifest \
        --union-data-dir "$CONTAMINATION_ROOT/union" \
        --component-dir "$CONTAMINATION_ARTIFACT_ROOT/fuzzy_cache/ConnectedComponentsStage" \
        --id-generator-path "$CONTAMINATION_ARTIFACT_ROOT/fuzzy_identification/fuzzy_id_generator.json" \
        --fuzzy-artifacts-manifest "$CONTAMINATION_ARTIFACT_ROOT/fuzzy_audit_artifacts_manifest.json" \
        --eval-manifest "$EVAL_ROOT/eval_contamination_manifest.json" \
        --output-dir "$CONTAMINATION_ROOT/manifest"
}

build_one_hash_view() {
    local hash_end="$1" root="$2" arm="$3"
    local source="$FUZZY_SOURCE"
    local extra=()
    if [[ "$arm" == raw ]]; then
        source="$RAW_SOURCE"
        extra=(--fuzzy-view-dir "$root/fuzzy" --skip-removal-fraction-gate)
    fi
    with_log "$LOG_ROOT/view_hash_$(hash_label "$hash_end")_${arm}.log" \
        "$PYTHON_BIN" -m scripts.build_fortified_hash_view \
        --arm "$arm" --source-root "$source" --output-dir "$root/$arm" \
        --validation-file "$VALIDATION_FILE" --contamination-manifest "$CONTAMINATION_MANIFEST" \
        --selection-seed "$SELECTION_SEED" --order-seed "$ORDER_SEED" \
        --hash-start 0 --hash-end "$hash_end" --num-buckets 512 \
        --buffer-rows 2000000 --workers "$VIEW_WORKERS" \
        --target-token-capacity "$TARGET_CAPACITY" \
        --capacity-world-size "$NUM_GPUS" --capacity-seq-len "$MAX_SEQ_LEN" \
        "${extra[@]}"
}

prepare_views() {
    require_file "$CONTAMINATION_MANIFEST"
    if [[ -f "$ACTIVE_HASH_FILE" ]]; then
        local existing
        existing="$(active_view_root)"
        require_file "$existing/raw/view_manifest.json"
        require_file "$existing/fuzzy/view_manifest.json"
        echo "Frozen views already selected: $existing"
        return
    fi
    local hash_end label root log rc archived
    for hash_end in 0.16 0.18 0.20; do
        label="$(hash_label "$hash_end")"
        root="$DATA_ROOT/views/hash_$label"
        log="$LOG_ROOT/view_hash_${label}_fuzzy.log"
        set +e
        build_one_hash_view "$hash_end" "$root" fuzzy
        rc=$?
        set -e
        if [[ "$rc" -eq 0 ]]; then
            build_one_hash_view "$hash_end" "$root" raw
            printf '%s\n' "$hash_end" > "$ACTIVE_HASH_FILE"
            echo "Selected hash interval [0,$hash_end): $root"
            return
        fi
        if grep -q "target tokens without rollover" "$log"; then
            archived="${root}_insufficient_$(date -u +%Y%m%dT%H%M%SZ)"
            mv "$root" "$archived"
            echo "Hash end $hash_end lacked capacity; preserved incomplete view at $archived"
            continue
        fi
        die "Fuzzy view build failed for hash end $hash_end"
    done
    die "No allowed hash interval supplied sufficient fuzzy capacity"
}

run_preflight() {
    validate_static_inputs
    local view hash_end spec
    view="$(active_view_root)"
    hash_end="$(tr -d '[:space:]' < "$ACTIVE_HASH_FILE")"
    require_file "$FROZEN_DEDUP_MANIFEST"
    spec="cw_4x|-|$view/raw|$view/fuzzy|$FROZEN_DEDUP_MANIFEST|$CONTAMINATION_MANIFEST"
    "$PYTHON_BIN" -m scripts.preflight_cc_snapshot_experiment \
        --condition "$spec" --expected-condition cw_4x \
        --num-iterations "$FULL_STEPS" --total-batch-size "$TOTAL_BATCH_SIZE" \
        --world-size "$NUM_GPUS" --sequence-length "$MAX_SEQ_LEN" \
        --target-token-capacity "$TARGET_CAPACITY" \
        --selection-seed "$SELECTION_SEED" --order-seed "$ORDER_SEED" \
        --expected-hash-start 0 --expected-hash-end "$hash_end" \
        --validation-file "$VALIDATION_FILE" --validation-sha256 "$VALIDATION_SHA256" \
        --tokenizer-file "$TOKENIZER_FILE" --output "$PREFLIGHT_MANIFEST"
}

assert_preflight() {
    require_file "$PREFLIGHT_MANIFEST"
    "$PYTHON_BIN" - "$PREFLIGHT_MANIFEST" <<'PY'
import json, sys
from scripts.experiment_fingerprints import training_code_sha256
p = json.load(open(sys.argv[1]))
assert p["status"] == "passed"
assert p["steps"] == 26448 and p["training_tokens"] == 27732738048
assert p["target_capacity"] == 27733786624 and p["world_size"] == 8
assert p["code_sha256"] == training_code_sha256(), "training code changed after preflight"
assert len(p["views"]) == 2
PY
}

gpu_preflight() {
    [[ -z "${NCCL_NVLS_ENABLE+x}" || "${ALLOW_NCCL_NVLS_OVERRIDE:-0}" == 1 ]] || \
        die "NCCL_NVLS_ENABLE must be unset on CW unless ALLOW_NCCL_NVLS_OVERRIDE=1"
    local names count
    names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
    count="$(wc -l <<< "$names")"
    [[ "$count" -eq "$NUM_GPUS" ]] || die "Expected $NUM_GPUS GPUs, found $count"
    while IFS= read -r name; do [[ "$name" == *H100* ]] || die "Expected H100, found $name"; done <<< "$names"
}

capture_environment() {
    local output="$1"
    mkdir -p "$(dirname "$output")"
    "$PYTHON_BIN" -m scripts.capture_experiment_environment \
        --output "$output" --expected-gpus "$NUM_GPUS" --expected-gpu-name H100 \
        --require-fa3 --shared-path "$EXPERIMENT_ROOT" --min-shared-free-gb 500
}

train() {
    local arm="$1" tag="$2" eval_every="$3" eval_tokens="$4" core_every="$5" core_max="$6"
    shift 6
    local view
    view="$(active_view_root)"
    "$TORCHRUN_BIN" --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_train -- \
        --depth=24 --num-iterations="$FULL_STEPS" --target-param-data-ratio=-1 \
        --total-batch-size="$TOTAL_BATCH_SIZE" --max-seq-len="$MAX_SEQ_LEN" \
        --device-batch-size="$DEVICE_BATCH_SIZE" --window-pattern=L \
        --fp8 --fp8-recipe=tensorwise --require-fa3 \
        --eval-every="$eval_every" --eval-tokens="$eval_tokens" \
        --core-metric-every="$core_every" --core-metric-max-per-task="$core_max" \
        --core-eval-seed="$CORE_SEED" --sample-every=-1 \
        --run="$tag" --seed="$MODEL_SEED" --model-tag="$tag" --data-source=parquet \
        --train-data-dir="$view/$arm" --val-data-dir="$VALIDATION_FILE" \
        --fail-on-data-epoch-rollover --require-zero-discarded-tokens "$@"
}

run_resume_smoke() {
    local arm="$1" root="$RUN_ROOT/smoke/$arm/seed_42/resume"
    local ref="$root/uninterrupted" cut="$root/resumed"
    local ref_tag="d24-fwef-cw4x-smoke-ref-${arm}-seed42"
    local cut_tag="d24-fwef-cw4x-smoke-cut-${arm}-seed42"
    if [[ -f "$root/resume_verification.json" ]]; then echo "Resume smoke already passed: $arm"; return; fi
    [[ ! -e "$ref/reference.log" && ! -e "$cut/interrupted.log" ]] || die "Incomplete resume smoke exists: $root"
    [[ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/$ref_tag" ]] || die "Checkpoint tag exists: $ref_tag"
    [[ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/$cut_tag" ]] || die "Checkpoint tag exists: $cut_tag"
    mkdir -p "$ref" "$cut"
    train "$arm" "$ref_tag" 256 "$SMOKE_EVAL_TOKENS" -1 100 \
        --save-every=-1 --stop-after-step=256 --audit-batch-steps=128 \
        2>&1 | tee "$ref/reference.log"
    train "$arm" "$cut_tag" 256 "$SMOKE_EVAL_TOKENS" -1 100 \
        --save-every=-1 --stop-after-step=128 --audit-batch-steps=128 \
        2>&1 | tee "$cut/interrupted.log"
    train "$arm" "$cut_tag" 256 "$SMOKE_EVAL_TOKENS" -1 100 \
        --save-every=-1 --stop-after-step=256 --resume-from-step=128 --audit-batch-steps=128 \
        2>&1 | tee "$cut/resumed.log"
    "$PYTHON_BIN" -m scripts.verify_resume_equivalence \
        --reference-checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/$ref_tag" \
        --resumed-checkpoint-dir "$NANOCHAT_BASE_DIR/base_checkpoints/$cut_tag" \
        --reference-log "$ref/reference.log" --resumed-log "$cut/resumed.log" \
        --output "$root/resume_verification.json" --step 256 --world-size 8 \
        --required-batch-step 128 --parameter-tolerance 1e-6
}

run_performance_smoke() {
    local arm="$1" root="$RUN_ROOT/smoke/$arm/seed_42/performance"
    local tag="d24-fwef-cw4x-smoke-perf-${arm}-seed42"
    if [[ -f "$root/smoke_gate.json" ]]; then echo "Performance smoke already passed: $arm"; return; fi
    [[ ! -e "$root/train.log" ]] || die "Incomplete performance smoke exists: $root"
    [[ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/$tag" ]] || die "Checkpoint tag exists: $tag"
    mkdir -p "$root"
    train "$arm" "$tag" 500 "$SMOKE_EVAL_TOKENS" -1 100 \
        --save-every=-1 --stop-after-step=500 2>&1 | tee "$root/train.log"
    "$PYTHON_BIN" -m scripts.check_fuzzy_ab_smoke \
        --log "$root/train.log" --output "$root/smoke_gate.json" \
        --full-training-tokens "$TRAINING_TOKENS" --min-tokens-per-sec 200000 \
        --max-data-wait-fraction 0.10 --max-projected-hours 42 --overhead-fraction 0.10
}

run_smoke_one() {
    local arm="$1" root="$RUN_ROOT/smoke/$arm/seed_42"
    validate_arm "$arm"; assert_preflight; gpu_preflight
    mkdir -p "$root"
    capture_environment "$root/environment_current.json"
    run_resume_smoke "$arm"
    run_performance_smoke "$arm"
}

record_full_config() {
    local arm="$1" run_dir="$2" tag="$3" environment="$4" view
    view="$(active_view_root)"
    "$PYTHON_BIN" -m scripts.compare_fortified_fuzzy_ab record-config \
        --run-dir "$run_dir" --condition cw_4x --arm "$arm" --seed "$MODEL_SEED" \
        --kind full --model-tag "$tag" --train-manifest "$view/$arm/view_manifest.json" \
        --validation-file "$VALIDATION_FILE" --tokenizer-file "$TOKENIZER_FILE" \
        --core-config-file "$CORE_BUNDLE/core.yaml" --preflight-manifest "$PREFLIGHT_MANIFEST" \
        --environment-file "$environment" --depth 24 --num-iterations "$FULL_STEPS" \
        --total-batch-size "$TOTAL_BATCH_SIZE" --max-seq-len "$MAX_SEQ_LEN" \
        --device-batch-size "$DEVICE_BATCH_SIZE" --eval-every "$EVAL_EVERY" \
        --eval-tokens "$EVAL_TOKENS" --core-metric-every "$CORE_EVERY" \
        --core-metric-max-per-task 500 --core-eval-seed "$CORE_SEED" \
        --window-pattern L --fp8 --require-fa3 --audit-val-bpb-steps "$FULL_STEPS" \
        --preserve-checkpoint-steps "$PRESERVE_STEPS"
}

final_eval_and_summary() {
    local arm="$1" run_dir="$2" tag="$3" view
    view="$(active_view_root)"
    if [[ ! -f "$run_dir/base_eval_core.csv" ]]; then
        "$TORCHRUN_BIN" --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_eval -- \
            --eval=core,bpb --max-per-task=-1 --split-tokens="$EVAL_TOKENS" \
            --device-batch-size="$DEVICE_BATCH_SIZE" --seed="$MODEL_SEED" \
            --core-eval-seed="$CORE_SEED" --model-tag="$tag" --data-source=parquet \
            --train-data-dir="$view/$arm" --val-data-dir="$VALIDATION_FILE" \
            --output-csv="$run_dir/base_eval_core.csv" 2>&1 | tee "$run_dir/base_eval.log"
    fi
    "$PYTHON_BIN" -m scripts.compare_fortified_fuzzy_ab summarize-run --run-dir "$run_dir"
}

run_full_one() {
    local arm="$1" run_dir="$RUN_ROOT/full/$arm/seed_42"
    local tag="d24-fwef-cw4x-full-${arm}-seed42"
    validate_arm "$arm"; assert_preflight; gpu_preflight
    [[ ! -f "$run_dir/run_summary.json" ]] || { echo "Full run already complete: $arm"; return; }
    [[ ! -e "$run_dir/train.log" ]] || die "Incomplete full run exists; use resume-one $arm"
    [[ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/$tag" ]] || die "Checkpoint tag exists: $tag"
    mkdir -p "$run_dir/attempts"
    local environment="$run_dir/attempts/environment_${SLURM_JOB_ID:-manual}_$(hostname).json"
    capture_environment "$environment"
    cp -p "$environment" "$run_dir/environment_current.json"
    record_full_config "$arm" "$run_dir" "$tag" "$environment"
    train "$arm" "$tag" "$EVAL_EVERY" "$EVAL_TOKENS" "$CORE_EVERY" 500 \
        --save-every="$SAVE_EVERY" --keep-last-checkpoints=2 \
        --preserve-checkpoint-steps="$PRESERVE_STEPS" --audit-val-bpb-steps="$FULL_STEPS" \
        2>&1 | tee "$run_dir/train.log"
    final_eval_and_summary "$arm" "$run_dir" "$tag"
}

latest_complete_step() {
    local checkpoint_dir="$1" model step rank complete
    for model in $(find "$checkpoint_dir" -maxdepth 1 -type f -name 'model_*.pt' | sort -r); do
        step="${model##*_}"; step="${step%.pt}"; complete=1
        [[ -f "$checkpoint_dir/meta_${step}.json" ]] || complete=0
        for rank in $(seq 0 7); do
            [[ -f "$checkpoint_dir/meta_${step}_rank${rank}.json" ]] || complete=0
            [[ -f "$checkpoint_dir/optim_${step}_rank${rank}.pt" ]] || complete=0
        done
        if [[ "$complete" -eq 1 ]]; then printf '%d\n' "$((10#$step))"; return; fi
    done
    return 1
}

run_resume_one() {
    local arm="$1" run_dir="$RUN_ROOT/full/$arm/seed_42"
    local tag="d24-fwef-cw4x-full-${arm}-seed42"
    local checkpoint_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$tag"
    validate_arm "$arm"; assert_preflight; gpu_preflight
    [[ ! -f "$run_dir/run_summary.json" ]] || { echo "Full run already complete: $arm"; return; }
    require_file "$run_dir/train.log"; require_dir "$checkpoint_dir"
    local step environment
    step="$(latest_complete_step "$checkpoint_dir")" || die "No complete distributed checkpoint: $checkpoint_dir"
    [[ "$step" -le "$FULL_STEPS" ]] || die "Checkpoint step exceeds horizon: $step"
    mkdir -p "$run_dir/attempts"
    environment="$run_dir/attempts/environment_${SLURM_JOB_ID:-manual}_$(hostname).json"
    capture_environment "$environment"
    cp -p "$environment" "$run_dir/environment_current.json"
    record_full_config "$arm" "$run_dir" "$tag" "$environment"
    if [[ "$step" -lt "$FULL_STEPS" ]]; then
        train "$arm" "$tag" "$EVAL_EVERY" "$EVAL_TOKENS" "$CORE_EVERY" 500 \
            --resume-from-step="$step" --save-every="$SAVE_EVERY" --keep-last-checkpoints=2 \
            --preserve-checkpoint-steps="$PRESERVE_STEPS" --audit-val-bpb-steps="$FULL_STEPS" \
            2>&1 | tee -a "$run_dir/train.log"
    fi
    final_eval_and_summary "$arm" "$run_dir" "$tag"
}

compare_node_environments() {
    local kind="$1" raw_env fuzzy_env
    raw_env="$RUN_ROOT/$kind/raw/seed_42/environment_current.json"
    fuzzy_env="$RUN_ROOT/$kind/fuzzy/seed_42/environment_current.json"
    require_file "$raw_env"; require_file "$fuzzy_env"
    "$PYTHON_BIN" - "$raw_env" "$fuzzy_env" <<'PY'
import json, sys
left, right = (json.load(open(path)) for path in sys.argv[1:])
assert left["compatibility_sha256"] == right["compatibility_sha256"], "paired nodes have different environment fingerprints"
print(left["compatibility_sha256"])
PY
}

run_paired_slurm() {
    local kind="$1"
    [[ "$kind" == smoke || "$kind" == full ]] || die "paired-slurm requires smoke or full"
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "paired-slurm must run inside a Slurm allocation"
    mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    [[ "${#hosts[@]}" -eq 2 ]] || die "Expected exactly two allocated nodes"
    local submode="${kind}-one" raw_rc=0 fuzzy_rc=0
    srun --nodes=1 --ntasks=1 --exclusive --nodelist="${hosts[0]}" \
        bash "$REPO_ROOT/runs/fineweb_edu_fortified_fuzzy_ab_4x_cw.sh" "$submode" raw &
    local raw_pid=$!
    srun --nodes=1 --ntasks=1 --exclusive --nodelist="${hosts[1]}" \
        bash "$REPO_ROOT/runs/fineweb_edu_fortified_fuzzy_ab_4x_cw.sh" "$submode" fuzzy &
    local fuzzy_pid=$!
    wait "$raw_pid" || raw_rc=$?
    wait "$fuzzy_pid" || fuzzy_rc=$?
    [[ "$raw_rc" -eq 0 && "$fuzzy_rc" -eq 0 ]] || \
        die "Paired $kind failed: raw_rc=$raw_rc fuzzy_rc=$fuzzy_rc"
    compare_node_environments "$kind"
}

generate_report() {
    local view
    view="$(active_view_root)"
    "$PYTHON_BIN" -m scripts.compare_fortified_fuzzy_ab report \
        --run-root "$RUN_ROOT" --output-dir "$REPORT_DIR" \
        --fuzzy-manifest "$view/fuzzy/view_manifest.json" \
        --raw-manifest "$view/raw/view_manifest.json" \
        --dedup-manifest "$FROZEN_DEDUP_MANIFEST" \
        --component-audit "$SOURCE_ROOT/work/fuzzy_component_audit.json" \
        --pair-audit "$SOURCE_ROOT/work/fuzzy_pair_audit.jsonl" \
        --seed 42 \
        --report-title "FineWeb-EDU-Fortified d24 4x-token long-horizon one-seed A/B on CW H100"
}

show_status() {
    echo "repo=$REPO_ROOT"
    echo "commit=$(git rev-parse HEAD)"
    echo "experiment_root=$EXPERIMENT_ROOT"
    if [[ -f "$ACTIVE_HASH_FILE" ]]; then echo "view_root=$(active_view_root)"; else echo "view_root=not-prepared"; fi
    if [[ -f "$PREFLIGHT_MANIFEST" ]]; then echo "preflight=present"; else echo "preflight=missing"; fi
    for arm in raw fuzzy; do
        if [[ -f "$RUN_ROOT/full/$arm/seed_42/run_summary.json" ]]; then
            echo "$arm=complete"
        elif [[ -f "$RUN_ROOT/full/$arm/seed_42/train.log" ]]; then
            echo "$arm=incomplete-resumable"
        else
            echo "$arm=not-started"
        fi
    done
}

case "$MODE" in
    prepare)
        validate_static_inputs
        mkdir -p "$EXPERIMENT_ROOT"
        free_kb="$(df -Pk "$EXPERIMENT_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
        if [[ -n "$free_kb" && "$free_kb" -lt $((350 * 1024 * 1024)) ]]; then
            die "At least 350 GiB free is required before view materialization"
        fi
        prepare_eval_and_contamination
        prepare_views
        run_preflight
        ;;
    preflight) run_preflight ;;
    smoke-one) validate_static_inputs; run_smoke_one "$ARM" ;;
    full-one) validate_static_inputs; run_full_one "$ARM" ;;
    resume-one) validate_static_inputs; run_resume_one "$ARM" ;;
    paired-slurm) validate_static_inputs; assert_preflight; run_paired_slurm "$ARM" ;;
    smoke) validate_static_inputs; assert_preflight; run_smoke_one raw; run_smoke_one fuzzy; compare_node_environments smoke ;;
    full) validate_static_inputs; assert_preflight; run_full_one raw; run_full_one fuzzy; compare_node_environments full ;;
    resume) validate_static_inputs; assert_preflight; run_resume_one "$ARM" ;;
    report) generate_report ;;
    status) show_status ;;
    *) die "Usage: $0 {prepare|preflight|smoke|full|resume ARM|status|report|smoke-one ARM|full-one ARM|resume-one ARM|paired-slurm smoke|full}" ;;
esac
