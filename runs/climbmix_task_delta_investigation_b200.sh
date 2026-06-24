#!/bin/bash

# Full-heavy investigation for ClimbMix SemDeDup task deltas.
# Launch this script with nohup for long stages, for example:
#   nohup bash runs/climbmix_task_delta_investigation_b200.sh all > all.log 2>&1 &

set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: $0 smoke|eval-only|removed-audit|full-seed-repeats|randomdrop-control|eps-sweep|report|all" >&2
    exit 2
fi

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_climbmix}"
export TASK_DELTA_ROOT="${TASK_DELTA_ROOT:-$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_task_delta}"
export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$TASK_DELTA_ROOT/logs"
PID_DIR="$TASK_DELTA_ROOT/pids"
DATA_DIR="$TASK_DELTA_ROOT/data"
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
export NUM_ITERATIONS="${NUM_ITERATIONS:-6612}"
export PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-9.5}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-2000}"
export CORE_METRIC_MAX_PER_TASK="${CORE_METRIC_MAX_PER_TASK:-500}"
export FINAL_CORE_MAX_PER_TASK="${FINAL_CORE_MAX_PER_TASK:--1}"
export BASE_EVAL_MODES="${BASE_EVAL_MODES:-core,bpb,sample}"
export SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}"
export PREP_DATASET="${PREP_DATASET:-0}"
export INPUT_DATA_DIR="${INPUT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_climbmix}"
export AUDIT_SAMPLE_SIZE="${AUDIT_SAMPLE_SIZE:-1000}"
export AUDIT_SEED="${AUDIT_SEED:-1337}"
export CORE_EVAL_SEED="${CORE_EVAL_SEED:-1337}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_MAX_LIMIT_FROM_API_SERVER="${RAY_MAX_LIMIT_FROM_API_SERVER:-100000}"

QUALITY_ROOT="${QUALITY_ROOT:-$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality}"
EXISTING_BASELINE_RUN_DIR="${EXISTING_BASELINE_RUN_DIR:-$QUALITY_ROOT/full-n170-r9p5-eps0p07-20260611T163000Z_baseline}"
EXISTING_SEMDEDUP_RUN_DIR="${EXISTING_SEMDEDUP_RUN_DIR:-$QUALITY_ROOT/full-n170-r9p5-eps0p07-20260612T080000Z_semdedup}"
EXISTING_BASELINE_MODEL_TAG="${EXISTING_BASELINE_MODEL_TAG:-d24-climbmix-nosd-n170-20260611T171735Z}"
EXISTING_SEMDEDUP_MODEL_TAG="${EXISTING_SEMDEDUP_MODEL_TAG:-d24-climbmix-semdedup-eps0p07-n170-20260612T172022Z}"
EXISTING_SEMDEDUP_DATA_DIR="${EXISTING_SEMDEDUP_DATA_DIR:-$EXISTING_SEMDEDUP_RUN_DIR/base_data_climbmix_semdedup_eps0p07_n170}"
RANDOM_DROP_REMOVED_DOCS="${RANDOM_DROP_REMOVED_DOCS:-291374}"
RANDOM_DROP_SEED_PRIMARY="${RANDOM_DROP_SEED_PRIMARY:-9001}"
RANDOM_DROP_SEED_SENSITIVITY="${RANDOM_DROP_SEED_SENSITIVITY:-9002}"
EPS_SWEEP_VALUES="${EPS_SWEEP_VALUES:-0.05 0.09 0.12}"
SEMD_MODEL="${SEMD_MODEL:-google/embeddinggemma-300m}"
SEMD_N_CLUSTERS="${SEMD_N_CLUSTERS:-100}"
SEMD_DISTANCE_METRIC="${SEMD_DISTANCE_METRIC:-cosine}"
SEMD_WHICH_TO_KEEP="${SEMD_WHICH_TO_KEEP:-hard}"
SEMD_PAIRWISE_BATCH_SIZE="${SEMD_PAIRWISE_BATCH_SIZE:-1024}"
SEMD_VLLM_ATTENTION_BACKEND="${SEMD_VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
SEMD_VLLM_ENFORCE_EAGER="${SEMD_VLLM_ENFORCE_EAGER:-1}"
SEMD_NO_RAY_PREINIT="${SEMD_NO_RAY_PREINIT:-1}"
SEMD_RAY_TEMP_DIR="${SEMD_RAY_TEMP_DIR:-/tmp/ray_hfang_taskdelta}"
RAY_PORT="${RAY_PORT:-24017}"
START_RAY="${START_RAY:-1}"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

run_quality() (
    export RUN_ROOT RUN_ID RUN_TAG_BASELINE RUN_TAG_SEMDEDUP RUN_TAG_RANDOMDROP
    export WANDB_MODE WANDB_RUN WANDB_RUN_BASELINE WANDB_RUN_SEMDEDUP WANDB_RUN_RANDOMDROP
    export SEED CORE_EVAL_SEED PYTHON_BIN DO_TRAIN DO_EVAL PREP_DATASET SKIP_UV_SYNC
    export NUM_GPUS DEPTH DEVICE_BATCH_SIZE NUM_TRAIN_SHARDS NUM_ITERATIONS PARAM_DATA_RATIO
    export INPUT_DATA_DIR STATS_MAX_DOCS MAX_DOCS AUDIT_SAMPLE_SIZE AUDIT_SEED SKIP_TOKEN_STATS
    export SEMD_EPS SEMD_MODEL SEMD_N_CLUSTERS SEMD_DISTANCE_METRIC SEMD_WHICH_TO_KEEP
    export SEMD_PAIRWISE_BATCH_SIZE SEMD_OUTPUT_DIR SEMD_OVERWRITE DO_SEMDEDUP
    export SEMD_VLLM_ATTENTION_BACKEND SEMD_VLLM_ENFORCE_EAGER SEMD_NO_RAY_PREINIT SEMD_RAY_TEMP_DIR
    export RANDOM_DROP_OUTPUT_DIR RANDOM_DROP_REMOVED_DOCS RANDOM_DROP_SEED RANDOM_DROP_OVERWRITE DO_RANDOM_DROP
    bash runs/climbmix_semdedup_quality_b200.sh "$@"
)

start_ray_if_needed() {
    if [[ "$START_RAY" != "1" ]]; then
        return
    fi
    if ! command -v ray >/dev/null 2>&1; then
        log "ray command not found; assuming Curator will manage Ray or fail clearly."
        return
    fi
    log "Starting isolated Ray head at 127.0.0.1:$RAY_PORT temp=$SEMD_RAY_TEMP_DIR"
    ray stop --force || true
    rm -rf "$SEMD_RAY_TEMP_DIR"
    mkdir -p "$SEMD_RAY_TEMP_DIR"
    ray start --head \
        --node-ip-address=127.0.0.1 \
        --port="$RAY_PORT" \
        --dashboard-host=127.0.0.1 \
        --temp-dir="$SEMD_RAY_TEMP_DIR"
    export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
    export RAY_TMPDIR="$SEMD_RAY_TEMP_DIR"
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
}

write_eval_summary() {
    local out_dir="$1"
    local arm="$2"
    local repeat="$3"
    local model_tag="$4"
    /usr/bin/python3 - "$out_dir" "$arm" "$repeat" "$model_tag" <<'PYEVAL'
import csv
import json
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
arm = sys.argv[2]
repeat = int(sys.argv[3])
model_tag = sys.argv[4]
log_text = (out_dir / "base_eval.log").read_text(errors="replace") if (out_dir / "base_eval.log").exists() else ""
core = None
matches = re.findall(r"CORE metric:\s+([0-9.]+)", log_text)
if matches:
    core = float(matches[-1])
train_bpb = None
val_bpb = None
m = re.findall(r"^train bpb:\s+([0-9.]+)", log_text, flags=re.MULTILINE)
if m:
    train_bpb = float(m[-1])
m = re.findall(r"^val bpb:\s+([0-9.]+)", log_text, flags=re.MULTILINE)
if m:
    val_bpb = float(m[-1])
per_task = {}
csv_path = out_dir / "base_eval_core.csv"
if csv_path.exists():
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            task = row[0].strip()
            acc = row[1].strip() if len(row) > 1 else ""
            centered = row[2].strip() if len(row) > 2 else ""
            per_task[task] = {
                "accuracy": float(acc) if acc else None,
                "centered": float(centered) if centered else None,
            }
summary = {
    "arm": arm,
    "repeat": repeat,
    "model_tag": model_tag,
    "run_dir": str(out_dir),
    "metrics": {"final_core": core, "train_bpb": train_bpb, "val_bpb": val_bpb},
    "per_task_core": per_task,
}
(out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Wrote eval-only summary: {out_dir / 'eval_summary.json'}")
PYEVAL
}

run_eval_only_one() {
    local arm="$1"
    local repeat="$2"
    local model_tag="$3"
    local data_dir="$4"
    local out_dir="$TASK_DELTA_ROOT/eval_only/${arm}_repeat${repeat}"
    mkdir -p "$out_dir"
    log "Eval-only $arm repeat=$repeat model=$model_tag"
    torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.base_eval -- \
        --eval=core,bpb \
        --max-per-task=-1 \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --seed=42 \
        --core-eval-seed="$CORE_EVAL_SEED" \
        --model-tag="$model_tag" \
        --data-source=parquet \
        --data-dir="$data_dir" 2>&1 | tee "$out_dir/base_eval.log"
    if compgen -G "$NANOCHAT_BASE_DIR/base_eval/base_model_*.csv" > /dev/null; then
        cp "$(ls -t "$NANOCHAT_BASE_DIR"/base_eval/base_model_*.csv | head -n 1)" "$out_dir/base_eval_core.csv"
    fi
    write_eval_summary "$out_dir" "$arm" "$repeat" "$model_tag"
}

stage_smoke() {
    log "Running static checks"
    bash -n runs/climbmix_semdedup_quality_b200.sh
    bash -n runs/climbmix_task_delta_investigation_b200.sh
    /usr/bin/python3 -m py_compile \
        nanochat/common.py \
        scripts/base_train.py \
        scripts/base_eval.py \
        scripts/build_random_drop_dataset.py \
        scripts/analyze_climbmix_removed_samples.py \
        scripts/aggregate_climbmix_task_delta.py

    log "Running random-drop plumbing smoke"
    RUN_ROOT="$TASK_DELTA_ROOT/smoke" \
    RUN_ID="smoke-randomdrop" \
    NUM_TRAIN_SHARDS=1 \
    MAX_DOCS=2000 \
    STATS_MAX_DOCS=2000 \
    RANDOM_DROP_REMOVED_DOCS=100 \
    RANDOM_DROP_SEED=9001 \
    RANDOM_DROP_OVERWRITE=1 \
    SKIP_TOKEN_STATS=1 \
    DO_TRAIN=0 \
    DO_EVAL=0 \
    PREP_DATASET=0 \
    SKIP_UV_SYNC=1 \
    run_quality randomdrop

    log "Running exact-smoke SemDeDup plumbing smoke"
    RUN_ROOT="$TASK_DELTA_ROOT/smoke" \
    RUN_ID="smoke-semdedup-exact" \
    NUM_TRAIN_SHARDS=1 \
    MAX_DOCS=2000 \
    STATS_MAX_DOCS=2000 \
    SEMD_BACKEND=exact-smoke \
    SEMD_OVERWRITE=1 \
    SKIP_TOKEN_STATS=1 \
    DO_TRAIN=0 \
    DO_EVAL=0 \
    PREP_DATASET=0 \
    SKIP_UV_SYNC=1 \
    run_quality semdedup
}

stage_eval_only() {
    for repeat in 1 2 3; do
        run_eval_only_one baseline "$repeat" "$EXISTING_BASELINE_MODEL_TAG" "$INPUT_DATA_DIR"
        run_eval_only_one semdedup "$repeat" "$EXISTING_SEMDEDUP_MODEL_TAG" "$EXISTING_SEMDEDUP_DATA_DIR"
    done
}

stage_removed_audit() {
    log "Auditing removed samples from current eps0.07 SemDeDup run"
    python -m scripts.analyze_climbmix_removed_samples \
        --run-dir "$EXISTING_SEMDEDUP_RUN_DIR" \
        --input-data-dir "$INPUT_DATA_DIR" \
        --output-dir "$TASK_DELTA_ROOT/removed_audit" \
        --sample-size "$AUDIT_SAMPLE_SIZE" \
        --seed "$AUDIT_SEED" \
        --num-train-shards "$NUM_TRAIN_SHARDS"
}

stage_full_seed_repeats() {
    for seed in 43 44; do
        log "Full baseline seed=$seed"
        RUN_ROOT="$TASK_DELTA_ROOT/full_seed_repeats" \
        RUN_ID="full-baseline-seed${seed}-n${NUM_TRAIN_SHARDS}-i${NUM_ITERATIONS}" \
        RUN_TAG_BASELINE="d${DEPTH}-climbmix-baseline-seed${seed}-n${NUM_TRAIN_SHARDS}-i${NUM_ITERATIONS}" \
        WANDB_RUN="climbmix-baseline-seed${seed}-n${NUM_TRAIN_SHARDS}-i${NUM_ITERATIONS}" \
        SEED="$seed" \
        DO_TRAIN=1 \
        DO_EVAL=1 \
        run_quality baseline

        log "Full SemDeDup eps0.07 seed=$seed reusing existing deduped data"
        RUN_ROOT="$TASK_DELTA_ROOT/full_seed_repeats" \
        RUN_ID="full-semdedup-eps0p07-seed${seed}-n${NUM_TRAIN_SHARDS}-i${NUM_ITERATIONS}" \
        RUN_TAG_SEMDEDUP="d${DEPTH}-climbmix-semdedup-eps0p07-seed${seed}-n${NUM_TRAIN_SHARDS}-i${NUM_ITERATIONS}" \
        WANDB_RUN="climbmix-semdedup-eps0p07-seed${seed}-n${NUM_TRAIN_SHARDS}-i${NUM_ITERATIONS}" \
        SEED="$seed" \
        SEMD_EPS=0.07 \
        SEMD_OUTPUT_DIR="$EXISTING_SEMDEDUP_DATA_DIR" \
        DO_SEMDEDUP=0 \
        DO_TRAIN=1 \
        DO_EVAL=1 \
        run_quality semdedup
    done
}

run_randomdrop_train() {
    local rd_seed="$1"
    local train_seed="$2"
    local suffix="$3"
    local rd_data_dir="$DATA_DIR/base_data_climbmix_randomdrop_drop${RANDOM_DROP_REMOVED_DOCS}_seed${rd_seed}_n${NUM_TRAIN_SHARDS}"
    local run_id="full-randomdrop-${suffix}-rdseed${rd_seed}-seed${train_seed}-n${NUM_TRAIN_SHARDS}-i${NUM_ITERATIONS}"
    local run_dir="$TASK_DELTA_ROOT/randomdrop_control/$run_id"
    if [[ -f "$run_dir/run_summary.json" ]]; then
        log "Skipping completed random-drop run: $run_dir"
        return
    fi
    local do_build=1
    if [[ -f "$rd_data_dir/random_drop_manifest.json" ]]; then
        do_build=0
    fi
    log "Random-drop control rd_seed=$rd_seed train_seed=$train_seed build=$do_build"
    RUN_ROOT="$TASK_DELTA_ROOT/randomdrop_control" \
    RUN_ID="$run_id" \
    RUN_TAG_RANDOMDROP="d${DEPTH}-climbmix-randomdrop-${suffix}-rdseed${rd_seed}-seed${train_seed}-n${NUM_TRAIN_SHARDS}-i${NUM_ITERATIONS}" \
    WANDB_RUN="climbmix-randomdrop-${suffix}-rdseed${rd_seed}-seed${train_seed}" \
    SEED="$train_seed" \
    RANDOM_DROP_OUTPUT_DIR="$rd_data_dir" \
    RANDOM_DROP_REMOVED_DOCS="$RANDOM_DROP_REMOVED_DOCS" \
    RANDOM_DROP_SEED="$rd_seed" \
    DO_RANDOM_DROP="$do_build" \
    DO_TRAIN=1 \
    DO_EVAL=1 \
    run_quality randomdrop
}

stage_randomdrop_control() {
    for seed in 42 43 44; do
        run_randomdrop_train "$RANDOM_DROP_SEED_PRIMARY" "$seed" primary
    done
    run_randomdrop_train "$RANDOM_DROP_SEED_SENSITIVITY" 42 sensitivity
}

run_semdedup_full() {
    local eps="$1"
    local train_seed="$2"
    local data_dir="$3"
    local eps_slug="${eps//./p}"
    local run_id="full-semdedup-eps${eps_slug}-seed${train_seed}-n${NUM_TRAIN_SHARDS}-i${NUM_ITERATIONS}"
    local run_dir="$TASK_DELTA_ROOT/eps_sweep/$run_id"
    if [[ -f "$run_dir/run_summary.json" ]]; then
        log "Skipping completed SemDeDup eps run: $run_dir"
        return
    fi
    local default_data_dir="$TASK_DELTA_ROOT/eps_sweep/data/base_data_climbmix_semdedup_eps${eps_slug}_n${NUM_TRAIN_SHARDS}"
    if [[ -z "$data_dir" ]]; then
        data_dir="$default_data_dir"
    fi
    local do_semdedup=1
    if [[ -f "$data_dir/semdedup_manifest.json" ]]; then
        do_semdedup=0
    fi
    log "SemDeDup eps=$eps train_seed=$train_seed build=$do_semdedup data=$data_dir"
    RUN_ROOT="$TASK_DELTA_ROOT/eps_sweep" \
    RUN_ID="$run_id" \
    RUN_TAG_SEMDEDUP="d${DEPTH}-climbmix-semdedup-eps${eps_slug}-seed${train_seed}-n${NUM_TRAIN_SHARDS}-i${NUM_ITERATIONS}" \
    WANDB_RUN="climbmix-semdedup-eps${eps_slug}-seed${train_seed}" \
    SEED="$train_seed" \
    SEMD_EPS="$eps" \
    SEMD_MODEL="$SEMD_MODEL" \
    SEMD_N_CLUSTERS="$SEMD_N_CLUSTERS" \
    SEMD_DISTANCE_METRIC="$SEMD_DISTANCE_METRIC" \
    SEMD_WHICH_TO_KEEP="$SEMD_WHICH_TO_KEEP" \
    SEMD_PAIRWISE_BATCH_SIZE="$SEMD_PAIRWISE_BATCH_SIZE" \
    SEMD_VLLM_ATTENTION_BACKEND="$SEMD_VLLM_ATTENTION_BACKEND" \
    SEMD_VLLM_ENFORCE_EAGER="$SEMD_VLLM_ENFORCE_EAGER" \
    SEMD_NO_RAY_PREINIT="$SEMD_NO_RAY_PREINIT" \
    SEMD_RAY_TEMP_DIR="$SEMD_RAY_TEMP_DIR" \
    SEMD_OUTPUT_DIR="$data_dir" \
    SEMD_OVERWRITE="${SEMD_OVERWRITE:-0}" \
    DO_SEMDEDUP="$do_semdedup" \
    DO_TRAIN=1 \
    DO_EVAL=1 \
    run_quality semdedup
}

select_eps_info() {
    /usr/bin/python3 - "$TASK_DELTA_ROOT" "$EXISTING_SEMDEDUP_RUN_DIR" "$EXISTING_SEMDEDUP_DATA_DIR" <<'PYSELECT'
import json
import sys
from pathlib import Path

task_root = Path(sys.argv[1])
existing_run = Path(sys.argv[2])
existing_data = sys.argv[3]
candidates = []
paths = list((task_root / "eps_sweep").rglob("run_summary.json"))
if (existing_run / "run_summary.json").exists():
    paths.append(existing_run / "run_summary.json")
for path in paths:
    summary = json.loads(path.read_text())
    run_dir = Path(summary.get("run_dir", path.parent))
    config = json.loads((run_dir / "run_config.json").read_text()) if (run_dir / "run_config.json").exists() else {}
    manifest_path = run_dir / "semdedup_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else summary.get("semdedup_manifest", {})
    eps = None
    if manifest.get("curator_config"):
        eps = manifest["curator_config"].get("eps")
    eps = eps if eps is not None else config.get("SEMD_EPS")
    metrics = summary.get("metrics", {})
    core = metrics.get("final_core")
    if eps is None or core is None:
        continue
    keep = manifest.get("keep_ratio_tokens") or manifest.get("keep_ratio_docs") or -1
    data_dir = summary.get("data_dir") or config.get("data_dir")
    if str(eps) == "0.07" and not data_dir:
        data_dir = existing_data
    candidates.append((float(eps), float(core), float(keep), data_dir or ""))
if not candidates:
    print("0.07\t" + existing_data)
    raise SystemExit
best_core = max(c[1] for c in candidates)
near = [c for c in candidates if best_core - c[1] <= 0.005]
near.sort(key=lambda item: (item[2], item[1]), reverse=True)
best = near[0]
print(f"{best[0]}\t{best[3]}")
PYSELECT
}

stage_eps_sweep() {
    start_ray_if_needed
    for eps in $EPS_SWEEP_VALUES; do
        run_semdedup_full "$eps" 42 ""
    done
    local selected
    selected="$(select_eps_info)"
    local selected_eps selected_data_dir
    selected_eps="$(printf '%s' "$selected" | cut -f1)"
    selected_data_dir="$(printf '%s' "$selected" | cut -f2-)"
    log "Selected eps for repeat seeds: eps=$selected_eps data=$selected_data_dir"
    if [[ "$selected_eps" == "0.07" || "$selected_eps" == "0.070000" ]]; then
        selected_data_dir="$EXISTING_SEMDEDUP_DATA_DIR"
    fi
    for seed in 43 44; do
        run_semdedup_full "$selected_eps" "$seed" "$selected_data_dir"
    done
}

stage_report() {
    log "Aggregating investigation report"
    /usr/bin/python3 -m scripts.aggregate_climbmix_task_delta \
        --task-root "$TASK_DELTA_ROOT" \
        --run-dir "$EXISTING_BASELINE_RUN_DIR" \
        --run-dir "$EXISTING_SEMDEDUP_RUN_DIR" \
        --output "$TASK_DELTA_ROOT/CLIMBMIX_SEMDEDUP_TASK_DELTA_INVESTIGATION.md"
}

run_mode() {
    case "$1" in
        smoke) stage_smoke ;;
        eval-only) stage_eval_only ;;
        removed-audit) stage_removed_audit ;;
        full-seed-repeats) stage_full_seed_repeats ;;
        randomdrop-control) stage_randomdrop_control ;;
        eps-sweep) stage_eps_sweep ;;
        report) stage_report ;;
        all)
            stage_smoke
            stage_eval_only
            stage_removed_audit
            stage_full_seed_repeats
            stage_randomdrop_control
            stage_eps_sweep
            stage_report
            ;;
        *)
            echo "Unknown mode: $1" >&2
            exit 2
            ;;
    esac
}

log "Starting task-delta investigation mode=$MODE pid=$$ root=$TASK_DELTA_ROOT"
run_mode "$MODE"
log "Finished task-delta investigation mode=$MODE"
