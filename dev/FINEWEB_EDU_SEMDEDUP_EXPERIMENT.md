# FineWeb-EDU SemDeDup + NanoChat Experiment

This runbook mirrors the ClimbMix SemDeDup quality experiment, but replaces the
pretraining source with FineWeb-EDU. The comparison is only:

- FineWeb-EDU baseline
- FineWeb-EDU SemDeDup

Do not mix ClimbMix results into the primary conclusion.

## Goal

Measure whether SemDeDup on FineWeb-EDU improves or preserves downstream
training quality and convergence efficiency.

Primary questions:

- Does SemDeDup reach the FineWeb-EDU baseline BPB in fewer steps, training
  tokens, or wall time?
- At a fixed training budget, does SemDeDup improve or preserve validation BPB
  and CORE?
- Are removed samples actually duplicate-like, with low false-positive risk?

## Fixed Setup

Keep these fixed across baseline and SemDeDup:

- Dataset: `karpathy/fineweb-edu-100b-shuffle`
- Validation shard: `shard_01822.parquet`
- Data root: `$HOME/.cache/nanochat_b200_fineweb_edu`
- Run root: `$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality`
- NanoChat tokenizer and token-byte mapping
- `NUM_GPUS=8`
- `DEPTH=24`
- `DEVICE_BATCH_SIZE=16`
- `PARAM_DATA_RATIO=9.5` for formal runs
- `SEED=42`
- Base pretraining and base eval only; no SFT/RL

## Background Execution

All time-consuming runs should be launched with `nohup` and allowed to continue
in the background.

Recommended one-command launch:

```bash
cd /home/nfs/hfang/curator_eval/nanochat
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_b200_fineweb_edu"
export RUN_ROOT="$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality"
mkdir -p "$RUN_ROOT/nohup_logs"

setsid nohup bash runs/fineweb_edu_semdedup_ab_nohup_b200.sh all \
  > "$RUN_ROOT/nohup_logs/all_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
echo $! > "$RUN_ROOT/nohup_logs/latest_all.pid"
```

Check progress:

```bash
tail -f "$RUN_ROOT/nohup_logs/"*.log
cat "$RUN_ROOT/nohup_logs/latest_all.pid"
```

If the baseline pilot is already complete and the SemDeDup pilot failed, resume
from the SemDeDup step with short Ray socket paths:

```bash
cd /home/nfs/hfang/curator_eval/nanochat
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_b200_fineweb_edu"
export RUN_ROOT="$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality"
mkdir -p "$RUN_ROOT/nohup_logs"

SKIP_UV_SYNC=1 setsid nohup bash -x runs/fineweb_edu_semdedup_ab_nohup_b200.sh resume \
  > "$RUN_ROOT/nohup_logs/resume_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
echo $! > "$RUN_ROOT/nohup_logs/latest_all.pid"
```

## Run Sequence

The orchestrator runs these steps serially:

1. `smoke`
   - `NUM_TRAIN_SHARDS=1`
   - `MAX_DOCS=2000`
   - `SEMD_BACKEND=exact-smoke`
   - `DO_TRAIN=0`
   - `DO_EVAL=0`
   - Confirms data download, shard selection, schema normalization, manifests,
     and report plumbing.

2. `pilot-baseline`
   - `RUN_ID=pilot-n8-i400-baseline`
   - `NUM_TRAIN_SHARDS=8`
   - `NUM_ITERATIONS=400`
   - `FINAL_CORE_MAX_PER_TASK=500`

3. `pilot-semdedup`
   - `RUN_ID=pilot-n8-i400-semdedup-eps0p07`
   - `NUM_TRAIN_SHARDS=8`
   - `NUM_ITERATIONS=400`
   - `SEMD_EPS=0.07`
   - `SEMD_N_CLUSTERS=100`
   - `SEMD_MODEL=google/embeddinggemma-300m`
   - `SEMD_RAY_TEMP_DIR=/tmp/fwe_ray_pilot`
   - `SEMD_VLLM_ATTENTION_BACKEND=TRITON_ATTN`
   - `SEMD_VLLM_ENFORCE_EAGER=1`
   - `SEMD_OVERWRITE=1` by default, so a failed pilot cache can be retried
   - `FINAL_CORE_MAX_PER_TASK=500`

4. `formal`
   - `RUN_ID=full-n170-r9p5-eps0p07-<UTC timestamp>`
   - `NUM_TRAIN_SHARDS=170`
   - `PARAM_DATA_RATIO=9.5`
   - `SEMD_EPS=0.07`
   - `SEMD_N_CLUSTERS=100`
   - `SEMD_MODEL=google/embeddinggemma-300m`
   - `SEMD_RAY_TEMP_DIR=/tmp/fwe_ray_formal`
   - `SEMD_VLLM_ATTENTION_BACKEND=TRITON_ATTN`
   - `SEMD_VLLM_ENFORCE_EAGER=1`
   - `CORE_METRIC_EVERY=2000`
   - `CORE_METRIC_MAX_PER_TASK=500`
   - `FINAL_CORE_MAX_PER_TASK=-1`

5. `report`
   - Generates
     `$RUN_ROOT/fineweb_edu_full_n170_r9p5_eps0p07_comparison.md`
   - Compares the latest matching formal baseline and SemDeDup run dirs.

## Report Contents

The comparison report must include:

- Summary table with docs/tokens/chars, SemDeDup input/output, keep ratios,
  removed docs/tokens, dedup runtime, training iterations/tokens/runtime,
  BPB, and final CORE.
- Per-task CORE centered-score deltas.
- Notes that BPB is the convergence-efficiency metric, CORE is the downstream
  quality metric, and removed/kept samples should be manually audited.

## Acceptance Criteria

Baseline and SemDeDup formal runs each produce:

- `run_config.json`
- `data_stats.json`
- `train.log`
- `base_eval.log`
- `base_eval_core.csv`
- `run_summary.json`
- `report.md`

SemDeDup formal run additionally produces:

- `semdedup_manifest.json`
- `removed_samples.jsonl`
- `kept_samples.jsonl`
- `kept_ids.txt`
- `removed_ids.txt`

The report title should be:

```text
# fineweb-edu SemDeDup Experiment Comparison
```

## Interpretation

Call SemDeDup effective if it reaches the baseline BPB in fewer steps, tokens,
or runtime without a meaningful CORE regression.

Call it neutral if BPB and CORE are close to baseline, while the removed samples
are mostly duplicate-like under manual audit.

Call it negative if BPB or CORE regresses materially, or if per-task CORE shows
systematic degradation.
