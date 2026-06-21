# ClimbMix SemDeDup Branch Report

## What

This branch turns the ClimbMix semantic deduplication experiment into a
repeatable NanoChat workflow. It compares:

- No SemDeDup: NanoChat base pretraining on ClimbMix parquet.
- SemDeDup: ClimbMix train shards reduced with NeMo Curator Semantic
  Deduplication, followed by the same NanoChat base pretraining and base eval.

The experiment focuses on semantic deduplication quality and downstream
training behavior. It does not optimize or profile cuVS clustering performance.

Main questions:

- At a fixed training budget, does SemDeDup improve or preserve BPB and CORE?
- Does SemDeDup reduce duplicated documents and tokens enough to justify the
  preprocessing cost?
- Are any quality changes broad based, or concentrated in specific CORE tasks?

## Changes From Upstream NanoChat

Compared with upstream NanoChat, this branch adds experiment plumbing around
ClimbMix and SemDeDup while keeping NanoChat model/training code unchanged.

Added files:

- `dev/CLIMBMIX_SEMDEDUP_EXPERIMENT.md`: experiment runbook and metric plan.
- `dev/CLIMBMIX_SEMDEDUP_BRANCH_REPORT.md`: this branch report and result
  summary.
- `runs/climbmix_semdedup_quality_b200.sh`: one command runner for baseline,
  SemDeDup, or both.
- `scripts/climbmix_data_stats.py`: tokenizer-based parquet data statistics.
- `scripts/compare_climbmix_experiments.py`: baseline vs SemDeDup markdown
  comparison generator.

Modified files:

- `scripts/build_semdedup_climbmix.py`
  - stages ClimbMix train shards with stable `doc_id`;
  - runs NeMo Curator `TextSemanticDeduplicationWorkflow`;
  - normalizes output back to NanoChat parquet schema with only `text`;
  - records token/doc/char stats and keep ratios;
  - writes `semdedup_manifest.json`, kept/removed IDs, and kept/removed sample
    audit files;
  - supports short Ray temp dirs and `--no-ray-preinit`;
  - supports vLLM embedding init workarounds such as `TRITON_ATTN` and
    `enforce_eager`.
- `runs/climbmix_semdedup_quality_b200.sh`
  - adds B200-oriented defaults;
  - controls SemDeDup, training, and eval through environment flags;
  - records `run_config.json`, `data_stats.json`, `train.log`,
    `base_eval.log`, `base_eval_core.csv`, `report.md`, and `run_summary.json`;
  - fixes run summary parsing for BPB, CORE, training tokens, runtime, and
    iterations.

NanoChat base model code, dataloader code, tokenizer code, and eval code are
not changed by this branch.

## How

### Environment Setup

Use the real repo:

```bash
cd /home/nfs/hfang/curator_eval/nanochat
git switch climbmix-semdedup-quality-runbook
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_b200_climbmix"
```

If NeMo Curator is not installed:

```bash
uv pip install --extra-index-url https://pypi.nvidia.com "nemo-curator[text_cuda12]"
```

For `google/embeddinggemma-300m`, Hugging Face gated model access must be
enabled for the user/token:

```bash
hf auth whoami
```

If W&B should be uploaded live:

```bash
wandb login
```

If W&B should be synced later:

```bash
export WANDB_MODE=offline
```

If W&B should be disabled:

```bash
WANDB_RUN=dummy
```

### Ray Setup For SemDeDup

Use a short Ray temp dir. This avoids UNIX socket path length failures from deep
experiment/cache paths.

```bash
ray stop --force || true
rm -rf /tmp/ray_hfang
mkdir -p /tmp/ray_hfang

export RAY_PORT=24017
export RAY_MAX_LIMIT_FROM_API_SERVER=100000
export RAY_TMPDIR=/tmp/ray_hfang
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

ray start --head \
  --node-ip-address=127.0.0.1 \
  --port="$RAY_PORT" \
  --dashboard-host=127.0.0.1 \
  --temp-dir="$RAY_TMPDIR"

export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
```

### Static Tests

```bash
bash -n runs/climbmix_semdedup_quality_b200.sh
PYTHONPYCACHEPREFIX=/tmp/codex_pycache /usr/bin/python3 -m py_compile \
  scripts/build_semdedup_climbmix.py \
  scripts/climbmix_data_stats.py \
  scripts/compare_climbmix_experiments.py
```

### Plumbing Smoke Test

This verifies data reading, tokenizer loading, manifest writing, and output
schema without running NanoChat training.

```bash
NUM_TRAIN_SHARDS=1 \
MAX_DOCS=2000 \
STATS_MAX_DOCS=2000 \
SEMD_BACKEND=exact-smoke \
DO_TRAIN=0 \
DO_EVAL=0 \
bash runs/climbmix_semdedup_quality_b200.sh semdedup
```

### Curator SemDeDup Smoke Test On B200

`google/embeddinggemma-300m` is an encoder-only/pooling model in vLLM. On this
B200 environment, the default flash-attention path hit a CUDA PTX toolchain
error, and `FLASHINFER` is not valid for encoder-only attention. The working
backend was `TRITON_ATTN` with eager execution.

```bash
RUN_ID=semdedup_triton_smoke \
NUM_TRAIN_SHARDS=1 \
MAX_DOCS=2000 \
STATS_MAX_DOCS=2000 \
SEMD_OVERWRITE=1 \
SEMD_RAY_TEMP_DIR=/tmp/ray_hfang \
SEMD_NO_RAY_PREINIT=1 \
SEMD_VLLM_ATTENTION_BACKEND=TRITON_ATTN \
SEMD_VLLM_ENFORCE_EAGER=1 \
DO_TRAIN=0 \
DO_EVAL=0 \
SKIP_TOKEN_STATS=1 \
SKIP_UV_SYNC=1 \
PREP_DATASET=0 \
bash runs/climbmix_semdedup_quality_b200.sh semdedup
```

### Full SemDeDup Run

This runs SemDeDup preprocessing, data stats, NanoChat base train, and base
eval. Use a fresh `RUN_ID` if preserving previous outputs matters.

```bash
RUN_ID=full-n170-r9p5-eps0p07-20260612T080000Z_semdedup \
WANDB_RUN=dummy \
NUM_GPUS=8 \
DEPTH=24 \
DEVICE_BATCH_SIZE=16 \
NUM_TRAIN_SHARDS=170 \
NUM_ITERATIONS=6612 \
PARAM_DATA_RATIO=9.5 \
NCCL_NVLS_ENABLE=0 \
SEMD_MODEL=google/embeddinggemma-300m \
SEMD_EPS=0.07 \
SEMD_N_CLUSTERS=100 \
SEMD_DISTANCE_METRIC=cosine \
SEMD_WHICH_TO_KEEP=hard \
SEMD_PAIRWISE_BATCH_SIZE=1024 \
SEMD_VLLM_ATTENTION_BACKEND=TRITON_ATTN \
SEMD_VLLM_ENFORCE_EAGER=1 \
SEMD_RAY_TEMP_DIR=/tmp/ray_hfang \
SEMD_NO_RAY_PREINIT=1 \
SEMD_OVERWRITE=1 \
CORE_METRIC_EVERY=2000 \
CORE_METRIC_MAX_PER_TASK=500 \
FINAL_CORE_MAX_PER_TASK=-1 \
BASE_EVAL_MODES=core,bpb,sample \
SKIP_UV_SYNC=1 \
PREP_DATASET=0 \
DO_SEMDEDUP=1 \
DO_TRAIN=1 \
DO_EVAL=1 \
bash runs/climbmix_semdedup_quality_b200.sh semdedup
```

### Background Run

```bash
nohup bash -lc '
set -euo pipefail
cd /home/nfs/hfang/curator_eval/nanochat
source .venv/bin/activate

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_b200_climbmix"
export RUN_ROOT="$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality"

ray stop --force || true
rm -rf /tmp/ray_hfang
mkdir -p /tmp/ray_hfang

export RAY_PORT=24017
export RAY_MAX_LIMIT_FROM_API_SERVER=100000
export RAY_TMPDIR=/tmp/ray_hfang
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

ray start --head \
  --node-ip-address=127.0.0.1 \
  --port="$RAY_PORT" \
  --dashboard-host=127.0.0.1 \
  --temp-dir="$RAY_TMPDIR"

export RAY_ADDRESS="127.0.0.1:$RAY_PORT"

RUN_ID=full-n170-r9p5-eps0p07-20260612T080000Z_semdedup \
WANDB_RUN=dummy \
NUM_GPUS=8 \
DEPTH=24 \
DEVICE_BATCH_SIZE=16 \
NUM_TRAIN_SHARDS=170 \
NUM_ITERATIONS=6612 \
PARAM_DATA_RATIO=9.5 \
NCCL_NVLS_ENABLE=0 \
SEMD_MODEL=google/embeddinggemma-300m \
SEMD_EPS=0.07 \
SEMD_N_CLUSTERS=100 \
SEMD_DISTANCE_METRIC=cosine \
SEMD_WHICH_TO_KEEP=hard \
SEMD_PAIRWISE_BATCH_SIZE=1024 \
SEMD_VLLM_ATTENTION_BACKEND=TRITON_ATTN \
SEMD_VLLM_ENFORCE_EAGER=1 \
SEMD_RAY_TEMP_DIR=/tmp/ray_hfang \
SEMD_NO_RAY_PREINIT=1 \
SEMD_OVERWRITE=1 \
CORE_METRIC_EVERY=2000 \
CORE_METRIC_MAX_PER_TASK=500 \
FINAL_CORE_MAX_PER_TASK=-1 \
BASE_EVAL_MODES=core,bpb,sample \
SKIP_UV_SYNC=1 \
PREP_DATASET=0 \
DO_SEMDEDUP=1 \
DO_TRAIN=1 \
DO_EVAL=1 \
bash runs/climbmix_semdedup_quality_b200.sh semdedup
' > "$HOME/climbmix_semdedup_full.nohup.log" 2>&1 &
```

Monitor:

```bash
tail -f "$HOME/climbmix_semdedup_full.nohup.log"
nvidia-smi
```

Cleanup after SemDeDup if Ray workers remain:

```bash
ray stop --force
```

### Parameter Reference

Runner mode:

- `baseline`: no SemDeDup, train/eval on ClimbMix.
- `semdedup`: SemDeDup arm. This can run preprocessing, train, and eval.
- `both`: baseline followed by SemDeDup.

Execution switches:

- `DO_SEMDEDUP=1`: build SemDeDup data.
- `DO_SEMDEDUP=0`: reuse existing SemDeDup data.
- `DO_TRAIN=1`: run NanoChat base train.
- `DO_EVAL=1`: run NanoChat base eval.
- `PREP_DATASET=0`: skip `python -m nanochat.dataset`.
- `SKIP_UV_SYNC=1`: skip `uv sync --extra gpu`.

Data/training:

- `NUM_TRAIN_SHARDS`: number of ClimbMix train shards. Formal run used `170`.
- `NUM_ITERATIONS`: fixed train iterations. Formal A/B used `6612`.
- `PARAM_DATA_RATIO`: training horizon if `NUM_ITERATIONS` is unset.
- `DEPTH`: NanoChat model depth. Formal run used `24`.
- `DEVICE_BATCH_SIZE`: per-device batch size. Formal run used `16`.
- `NUM_GPUS`: number of GPUs. Formal run used `8`.

SemDeDup:

- `SEMD_MODEL`: embedding model. Formal run used `google/embeddinggemma-300m`.
- `SEMD_EPS`: duplicate threshold. Formal run used `0.07`.
- `SEMD_N_CLUSTERS`: k-means clusters. Formal run used `100`.
- `SEMD_DISTANCE_METRIC`: `cosine` or `l2`. Formal run used `cosine`.
- `SEMD_WHICH_TO_KEEP`: duplicate retention strategy. Formal run used `hard`.
- `SEMD_PAIRWISE_BATCH_SIZE`: pairwise similarity batch size.
- `SEMD_VLLM_ATTENTION_BACKEND`: vLLM attention backend. B200 run used
  `TRITON_ATTN`.
- `SEMD_VLLM_ENFORCE_EAGER=1`: passes `enforce_eager=True` to vLLM.
- `SEMD_RAY_TEMP_DIR`: short Ray temp dir, for example `/tmp/ray_hfang`.
- `SEMD_NO_RAY_PREINIT=1`: use the externally started Ray cluster.
- `SEMD_OVERWRITE=1`: overwrite the run's SemDeDup output/cache.

Evaluation:

- `EVAL_EVERY`: BPB eval cadence during training.
- `EVAL_TOKENS`: number of tokens for BPB eval.
- `CORE_METRIC_EVERY`: CORE eval cadence during training.
- `CORE_METRIC_MAX_PER_TASK`: sampled examples per CORE task during training.
- `FINAL_CORE_MAX_PER_TASK`: final eval examples per task. `-1` means full.
- `BASE_EVAL_MODES`: final eval modes, for example `core,bpb,sample`.

## Implementation Details

The SemDeDup builder stages the selected ClimbMix train shards into parquet
files with stable IDs:

- `doc_id`: stable document identifier used for kept/removed ID tracking.
- `source_file`: source shard name for audit samples.
- `text`: original text.

Curator then runs:

- embedding generation through `VLLMEmbeddingModelStage`;
- semantic duplicate identification using `eps`, `n_clusters`, and distance
  metric;
- duplicate removal to produce deduplicated parquet.

The builder normalizes Curator output to NanoChat's expected parquet format:

- train shards contain only a `text` column;
- validation uses the same final ClimbMix shard as baseline and is not deduped.

The run wrapper records all experiment artifacts into:

```text
$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/$RUN_ID/
```

Important artifacts:

- `run_config.json`: environment and run parameters.
- `data_stats.json`: tokenizer-based docs/chars/tokens stats.
- `semdedup_manifest.json`: SemDeDup config, keep ratios, runtime, artifacts.
- `removed_samples.jsonl` and `kept_samples.jsonl`: stable audit samples.
- `kept_ids.txt` and `removed_ids.txt`: document ID lists when available.
- `train.log`: NanoChat training log with BPB and CORE curves.
- `base_eval.log`: final BPB/CORE/sample eval log.
- `base_eval_core.csv`: per-task CORE results.
- `run_summary.json`: parsed aggregate metrics and curves.
- `report.md`: NanoChat generated report.

The comparison script reads the run directories and writes a markdown A/B table:

```bash
.venv/bin/python -m scripts.compare_climbmix_experiments \
  --baseline "$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/full-n170-r9p5-eps0p07-20260611T163000Z_baseline" \
  --semdedup "$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/full-n170-r9p5-eps0p07-20260612T080000Z_semdedup" \
  --output "$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/full-n170-r9p5-eps0p07-20260612T080000Z_comparison.md"
```

## Results

Completed run directories:

- Baseline:
  `$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/full-n170-r9p5-eps0p07-20260611T163000Z_baseline`
- SemDeDup:
  `$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/full-n170-r9p5-eps0p07-20260612T080000Z_semdedup`
- Comparison:
  `$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/full-n170-r9p5-eps0p07-20260612T080000Z_comparison.md`

### Summary Table

| Metric | No SemDeDup | SemDeDup | Delta |
|---|---:|---:|---:|
| run id | full-n170-r9p5-eps0p07-20260611T163000Z_baseline | full-n170-r9p5-eps0p07-20260612T080000Z_semdedup | - |
| model tag | d24-climbmix-nosd-n170-20260611T171735Z | d24-climbmix-semdedup-eps0p07-n170-20260612T172022Z | - |
| data docs | 14,386,176 | 14,094,802 | -291,374 |
| data tokens | 9,158,993,442 | 8,948,462,810 | -210,530,632 |
| data chars | 42,783,740,697 | 41,770,785,554 | -1,012,955,143 |
| SemDeDup input docs | - | 14,386,176 | - |
| SemDeDup output docs | - | 14,094,802 | - |
| SemDeDup input tokens | - | 9,158,993,442 | - |
| SemDeDup output tokens | - | 8,948,462,810 | - |
| doc keep ratio | - | 0.979746 | - |
| token keep ratio | - | 0.977014 | - |
| removed docs | - | 291,374 | - |
| removed tokens | - | 210,530,632 | - |
| dedup runtime sec | - | 4692.978460 | - |
| embedding model | - | google/embeddinggemma-300m | - |
| eps | - | 0.070000 | - |
| n clusters | - | 100 | - |
| train iterations | 6,612 | 6,612 | 0 |
| train tokens | 6,933,184,512 | 6,933,184,512 | 0 |
| train runtime sec | 5149.800000 | 5149.800000 | 0.000000 |
| min val BPB | 0.711159 | 0.711762 | 0.000603 |
| final train val BPB | 0.711159 | 0.711762 | 0.000603 |
| base eval val BPB | 0.708492 | 0.709092 | 0.000600 |
| final CORE | 0.268700 | 0.277500 | 0.008800 |
| steps to baseline BPB <= 0.711159 | 6,612 | - | - |
| tokens to baseline BPB | 6,933,184,512 | - | - |
| runtime sec to baseline BPB | 5149.800000 | - | - |

### Per-Task CORE Delta

| CORE Task | No SemDeDup Centered | SemDeDup Centered | Delta |
|---|---:|---:|---:|
| CORE | 0.268661 | 0.277506 | 0.008845 |
| agi_eval_lsat_ar | 0.103261 | 0.097826 | -0.005435 |
| arc_challenge | 0.208191 | 0.205916 | -0.002275 |
| arc_easy | 0.602694 | 0.602694 | 0.000000 |
| bigbench_cs_algorithms | 0.406818 | 0.391667 | -0.015151 |
| bigbench_dyck_languages | 0.114000 | 0.117000 | 0.003000 |
| bigbench_language_identification | 0.167987 | 0.175798 | 0.007811 |
| bigbench_operators | 0.190476 | 0.171429 | -0.019047 |
| bigbench_qa_wikidata | 0.473845 | 0.481768 | 0.007923 |
| bigbench_repeat_copy_logic | 0.000000 | 0.000000 | 0.000000 |
| boolq | -0.073555 | -0.004346 | 0.069209 |
| commonsense_qa | 0.013104 | 0.140049 | 0.126945 |
| copa | 0.300000 | 0.340000 | 0.040000 |
| coqa | 0.312414 | 0.326945 | 0.014531 |
| hellaswag | 0.424550 | 0.418310 | -0.006240 |
| hellaswag_zeroshot | 0.419638 | 0.409945 | -0.009693 |
| jeopardy | 0.110534 | 0.125177 | 0.014643 |
| lambada_openai | 0.453134 | 0.451970 | -0.001164 |
| openbook_qa | 0.210667 | 0.221333 | 0.010666 |
| piqa | 0.487486 | 0.507073 | 0.019587 |
| squad | 0.467171 | 0.489877 | 0.022706 |
| winograd | 0.362637 | 0.318681 | -0.043956 |
| winogrande | 0.155485 | 0.116022 | -0.039463 |

## Conclusions

At the fixed 6,612 iteration budget:

- SemDeDup removed 291,374 documents and 210.5M tokens, about 2.3 percent of
  ClimbMix train tokens.
- SemDeDup did not improve BPB in this single run. Final train-time validation
  BPB increased from 0.711159 to 0.711762, and base eval validation BPB
  increased from 0.708492 to 0.709092. Lower BPB is better, so this is a small
  regression.
- SemDeDup improved final CORE from 0.2687 to 0.2775, a +0.0088 absolute delta.
- The CORE gain is not uniform. The largest positive centered deltas include
  `commonsense_qa`, `boolq`, `copa`, `squad`, and `piqa`; the largest negative
  deltas include `winograd`, `winogrande`, `bigbench_operators`, and
  `bigbench_cs_algorithms`.
- SemDeDup did not reach the baseline final BPB target within this fixed run,
  so this run does not support "same BPB in fewer iterations."
- This run does support a cautious "better CORE at fixed iterations, with a
  small BPB regression" observation.

Recommended next checks:

- Repeat with at least one more seed before treating the +0.0088 CORE delta as
  stable.
- Manually inspect `removed_samples.jsonl` and label false positives.
- Run an `eps` sweep, for example `0.05`, `0.07`, and `0.09`, because the
  current point may be slightly too aggressive for BPB.
- Track source and length distribution shifts before and after SemDeDup.
- If W&B is needed after offline runs, use `WANDB_MODE=offline` and then
  `wandb sync`, instead of `WANDB_RUN=dummy`.

## Follow-Up Task-Delta Investigation

The recommended follow-up checks were implemented and run after the initial
Phase 4-style baseline vs SemDeDup comparison. See
`dev/CLIMBMIX_SEMDEDUP_TASK_DELTA_FINAL_REPORT.md` for the final investigation
summary.

Short version:

- Eval-only repeats were deterministic for the checked checkpoints.
- Random-drop controls did not fully explain the large `commonsense_qa`
  improvement under SemDeDup eps0.07.
- `winogrande` weakened under both random drop and SemDeDup, so that movement
  remains consistent with generic data-removal or seed-sensitive effects.
- eps0.07 was the best SemDeDup threshold among the tried values for aggregate
  CORE, but the gain was small and BPB was slightly worse.
- The current evidence supports reporting a task-level redistribution signal,
  not a robust aggregate quality or training-efficiency win.

