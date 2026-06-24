# ClimbMix SemDeDup + NanoChat Experiment

This runbook fixes the workflow for comparing NanoChat pretraining on ClimbMix
with and without NeMo Curator Semantic Deduplication. It focuses only on
semantic deduplication quality, not cuVS clustering performance.

## Goal

Answer two questions:

- Does ClimbMix + SemDeDup reach the same BPB or CORE in fewer iterations,
  fewer training tokens, or less training wall time?
- At a fixed training budget, does ClimbMix + SemDeDup improve or preserve BPB
  and CORE compared with no SemDeDup?

Use BPB as the primary convergence-efficiency curve. Use CORE as the downstream
quality metric. Treat small CORE deltas cautiously until there are repeat runs;
inspect per-task CORE deltas before interpreting the mean.

## Fixed A/B Setup

Keep these fixed across no-dedup and SemDeDup runs:

- ClimbMix parquet source and selected train shard count.
- Final sorted parquet shard as validation; do not run SemDeDup on validation.
- NanoChat tokenizer and token-byte mapping.
- Model depth, batch size, training horizon, eval cadence, and random seed
  where applicable.
- Base pretraining and base evaluation only; do not include SFT/RL.

Default B200-aligned settings:

- `NUM_GPUS=8`
- `DEPTH=24`
- `DEVICE_BATCH_SIZE=16`
- `PARAM_DATA_RATIO=9.5`
- `--window-pattern=L`
- `--fp8`
- `NCCL_NVLS_ENABLE=0`

## Metrics To Record

Primary result metrics:

- `val BPB` during training.
- final `CORE` from `scripts.base_eval`.
- per-task CORE centered scores.
- `iterations_to_target_BPB`.
- `tokens_to_target_BPB`.
- `runtime_to_target_BPB`.
- `final_BPB_at_fixed_iterations`.
- `final_CORE_at_fixed_iterations`.

Data metrics:

- input/output documents.
- input/output tokens using the NanoChat tokenizer.
- input/output characters and parquet bytes.
- document, token, and character keep ratios.
- document char/token length quantiles.
- training tokens, iterations, and wall time.

SemDeDup quality/config metrics:

- embedding model used.
- `eps`, `n_clusters`, distance metric, `which_to_keep`.
- embedding context/truncation setting, especially `embedding_max_chars`.
- dedup runtime.
- removed documents and tokens.
- removed and kept sample audit files.
- optional kept/removed doc ID files when `doc_id` survives Curator output.

Manual inspection labels for `removed_samples.jsonl`:

- exact duplicate.
- near duplicate.
- semantic duplicate or paraphrase.
- same template but different useful content.
- related but both valuable.
- false positive.
- unclear.

## Scripts

Unified runner:

```bash
bash runs/climbmix_semdedup_quality_b200.sh baseline|semdedup|both
```

Data stats:

```bash
python -m scripts.parquet_data_stats \
  --data-dir "$NANOCHAT_BASE_DIR/base_data_climbmix" \
  --split train \
  --num-train-shards 8 \
  --output /tmp/data_stats.json
```

Comparison report:

```bash
python -m scripts.compare_semdedup_experiments \
  --baseline "$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/<baseline_run_id>" \
  --semdedup "$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/<semdedup_run_id>" \
  --output "$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/comparison.md"
```

Each run directory contains:

- `run_config.json`
- `data_stats.json`
- `train.log`
- `base_eval.log`
- `base_eval_core.csv` when eval runs CORE.
- `report.md`
- `run_summary.json`
- SemDeDup-only: `semdedup_manifest.json`, `removed_samples.jsonl`,
  `kept_samples.jsonl`, and, when available, `kept_ids.txt` and
  `removed_ids.txt`.

## Smoke Runs

Plumbing smoke without training:

```bash
cd /home/nfs/hfang/curator_eval/nanochat
NUM_TRAIN_SHARDS=1 MAX_DOCS=2000 SEMD_BACKEND=exact-smoke DO_TRAIN=0 DO_EVAL=0 \
  bash runs/climbmix_semdedup_quality_b200.sh semdedup
```

No-SemDeDup pilot:

```bash
cd /home/nfs/hfang/curator_eval/nanochat
WANDB_RUN=climbmix-nosd-n8-i400 NUM_TRAIN_SHARDS=8 NUM_ITERATIONS=400 \
  bash runs/climbmix_semdedup_quality_b200.sh baseline
```

SemDeDup pilot:

```bash
cd /home/nfs/hfang/curator_eval/nanochat
WANDB_RUN=climbmix-semdedup-eps0p07-n8-i400 NUM_TRAIN_SHARDS=8 NUM_ITERATIONS=400 \
SEMD_EPS=0.07 SEMD_N_CLUSTERS=100 SEMD_MODEL=google/embeddinggemma-300m INSTALL_CURATOR=1 \
  bash runs/climbmix_semdedup_quality_b200.sh semdedup
```

Formal A/B:

```bash
cd /home/nfs/hfang/curator_eval/nanochat
NUM_TRAIN_SHARDS=170 PARAM_DATA_RATIO=9.5 CORE_METRIC_EVERY=2000 CORE_METRIC_MAX_PER_TASK=500 \
FINAL_CORE_MAX_PER_TASK=-1 bash runs/climbmix_semdedup_quality_b200.sh both
```

## Recommended W&B Panels

- `val/bpb` vs step.
- `val/bpb` vs training tokens.
- `core_metric` vs step.
- `train/tok_per_sec` vs step.
- final CORE summary.
- per-task CORE centered scores.
- SemDeDup keep ratios from `semdedup_manifest.json`.
- document token length before/after SemDeDup.
- removed sample audit precision from manual labeling.

## Interpretation

Report SemDeDup as helpful only if at least one of these holds:

- it reaches the no-dedup target BPB in fewer steps/tokens/time without CORE
  regression;
- it improves final BPB or CORE at fixed iterations;
- it removes a meaningful number of duplicate tokens with low sample-audit
  false-positive risk and no obvious length/source distribution distortion.

Do not claim quality improvement from removal ratio alone. Removal ratio is a
data reduction statistic, not a quality metric.
