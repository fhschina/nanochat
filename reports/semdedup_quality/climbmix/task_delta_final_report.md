# ClimbMix SemDeDup Task-Delta Final Report

## What This Investigated

The first ClimbMix SemDeDup run showed a surprising task-level split:
`commonsense_qa` improved a lot while `winograd` / `winogrande` moved in the
opposite direction. This report records the follow-up investigation added on
top of the original ClimbMix SemDeDup runbook.

The goal was to distinguish five possible explanations:

- Eval noise from CORE or BPB evaluation.
- Training seed variance.
- Generic data-removal effects from dropping the same amount of data.
- SemDeDup-specific data selection effects.
- Sensitivity to the SemDeDup threshold `eps`.

This work is about semantic deduplication quality. It does not benchmark or
optimize cuVS clustering performance.

## Implementation Added

This branch now includes:

- Seed controls in `scripts/base_train.py` and `scripts/base_eval.py`.
- Deterministic CORE subsampling through `--core-eval-seed`.
- `randomdrop` mode in `runs/climbmix_semdedup_quality_b200.sh`.
- `runs/climbmix_task_delta_investigation_b200.sh`, a resumable background
  orchestrator for eval-only repeats, seed repeats, random-drop controls,
  eps sweeps, removed-sample audit, and final aggregation.
- `scripts/build_random_drop_dataset.py`, which creates a control dataset by
  randomly dropping the same number of train documents removed by SemDeDup.
- `scripts/analyze_climbmix_removed_samples.py`, which audits removed vs kept
  documents for source, length, QA-like, coreference-like, and boilerplate-like
  patterns.
- `scripts/aggregate_climbmix_task_delta.py`, which collects run summaries,
  CORE CSVs, eval-only logs, manifests, and audit summaries into a markdown
  report.

Long-running stages are designed to be launched with `nohup ... &`; each stage
writes logs and pid files under:

```bash
$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_task_delta/logs
$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_task_delta/pids
```

## Completed Matrix

The final aggregation found:

- 15 train/eval run summaries.
- 6 eval-only summaries.
- 1 removed-sample audit.

Completed stages:

- Eval-only repeats: 3 repeats each for the existing baseline and SemDeDup
  checkpoints.
- Removed-sample audit: 1000 removed docs vs 1000 kept docs.
- Full seed repeats: baseline and SemDeDup eps0.07 for seeds 43 and 44, using
  the previous seed-42 style run as an anchor where available.
- Random-drop control: same removed-doc count as eps0.07 SemDeDup, with
  primary random-drop seed 9001 across training seeds 42, 43, 44, plus one
  sensitivity run with random-drop seed 9002.
- Eps sweep: SemDeDup eps values 0.05, 0.07, 0.09, and 0.12, with extra eps0.07
  seed repeats.

The generated aggregate artifact is:

```bash
/home/nfs/hfang/.cache/nanochat_b200_climbmix/experiments/climbmix_semdedup_task_delta/CLIMBMIX_SEMDEDUP_TASK_DELTA_INVESTIGATION.md
```

## Headline Results

### Eval-Only Stability

| Arm | Repeats | CORE mean | CORE std | Val BPB mean | Val BPB std |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3 | 0.2687 | 0.0000 | 0.708492 | 0.000000 |
| semdedup | 3 | 0.2775 | 0.0000 | 0.709092 | 0.000000 |

Eval-only repeats are deterministic for the checked checkpoints. The observed
task deltas are therefore not explained by repeated evaluation noise.

### Arm Summary

| Arm | Runs | Seeds | CORE mean | CORE std | Val BPB mean | Keep docs | Keep tokens | Removed docs |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3 | 43, 44 | 0.2732 | 0.0046 | 0.708398 | | | |
| randomdrop_seed9001 | 3 | 42, 43, 44 | 0.2733 | 0.0059 | 0.708527 | 0.9797 | 0.9797 | 291374 |
| randomdrop_seed9002 | 1 | 42 | 0.2710 | | 0.708680 | 0.9797 | 0.9798 | 291374 |
| semdedup_eps0.05 | 1 | 42 | 0.2717 | | 0.708909 | 0.9868 | 0.9868 | 189542 |
| semdedup_eps0.07 | 5 | 43, 44 | 0.2767 | 0.0035 | 0.708924 | 0.9797 | 0.9770 | 291374 |
| semdedup_eps0.09 | 1 | 42 | 0.2656 | | 0.709291 | 0.9697 | 0.9610 | 435382 |
| semdedup_eps0.12 | 1 | 42 | 0.2719 | | 0.709772 | 0.9453 | 0.9214 | 786367 |

The overall CORE gain from eps0.07 is small. It is also within the rough scale
of seed-to-seed variation seen in the baseline and random-drop controls. BPB is
slightly worse after SemDeDup, so the current evidence does not support a claim
that SemDeDup improves training efficiency or perplexity-like fit at this
budget.

### Focus Task Means

| Arm | commonsense_qa | winograd | winogrande |
| --- | ---: | ---: | ---: |
| baseline | 0.0599 | 0.3651 | 0.1629 |
| randomdrop_seed9001 | 0.0773 | 0.3529 | 0.1450 |
| randomdrop_seed9002 | 0.0459 | 0.3846 | 0.1208 |
| semdedup_eps0.05 | 0.1308 | 0.3114 | 0.1365 |
| semdedup_eps0.07 | 0.1632 | 0.3934 | 0.1252 |
| semdedup_eps0.09 | 0.0274 | 0.4359 | 0.1539 |
| semdedup_eps0.12 | 0.1011 | 0.3480 | 0.1555 |

The `commonsense_qa` jump is much larger for SemDeDup eps0.07 than for the
matched random-drop control, so it is not fully explained by removing 291,374
documents. `winogrande` drops for both SemDeDup eps0.07 and the random-drop
controls, which suggests that at least part of that task movement may be a
generic data-removal or seed-sensitive effect.

### Seed-Matched eps0.07 Delta vs Baseline

| Seed | Delta CORE | Delta BPB | Delta commonsense_qa | Delta winograd | Delta winogrande |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | +0.0037 | +0.000502 | +0.1658 | +0.0073 | -0.0710 |
| 44 | +0.0028 | +0.000480 | +0.0543 | +0.0659 | -0.0332 |

Positive BPB delta means worse BPB. The CORE delta is positive for both
seed-matched repeats, but the magnitude is modest. The task-level pattern is
more interesting than the aggregate CORE movement.

### Removed-Sample Audit

| Group | Docs | QA-like | Coref-like | Boilerplate-like | p50 chars | p95 chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| removed | 1000 | 0.4330 | 0.9020 | 0.0570 | 3040 | 8668 |
| kept | 1000 | 0.5150 | 0.9170 | 0.0230 | 2399 | 7130 |

The removed sample has fewer QA-like documents than the kept sample and more
boilerplate-like documents. The removed docs are also longer on average. This
is directionally consistent with SemDeDup removing repeated or templated long
content, but the audit does not by itself prove causality for the downstream
CORE task changes.

## Interpretation

The cleanest conclusion is:

- The original task deltas are not an eval-only artifact.
- They are not fully explained by random removal of the same number of docs.
- The SemDeDup eps0.07 result shows a real-looking task redistribution:
  `commonsense_qa` benefits more than the random-drop control, while
  `winogrande` weakens in both random-drop and SemDeDup.
- The aggregate CORE improvement is small and should be treated as
  inconclusive without more seeds or a larger evaluation budget.
- BPB is slightly worse for SemDeDup in the completed runs, so there is no
  evidence here that SemDeDup reaches the same BPB faster.
- Threshold choice matters. eps0.09 and eps0.12 remove more data and do not
  improve CORE/BPB over eps0.07. eps0.05 removes less and does not match the
  eps0.07 CORE/task behavior.

## Practical Recommendation

For reporting to the team:

- Say the ClimbMix baseline, SemDeDup, random-drop, seed-repeat, eval-repeat,
  and eps-sweep pipelines all ran successfully.
- Do not claim a robust aggregate quality win yet.
- Say eps0.07 is the best of the tried SemDeDup thresholds on CORE, but it has
  slightly worse BPB and a task-level tradeoff.
- Treat the commonsense_qa improvement as the most interesting signal. It looks
  SemDeDup-specific relative to random drop.
- Treat the winogrande drop as unresolved because random drop also weakens it.
- If the team wants a stronger conclusion, run more seeds and/or evaluate
  `commonsense_qa`, `winograd`, and `winogrande` with a larger per-task budget.

## How To Reproduce The Investigation

```bash
cd /home/nfs/hfang/curator_eval/nanochat
source .venv/bin/activate

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_b200_climbmix"
export TASK_DELTA_ROOT="$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_task_delta"
mkdir -p "$TASK_DELTA_ROOT/logs" "$TASK_DELTA_ROOT/pids"

nohup bash runs/climbmix_task_delta_investigation_b200.sh all \
  > "$TASK_DELTA_ROOT/logs/all.$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
echo $! > "$TASK_DELTA_ROOT/pids/all.pid"
```

To rerun only the final aggregation:

```bash
python -m scripts.aggregate_climbmix_task_delta \
  --task-root "$TASK_DELTA_ROOT" \
  --quality-root "$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality" \
  --output "$TASK_DELTA_ROOT/CLIMBMIX_SEMDEDUP_TASK_DELTA_INVESTIGATION.md"
```

To check a background stage:

```bash
ps -fp "$(cat "$TASK_DELTA_ROOT/pids/all.pid")"
tail -f "$(ls -t "$TASK_DELTA_ROOT"/logs/all.*.log | head -n 1)"
```

## Caveats

- Some SemDeDup eps0.07 rows include the earlier completed anchor run, while
  the strict seed-matched comparison is based on seeds 43 and 44.
- CORE aggregate deltas are small compared with seed-level variation.
- BPB and CORE answer different questions: BPB is the training/validation
  compression curve, while CORE is downstream task quality. In this result,
  they do not move in the same direction.
- The aggregate artifact lives under `$NANOCHAT_BASE_DIR`, not in git, because
  it is generated from experiment outputs.
