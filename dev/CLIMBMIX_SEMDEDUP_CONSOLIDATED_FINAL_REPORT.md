# ClimbMix SemDeDup Consolidated Final Report

Date: 2026-06-22

This report supersedes the initial single-run branch report and integrates the
follow-up task-delta investigation. It combines:

- the original Phase 4-style no-SemDeDup vs SemDeDup A/B run;
- eval-only repeats;
- seed-repeat runs;
- random-drop controls;
- SemDeDup eps sweep runs;
- removed-sample audit results;
- additional interpretation and caveats from inspecting the artifacts.

The scope is semantic deduplication quality for ClimbMix + NanoChat. It does
not benchmark cuVS clustering performance.

## Executive Summary

The short answer is:

- The pipeline works end to end for ClimbMix baseline, SemDeDup, random-drop
  controls, seed repeats, eval-only repeats, and eps sweeps.
- SemDeDup eps0.07 removes a small but non-trivial amount of data:
  291,374 train documents and 210.5M train tokens, about 2.0% of docs and 2.3%
  of tokens.
- The original single-run result showed better CORE at fixed iterations
  (+0.0088 absolute) but slightly worse BPB (+0.0006, where lower is better).
- The multi-run result does not support a robust aggregate quality win yet.
  SemDeDup eps0.07 has a small CORE advantage over baseline, but the advantage
  is on the same scale as run-to-run variation.
- BPB is consistently slightly worse for SemDeDup. There is no evidence from
  these runs that SemDeDup reaches the same BPB faster.
- The most interesting signal is task-level redistribution:
  `commonsense_qa` improves much more under SemDeDup eps0.07 than under the
  matched random-drop control.
- `winogrande` remains unresolved: it weakens under both SemDeDup and
  random-drop controls, so it may be a generic data-removal or seed-sensitive
  effect.
- The initial `winograd` drop from the single run did not reproduce in the
  seed-repeat view; in the multi-run mean, `winograd` is higher for SemDeDup
  eps0.07 than baseline.
- eps choice matters. eps0.07 is the best tried point for aggregate CORE, while
  eps0.09 and eps0.12 remove more data and do not improve CORE or BPB.

The best current claim is therefore:

> SemDeDup eps0.07 produces a task-level redistribution signal on ClimbMix,
> especially a `commonsense_qa` gain that random-drop does not explain. It does
> not yet demonstrate a robust aggregate CORE win or a BPB/training-efficiency
> win.

## Source Reports And Artifacts

Repo reports:

- `dev/CLIMBMIX_SEMDEDUP_BRANCH_REPORT.md`: original branch report. Its result
  section is mainly the first full A/B run.
- `dev/CLIMBMIX_SEMDEDUP_TASK_DELTA_FINAL_REPORT.md`: follow-up investigation
  report covering eval-only, random-drop, seed repeats, eps sweep, and audit.
- `dev/CLIMBMIX_SEMDEDUP_CONSOLIDATED_FINAL_REPORT.md`: this consolidated
  final report.

Generated artifact:

```bash
/home/nfs/hfang/.cache/nanochat_b200_climbmix/experiments/climbmix_semdedup_task_delta/CLIMBMIX_SEMDEDUP_TASK_DELTA_INVESTIGATION.md
```

Important caveat: the generated aggregate scanner can discover smoke/plumbing
runs if pointed at broad experiment roots. This consolidated report uses only
the 15 runs that have real `final_core` and `base_eval_val_bpb` values. Smoke
runs are excluded from conclusions.

## Experiment Setup

Common setup:

- Dataset: ClimbMix parquet.
- Model: NanoChat base model, depth 24.
- Hardware target: 8x B200.
- Training budget: 6,612 iterations for full runs.
- Training tokens: 6,933,184,512 tokens per full run.
- Validation BPB: NanoChat BPB on the same held-out ClimbMix validation shard.
  This is not external Pile BPB.
- CORE: NanoChat CORE evaluation.
- Validation shard is not deduplicated.
- SemDeDup embedding model: `google/embeddinggemma-300m`.
- Main SemDeDup config: `eps=0.07`, `n_clusters=100`,
  `distance_metric=cosine`, `which_to_keep=hard`.

What was completed:

| Stage | Completed | Notes |
| --- | ---: | --- |
| Initial baseline vs SemDeDup A/B | yes | One full run per arm |
| Eval-only repeats | yes | 3 repeats each for existing baseline and SemDeDup checkpoints |
| Full seed repeats | yes | Baseline and SemDeDup eps0.07 for seeds 43 and 44 |
| Random-drop control | yes | Drops same 291,374 docs as eps0.07 SemDeDup |
| Random-drop sensitivity | yes | Additional random-drop seed 9002, train seed 42 |
| SemDeDup eps sweep | yes | eps 0.05, 0.07, 0.09, 0.12 |
| Removed-sample audit | yes | 1000 removed docs vs 1000 kept docs |

## Run Inventory

Only valid full/eval runs are included here. Smoke runs with no final metrics
are excluded.

| Arm | Stage | Seed | CORE | Val BPB | Run |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | initial A/B | anchor | 0.2687 | 0.708492 | `full-n170-r9p5-eps0p07-20260611T163000Z_baseline` |
| baseline | full_seed_repeats | 43 | 0.2729 | 0.708412 | `full-baseline-seed43-n170-i6612` |
| baseline | full_seed_repeats | 44 | 0.2779 | 0.708290 | `full-baseline-seed44-n170-i6612` |
| semdedup eps0.07 | initial A/B | anchor | 0.2775 | 0.709092 | `full-n170-r9p5-eps0p07-20260612T080000Z_semdedup` |
| semdedup eps0.07 | full_seed_repeats | 43 | 0.2766 | 0.708914 | `full-semdedup-eps0p07-seed43-n170-i6612` |
| semdedup eps0.07 | full_seed_repeats | 44 | 0.2807 | 0.708770 | `full-semdedup-eps0p07-seed44-n170-i6612` |
| semdedup eps0.07 | eps_sweep repeat | 43 | 0.2777 | 0.708886 | `full-semdedup-eps0p07-seed43-n170-i6612` |
| semdedup eps0.07 | eps_sweep repeat | 44 | 0.2710 | 0.708959 | `full-semdedup-eps0p07-seed44-n170-i6612` |
| randomdrop seed9001 | randomdrop_control | 42 | 0.2713 | 0.708628 | `full-randomdrop-primary-rdseed9001-seed42-n170-i6612` |
| randomdrop seed9001 | randomdrop_control | 43 | 0.2799 | 0.708531 | `full-randomdrop-primary-rdseed9001-seed43-n170-i6612` |
| randomdrop seed9001 | randomdrop_control | 44 | 0.2687 | 0.708423 | `full-randomdrop-primary-rdseed9001-seed44-n170-i6612` |
| randomdrop seed9002 | randomdrop_control | 42 | 0.2710 | 0.708680 | `full-randomdrop-sensitivity-rdseed9002-seed42-n170-i6612` |
| semdedup eps0.05 | eps_sweep | 42 | 0.2717 | 0.708909 | `full-semdedup-eps0p05-seed42-n170-i6612` |
| semdedup eps0.09 | eps_sweep | 42 | 0.2656 | 0.709291 | `full-semdedup-eps0p09-seed42-n170-i6612` |
| semdedup eps0.12 | eps_sweep | 42 | 0.2719 | 0.709772 | `full-semdedup-eps0p12-seed42-n170-i6612` |

Note: the two eps0.07 "repeat" rows share seed values with earlier eps0.07
runs but came from separate completed run directories. The fact that they are
not identical is itself useful evidence: training is not fully deterministic
even after adding seed controls.

## Initial Phase 4 A/B Result

The first full baseline vs SemDeDup comparison used the same 6,612-iteration
training budget.

| Metric | No SemDeDup | SemDeDup eps0.07 | Delta |
| --- | ---: | ---: | ---: |
| Train docs | 14,386,176 | 14,094,802 | -291,374 |
| Train tokens | 9,158,993,442 | 8,948,462,810 | -210,530,632 |
| Doc keep ratio | 1.0000 | 0.979746 | -0.020254 |
| Token keep ratio | 1.0000 | 0.977014 | -0.022986 |
| Dedup runtime sec | - | 4,692.98 | - |
| Train iterations | 6,612 | 6,612 | 0 |
| Train tokens consumed | 6,933,184,512 | 6,933,184,512 | 0 |
| Final train-time val BPB | 0.711159 | 0.711762 | +0.000603 |
| Base eval val BPB | 0.708492 | 0.709092 | +0.000600 |
| Final CORE | 0.2687 | 0.2775 | +0.0088 |

Interpretation of the first A/B run:

- SemDeDup improved CORE in that one run.
- SemDeDup slightly worsened BPB.
- Because lower BPB is better, this did not support "same BPB in fewer
  iterations."
- The CORE gain was highly non-uniform across tasks, motivating the follow-up
  investigation.

## Multi-Run Aggregate

This table includes only valid full/eval runs with final CORE and BPB.

| Arm | Runs | Seeds | CORE mean | CORE std | Val BPB mean | Keep tokens | Removed docs |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3 | anchor, 43, 44 | 0.2732 | 0.0046 | 0.708398 | - | - |
| randomdrop seed9001 | 3 | 42, 43, 44 | 0.2733 | 0.0059 | 0.708527 | 0.979694 | 291,374 |
| randomdrop seed9002 | 1 | 42 | 0.2710 | - | 0.708680 | 0.979765 | 291,374 |
| semdedup eps0.05 | 1 | 42 | 0.2717 | - | 0.708909 | 0.986787 | 189,542 |
| semdedup eps0.07 | 5 | anchor, 43, 44 | 0.2767 | 0.0035 | 0.708924 | 0.977014 | 291,374 |
| semdedup eps0.09 | 1 | 42 | 0.2656 | - | 0.709291 | 0.960974 | 435,382 |
| semdedup eps0.12 | 1 | 42 | 0.2719 | - | 0.709772 | 0.921398 | 786,367 |

Key reading:

- SemDeDup eps0.07 has the highest aggregate CORE mean among these groups, but
  the mean advantage over baseline is small: about +0.0035.
- Random-drop seed9001 has almost the same aggregate CORE as baseline:
  0.2733 vs 0.2732.
- BPB is worse for every SemDeDup eps group than baseline.
- More aggressive deduplication worsens BPB monotonically in this sweep.
- CORE is non-monotonic with eps: eps0.07 is best, eps0.09 is worst, eps0.12
  partially recovers but remains BPB-worse.

## Eval-Only Stability

Eval-only repeats on existing checkpoints were stable:

| Arm | Repeats | CORE mean | CORE std | Val BPB mean | Val BPB std |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3 | 0.2687 | 0.0000 | 0.708492 | 0.000000 |
| semdedup | 3 | 0.2775 | 0.0000 | 0.709092 | 0.000000 |

This rules out repeated evaluation noise for the original checkpoint pair. It
does not rule out training variance, because full training runs are not fully
deterministic.

## Task-Level Results

### Focus Tasks

| Arm | commonsense_qa | winograd | winogrande |
| --- | ---: | ---: | ---: |
| baseline | 0.0599 | 0.3651 | 0.1629 |
| randomdrop seed9001 | 0.0773 | 0.3529 | 0.1450 |
| randomdrop seed9002 | 0.0459 | 0.3846 | 0.1208 |
| semdedup eps0.05 | 0.1308 | 0.3114 | 0.1365 |
| semdedup eps0.07 | 0.1632 | 0.3934 | 0.1252 |
| semdedup eps0.09 | 0.0274 | 0.4359 | 0.1539 |
| semdedup eps0.12 | 0.1011 | 0.3480 | 0.1555 |

The main Sarah question was why there is a large gap in `commonsense_qa` and a
large movement in `winograd` / `winogrande`.

Current answer:

- `commonsense_qa`: the gain is the strongest task-level signal. SemDeDup
  eps0.07 is much higher than baseline and random-drop, so this is not fully
  explained by removing the same number of documents.
- `winogrande`: the drop is real in SemDeDup eps0.07, but random-drop also
  weakens it. This points to a generic data-removal or seed-sensitive effect,
  not a clean SemDeDup-specific regression.
- `winograd`: the initial single-run drop did not reproduce in the multi-run
  mean. SemDeDup eps0.07 is higher than baseline in the aggregate focus-task
  table, so the first `winograd` drop should not be treated as stable.

### Largest SemDeDup eps0.07 Task Movements vs Baseline

Mean task deltas across available valid runs:

| Task | Baseline | SemDeDup eps0.07 | Random-drop seed9001 | SemDeDup - Baseline |
| --- | ---: | ---: | ---: | ---: |
| commonsense_qa | 0.0599 | 0.1632 | 0.0773 | +0.1033 |
| winograd | 0.3651 | 0.3934 | 0.3529 | +0.0283 |
| piqa | 0.4864 | 0.5034 | 0.4940 | +0.0170 |
| agi_eval_lsat_ar | 0.0743 | 0.0880 | 0.0960 | +0.0138 |
| arc_challenge | 0.2052 | 0.2159 | 0.1881 | +0.0108 |
| winogrande | 0.1629 | 0.1252 | 0.1450 | -0.0377 |
| bigbench_cs_algorithms | 0.4179 | 0.3868 | 0.4250 | -0.0311 |
| bigbench_operators | 0.1889 | 0.1648 | 0.1698 | -0.0241 |
| squad | 0.4680 | 0.4520 | 0.4615 | -0.0160 |

The positive side is dominated by `commonsense_qa`; the negative side is more
spread across `winogrande`, algorithmic reasoning, operator reasoning, and
SQuAD.

## Removed-Sample Audit

Audit sample size: 1000 removed docs and 1000 kept docs.

| Group | Docs | QA-like | Coref-like | Boilerplate-like | p50 chars | p95 chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| removed | 1000 | 0.4330 | 0.9020 | 0.0570 | 3040 | 8668 |
| kept | 1000 | 0.5150 | 0.9170 | 0.0230 | 2399 | 7130 |

Interpretation:

- Removed docs are longer than kept docs.
- Removed docs are more boilerplate-like.
- Removed docs are less QA-like by this simple heuristic.
- Coreference-like heuristic rates are high for both groups and not strongly
  separated.

This is directionally consistent with SemDeDup removing longer, templated, or
boilerplate-ish duplicate content. It may also explain why QA-like downstream
tasks can improve: the dedup process may be preserving proportionally more
QA-like material while removing other repeated long-form material. That is only
a hypothesis; the heuristic audit is not a causal proof.

## Wider Insights

### 1. BPB And CORE Diverge

SemDeDup eps0.07 improves or preserves some downstream task scores but worsens
ClimbMix validation BPB. This means BPB and CORE are measuring different
effects:

- BPB tracks held-out ClimbMix compression.
- CORE tracks downstream task behavior.

For this experiment, SemDeDup does not improve the language-modeling objective
on the held-out ClimbMix shard, but it may change the composition of training
examples in a way that helps selected downstream tasks.

### 2. The Aggregate CORE Win Is Small

The eps0.07 CORE mean is 0.2767 vs baseline 0.2732. That is only about +0.0035,
while baseline and random-drop standard deviations are around 0.0046-0.0059.
This is why aggregate CORE should be described as inconclusive.

The task-level movements are much larger than the aggregate movement. That is
the real story.

### 3. Random-Drop Is An Important Control

Random-drop seed9001 matches SemDeDup's removed document count but does not
replicate the large `commonsense_qa` gain. This makes the `commonsense_qa`
signal more interesting.

However, random-drop does weaken `winogrande`, so the `winogrande` drop cannot
yet be attributed cleanly to semantic deduplication itself.

### 4. Training Is Not Fully Deterministic

Eval-only repeats are deterministic, but full training repeats with the same
seed are not identical. This is visible in the repeated eps0.07 runs with seeds
43 and 44.

Likely contributors include distributed training, GPU kernel nondeterminism,
FP8 behavior, dataloader ordering details, or run-to-run environment variation.
This does not invalidate the experiment, but it means future conclusions should
use multiple independent runs rather than assuming `--seed` gives exact
reproducibility.

### 5. More Dedup Is Not Better

The eps sweep shows that stronger SemDeDup thresholds remove more data but do
not improve quality:

- eps0.05: removes 189,542 docs, CORE 0.2717, BPB 0.708909.
- eps0.07: removes 291,374 docs, CORE 0.2767, BPB 0.708924.
- eps0.09: removes 435,382 docs, CORE 0.2656, BPB 0.709291.
- eps0.12: removes 786,367 docs, CORE 0.2719, BPB 0.709772.

eps0.07 is the best tried threshold for CORE, but it still worsens BPB. eps0.09
and eps0.12 look too aggressive.

### 6. The Removal Rate Is Modest

At eps0.07, SemDeDup removes only about 2.3% of train tokens. This is not a
large compression of the dataset. If the goal is runtime or training-cost
reduction, this specific setting is unlikely to move the needle by itself.

Also, the training run consumed the same number of tokens in both arms. The
experiment tests quality at fixed compute, not cost reduction from shorter
training.

### 7. SemDeDup Removes Longer-Than-Average Documents

SemDeDup eps0.07 removes 2.0% of documents but 2.3% of tokens. Random-drop with
the same document count removes about 2.0% of tokens. The audit also shows
removed docs have higher p50 and p95 character lengths.

This length skew may matter for tasks such as `winogrande`, `squad`, and other
tasks that benefit from narrative or long-context patterns.

### 8. There May Be A Domain/Source Effect

The task-level redistribution could come from source skew: SemDeDup may remove
repeated material from some sources more than others. The current audit has
basic source counts, but the final report does not yet map sources to task-like
domains.

This is a promising next analysis because it could explain why
`commonsense_qa` improves while `winogrande` weakens.

### 9. The Current Aggregate Script Needs Filtering

When scanning broad experiment roots, the aggregate script can count smoke runs
that have `run_summary.json` but no real final metrics. This can distort
metadata averages such as keep ratio. The clean analysis in this report filters
to runs with final CORE and BPB.

Future reporting should either:

- pass only explicit full run directories; or
- update the aggregator to exclude runs without final metrics from arm-level
  summaries.

## Answer To The Original Hypotheses

### Do we reach the same BPB/CORE in fewer iterations after SemDeDup?

Not shown.

The current fixed-iteration runs do not show a BPB efficiency win. SemDeDup did
not reach the baseline final BPB target in the initial A/B run, and its BPB is
slightly worse across the follow-up runs.

For CORE, the answer is also not established. The runs are fixed-budget rather
than early-stopping-to-target experiments, and the aggregate CORE gain is small.

### Do we get better BPB/CORE at constant iterations?

At constant iterations:

- BPB: no. SemDeDup is slightly worse.
- CORE: maybe, but weak at aggregate level. eps0.07 has the best aggregate CORE
  mean, but the advantage is small relative to variation.
- Task-level CORE: yes for some tasks, especially `commonsense_qa`; no for
  others, especially `winogrande` and some algorithm/operator tasks.

## Recommendations

For team communication:

- Report that the full experiment pipeline is working and that the follow-up
  controls were run.
- Do not claim SemDeDup is clearly better overall.
- Do not claim training-efficiency or BPB improvement.
- Present eps0.07 as an interesting threshold with task-level tradeoffs.
- Highlight `commonsense_qa` as the strongest positive signal.
- Highlight `winogrande` as unresolved because random-drop also weakens it.
- Say `winograd` is not a stable negative finding after seed repeats.

For next experiments:

- Run more independent seeds for baseline, SemDeDup eps0.07, and random-drop
  using the same seed set.
- Use confidence intervals or bootstrap over CORE examples, especially for
  `commonsense_qa`, `winograd`, and `winogrande`.
- Run a narrower eps sweep around 0.06-0.08 if eps0.07 remains interesting.
- Add a source/domain distribution audit for kept vs removed docs.
- Manually label a larger sample of removed pairs to estimate false positive
  removals.
- Compare against an exact-dedup or near-dedup control to separate "duplicate
  cleanup" from semantic-cluster selection effects.
- If the goal is efficiency, run early-stop experiments to a BPB or CORE target
  instead of only fixed-iteration experiments.
- Try another dataset, such as FineWeb-Edu or another mixture, to see whether
  the `commonsense_qa` vs `winogrande` tradeoff is specific to ClimbMix.

## Final Takeaway

SemDeDup on ClimbMix is not a simple win/loss story. The robust finding is that
it changes which capabilities improve and which degrade. eps0.07 appears to
remove a small amount of long, more boilerplate-like duplicate content and
creates a notable `commonsense_qa` improvement that random-drop does not
explain. But aggregate CORE is only slightly higher, BPB is worse, and some
tasks weaken. The right conclusion is a cautious, useful signal: SemDeDup is
worth further targeted study, but this result is not yet strong enough to claim
general quality or efficiency improvement.
