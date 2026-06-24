# FineWeb-EDU SemDeDup Final Report

Report date: 2026-06-24

This is the canonical FineWeb-EDU report for the SemDeDup quality experiment. It uses the same section structure as the ClimbMix final report so the two datasets can be compared directly. Supporting details remain in `repeats_report.md` and `runbook.md`.

## Executive Summary

SemDeDup eps0.07 on FineWeb-EDU looks more favorable than on ClimbMix for BPB, but the quality conclusion is still cautious.

- Three paired full-run seeds completed: 42, 43, and 44.
- SemDeDup removes 633,070 train documents and 775.6M train tokens: 6.99% of docs and 8.33% of tokens.
- BPB improves consistently under SemDeDup. The paired mean delta is -0.000624 final train validation BPB and -0.000498 base-eval validation BPB, where lower is better.
- CORE improves slightly on average: +0.0031 paired mean, but with standard deviation 0.0059 across three seeds.
- Training runtime is lower on average by 454 seconds, but both arms train for the same 6,612 iterations and consume the same 6.933B tokens. This is not an early-stop efficiency result.
- The biggest per-task negative signal is `boolq`. The biggest positive signals include `agi_eval_lsat_ar`, `winograd`, and `copa`.
- Removed/kept sample files exist, but manual labeling has not been completed. Random-drop, eps sweep, and order-preserving controls have not been run for FineWeb-EDU yet.

The best current claim is: FineWeb-EDU SemDeDup eps0.07 improves BPB and has a small positive average CORE delta across three seeds, but task-level variance and missing controls mean it should be treated as promising rather than final.

## Dataset And Setup

| Item | Value |
| --- | --- |
| Dataset | `karpathy/fineweb-edu-100b-shuffle` |
| Data root | `$HOME/.cache/nanochat_b200_fineweb_edu` |
| Run root | `$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality` |
| Validation shard | `shard_01822.parquet`, not deduplicated |
| Train shards | 170 |
| Model | NanoChat base model |
| Depth | 24 |
| Hardware target | 8x B200 |
| Device batch size | 16 |
| Training horizon | 6,612 iterations |
| Training tokens consumed | 6,933,184,512 per full run |
| Primary convergence metric | Validation BPB, lower is better |
| Downstream metric | CORE, higher is better |
| Eval modes | `core,bpb,sample` |
| Base-only scope | No SFT or RL |

## A/B Design

The primary comparison is:

- FineWeb-EDU baseline: original selected train shards.
- FineWeb-EDU SemDeDup: same source shards after semantic deduplication.

Held fixed across arms:

- dataset source and validation shard;
- tokenizer and token-byte mapping;
- model depth and training budget;
- base pretraining and base evaluation only;
- CORE evaluation settings.

The final report uses three paired seeds. Pilot and smoke runs are excluded from the reported metrics.

## SemDeDup Configuration

| Parameter | Value |
| --- | --- |
| Backend | NeMo Curator semantic deduplication |
| Embedding model | `google/embeddinggemma-300m` |
| eps | 0.07 |
| Cosine similarity cutoff | 0.93 |
| n clusters | 100 |
| Distance metric | cosine |
| Which to keep | hard |
| Pairwise batch size | 1024 |
| vLLM attention backend | `TRITON_ATTN` |
| vLLM eager mode | enabled |

The validation shard is copied through unchanged and is not deduplicated.

## Data Reduction Summary

| Metric | Baseline | SemDeDup eps0.07 | Delta |
| --- | ---: | ---: | ---: |
| Train docs | 9,060,352 | 8,427,282 | -633,070 |
| Train tokens | 9,309,977,010 | 8,534,372,009 | -775,605,001 |
| Doc keep ratio | 1.000000 | 0.930127 | -0.069873 |
| Token keep ratio | 1.000000 | 0.916691 | -0.083309 |
| Dedup runtime sec | - | 875.11 | - |

FineWeb-EDU has a much larger duplicate-like mass than ClimbMix at the same eps0.07 cutoff.

## Similarity ECDF

![SemDeDup similarity ECDF](semdedup_similarity_ecdf.svg)

| ECDF statistic | Value |
| --- | ---: |
| total documents | 9,060,352 |
| eps | 0.070000 |
| similarity threshold | 0.930000 |
| documents below threshold | 8,427,282 |
| removed documents | 633,070 |
| removed ratio | 0.069873 |
| raw max similarity | 1.000002 |
| raw >1.0 documents | 174,879 |
| raw >1.0 ratio | 0.019302 |
| raw ==1.0 documents | 34,696 |
| sim >=0.999999 documents | 362,218 |
| sim >=0.999999 ratio | 0.039978 |

The x-axis is cosine similarity. With cosine distance, eps maps to similarity cutoff `1 - eps`. Raw Curator scores can be slightly above 1.0 from floating-point roundoff; the plotted ECDF clips scores to `[0, 1]`.

## Training Results

All paired full runs used the same fixed 6,612-iteration budget.

| Seed | Baseline run | SemDeDup run | Baseline BPB | SemDeDup BPB | Delta BPB | Baseline CORE | SemDeDup CORE | Delta CORE |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `full-n170-r9p5-eps0p07-20260622T192524Z_baseline` | `full-n170-r9p5-eps0p07-20260622T192524Z_semdedup` | 0.751186 | 0.750783 | -0.000403 | 0.246800 | 0.244200 | -0.002600 |
| 43 | `repeat-20260624T000845Z-seed43-n170-r9p5-eps0p07_baseline` | `repeat-20260624T000845Z-seed43-n170-r9p5-eps0p07_semdedup` | 0.751378 | 0.750818 | -0.000560 | 0.238000 | 0.247100 | +0.009100 |
| 44 | `repeat-20260624T000845Z-seed44-n170-r9p5-eps0p07_baseline` | `repeat-20260624T000845Z-seed44-n170-r9p5-eps0p07_semdedup` | 0.751841 | 0.750931 | -0.000910 | 0.244400 | 0.247200 | +0.002800 |

Reading:

- BPB improves for all three seeds.
- CORE improves for two of three seeds.
- The average CORE delta is positive, but the seed-to-seed spread is larger than the mean.

## Multi-Seed / Repeat Results

| Metric | Better | Baseline mean +/- sd | SemDeDup mean +/- sd | Paired delta mean +/- sd |
| --- | --- | ---: | ---: | ---: |
| min val BPB | lower | 0.751468 +/- 0.000337 | 0.750844 +/- 0.000077 | -0.000624 +/- 0.000260 |
| final train val BPB | lower | 0.751468 +/- 0.000337 | 0.750844 +/- 0.000077 | -0.000624 +/- 0.000260 |
| base eval val BPB | lower | 0.750794 +/- 0.000327 | 0.750296 +/- 0.000076 | -0.000498 +/- 0.000251 |
| final CORE | higher | 0.243067 +/- 0.004549 | 0.246167 +/- 0.001704 | +0.003100 +/- 0.005856 |
| train runtime sec | lower | 5822.800000 +/- 440.407493 | 5368.400000 +/- 71.388234 | -454.400000 +/- 406.820059 |
| train iterations | lower | 6612.000000 +/- 0.000000 | 6612.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| train tokens | lower | 6933184512.000000 +/- 0.000000 | 6933184512.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| steps to paired baseline BPB | lower | 6612.000000 +/- 0.000000 | 6612.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| runtime sec to paired baseline BPB | lower | 5822.800000 +/- 440.407493 | 5368.400000 +/- 71.388234 | -454.400000 +/- 406.820059 |

The BPB improvement is consistent. The CORE improvement is positive on average but not yet robust enough to claim a broad downstream quality win.

## Per-Task CORE Delta

Task deltas are paired means across the three completed seeds.

| Task | N | Delta mean | Delta sd | Min delta | Max delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| boolq | 3 | -0.100864 | 0.067870 | -0.162563 | -0.028167 |
| openbook_qa | 3 | -0.000889 | 0.021388 | -0.021333 | 0.021333 |
| hellaswag_zeroshot | 3 | -0.000753 | 0.005818 | -0.005577 | 0.005709 |
| arc_easy | 3 | -0.000374 | 0.017033 | -0.015713 | 0.017957 |
| bigbench_cs_algorithms | 3 | -0.000252 | 0.043403 | -0.040909 | 0.045455 |
| CORE | 3 | 0.003121 | 0.005805 | -0.002522 | 0.009075 |
| squad | 3 | 0.004100 | 0.007599 | -0.004257 | 0.010596 |
| piqa | 3 | 0.004352 | 0.023615 | -0.022851 | 0.019586 |
| lambada_openai | 3 | 0.006727 | 0.007153 | -0.001359 | 0.012226 |
| bigbench_qa_wikidata | 3 | 0.007841 | 0.004742 | 0.002461 | 0.011416 |
| bigbench_repeat_copy_logic | 3 | 0.010417 | 0.018042 | 0.000000 | 0.031250 |
| copa | 3 | 0.033333 | 0.030551 | 0.000000 | 0.060000 |
| winograd | 3 | 0.043956 | 0.045751 | -0.007326 | 0.080586 |
| agi_eval_lsat_ar | 3 | 0.045290 | 0.033207 | 0.016304 | 0.081522 |

The large `boolq` regression needs follow-up. The strongest positive movements are `agi_eval_lsat_ar`, `winograd`, and `copa`.

## Removed / Kept Sample Audit

Removed and kept sample files are produced for SemDeDup runs, but manual labels have not been completed yet.

| Audit item | Status |
| --- | --- |
| `removed_samples.jsonl` | available |
| `kept_samples.jsonl` | available |
| manual duplicate/false-positive labels | not run yet |
| heuristic kept-vs-removed summary | not run yet |

This is the main missing quality-control step for FineWeb-EDU. The ECDF shows a large exact/near-exact duplicate mass, but sample labeling is still needed to estimate false-positive risk.

## Additional Controls

| Control | Status | Result |
| --- | --- | --- |
| Random-drop same doc count | not run yet | Needed to separate semantic selection from generic data removal. |
| Random-drop sensitivity seed | not run yet | Not available. |
| eps sweep | not run yet | Only eps0.07 has completed paired formal runs. |
| Eval-only repeats | not run yet | Not available as a separate stability check. |
| Order-preserving SemDeDup | not run | FineWeb wrapper keeps outputs isolated; no ClimbMix-style shard-order follow-up has been run. |

FineWeb-EDU currently has a cleaner paired A/B result than ClimbMix, but fewer diagnostic controls.

## Interpretation

FineWeb-EDU has a larger high-similarity mass than ClimbMix, and SemDeDup removes a larger fraction of the dataset at the same eps0.07 threshold. This matches the BPB result: reducing duplicate-like training data appears to slightly improve held-out FineWeb-EDU compression.

The downstream result is more mixed. CORE is positive on average, but the mean delta is small compared with seed variance. The task table suggests that FineWeb-EDU SemDeDup changes task behavior rather than uniformly improving all tasks. `boolq` in particular should be inspected before making a strong quality claim.

## Limitations

- Only three paired seeds are available.
- Random-drop and eps-sweep controls have not been run for FineWeb-EDU.
- Removed/kept samples have not been manually labeled.
- Runtime improvements are not early-stop results; both arms run the same number of iterations and consume the same number of training tokens.
- The per-task table shows substantial task-specific variance.
- The report does not benchmark cuVS clustering performance.

## Artifact Inventory

Repo artifacts:

- `reports/semdedup_quality/fineweb_edu/final_report.md`: this canonical report.
- `reports/semdedup_quality/fineweb_edu/repeats_report.md`: generated multi-seed repeat report.
- `reports/semdedup_quality/fineweb_edu/semdedup_similarity_ecdf.svg`: ECDF plot.
- `reports/semdedup_quality/fineweb_edu/semdedup_similarity_ecdf.json`: ECDF stats.
- `reports/semdedup_quality/fineweb_edu/runbook.md`: execution runbook.

Generated run artifacts:

```text
$HOME/.cache/nanochat_b200_fineweb_edu/experiments/fineweb_edu_semdedup_quality/
```
