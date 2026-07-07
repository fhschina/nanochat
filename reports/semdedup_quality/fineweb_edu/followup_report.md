# FineWeb-EDU SemDeDup Follow-up Report

Report date: 2026-07-07

This follow-up extends the FineWeb-EDU SemDeDup quality report with eps
sweeps, five-seed expansion, random-drop controls, BoolQ diagnostics, and
removed-sample audits. It should be read together with `final_report.md`, which
documents the original eps0.07 three-seed result and setup.

Manual precision labeling is not completed. Heuristic audits and pair audit
label sheets are triage artifacts only.

## Executive Summary

FineWeb-EDU SemDeDup remains promising, but the best threshold in this follow-up
is lower than the original eps0.07 setting.

- The promotion rule selected `eps=0.03`: compared with eps0.07 seed42, it
  improved BPB by at least 0.0002 while keeping CORE within the allowed window.
- The selected eps0.03 removes 506,761 train documents and 591.98M train tokens,
  less than eps0.07's 633,070 documents and 775.61M tokens.
- Across the five-seed promoted run, eps0.03 has mean validation BPB `0.750329`
  and mean CORE `0.2477`.
- Relative to matched full baseline seeds 43-46, eps0.03 improves BPB on every
  seed. CORE is mixed: positive on seeds 43/44, neutral on 46, and negative on
  45.
- The random-drop controls do not obviously dominate eps0.03 SemDeDup. The
  five-seed doc-matched random-drop at the original eps0.07 removal count has
  mean BPB `0.750884` and mean CORE `0.2422`.
- BoolQ remains noisy and task-specific: promoted SemDeDup improves BoolQ on
  seeds 43/44/45 and worsens it on 42/46.
- Pair audit sheets and removed/kept heuristic audits were generated for eps
  0.03, 0.05, 0.07, 0.09, and 0.10, but no human precision estimate should be
  claimed from them yet.

The best current claim is: for FineWeb-EDU at this budget, eps0.03 is the most
attractive SemDeDup setting tested so far, with consistent BPB improvement over
matched full baselines and competitive CORE, but downstream task behavior and
manual false-positive risk still need review.

## Dataset And Setup

| Item | Value |
| --- | --- |
| Dataset | `karpathy/fineweb-edu-100b-shuffle` |
| Data root | `$HOME/.cache/nanochat_b200_fineweb_edu` |
| Original quality root | `$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_quality` |
| Follow-up root | `$NANOCHAT_BASE_DIR/experiments/fineweb_edu_semdedup_followup` |
| Validation shard | `shard_01822.parquet`, not deduplicated |
| Train shards | 170 |
| Model | NanoChat base model |
| Depth | 24 |
| Hardware target | 8x B200 |
| Device batch size | 16 |
| Training horizon | 6,612 iterations |
| Context length | 2048 |
| Primary convergence metric | Validation BPB, lower is better |
| Downstream metric | CORE, higher is better |
| Base-only scope | No SFT or RL |

## Follow-up Matrix

The follow-up ran or reused the following matrix:

- Reused the original eps0.07 baseline/SemDeDup quality artifacts for seeds
  42/43/44.
- Ran eps sweep for eps 0.03, 0.05, 0.09, and 0.10 at seed42.
- Promoted eps0.03 by the predefined BPB/CORE rule.
- Added baseline seeds 45/46.
- Added promoted eps0.03 SemDeDup seeds 43/44/45/46, plus the sweep seed42.
- Ran doc-matched random-drop controls for the original eps0.07 removed-doc
  count across training seeds 42-46.
- Ran one doc-matched random-drop sensitivity seed with random-drop seed 9002.
- Ran token-matched random-drop for the original eps0.07 removed-token count at
  seed42.
- Because the promoted eps was not 0.07, ran an additional doc-matched
  random-drop using the promoted eps0.03 removed-doc count.
- Generated heuristic removed/kept audits and pair audit label sheets for eps
  0.03, 0.05, 0.07, 0.09, and 0.10.

## EPS Sweep And Promotion

Promotion rule: versus eps0.07 seed42, require BPB improvement >= 0.0002 and
CORE no worse than 0.005; choose the lowest BPB, with ties within 0.0001 going
to fewer removed tokens.

| EPS | Seed | Val BPB | CORE | Removed docs | Removed tokens | Eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.07 | 42 | 0.750224 | 0.2442 | 633,070 | 775,605,001 | reference |
| 0.03 | 42 | 0.749828 | 0.2497 | 506,761 | 591,983,518 | yes |
| 0.05 | 42 | 0.750040 | 0.2385 | 564,673 | 674,170,865 | no |
| 0.09 | 42 | 0.750236 | 0.2648 | 745,890 | 949,184,575 | no |
| 0.10 | 42 | 0.750802 | 0.2587 | 831,176 | 1,075,119,963 | no |

eps0.03 is the only tested sweep point that passes the promotion rule.

## Seed Expansion

The matched full-run comparison is strongest for seeds 43-46. Seed42 in the
generated aggregate has both pilot and full historical artifacts available, so
the generated table should not be used as the canonical paired seed42 delta.

| Seed | Baseline run | Promoted SemDeDup run | Delta BPB | Delta CORE |
| ---: | --- | --- | ---: | ---: |
| 43 | `repeat-20260624T000845Z-seed43-n170-r9p5-eps0p07_baseline` | `full-semdedup-eps0p03-seed43-n170-r9p5` | -0.000503 | +0.0082 |
| 44 | `repeat-20260624T000845Z-seed44-n170-r9p5-eps0p07_baseline` | `full-semdedup-eps0p03-seed44-n170-r9p5` | -0.000319 | +0.0038 |
| 45 | `full-baseline-seed45-n170-r9p5` | `full-semdedup-eps0p03-seed45-n170-r9p5` | -0.000492 | -0.0068 |
| 46 | `full-baseline-seed46-n170-r9p5` | `full-semdedup-eps0p03-seed46-n170-r9p5` | -0.000661 | 0.0000 |

BPB improves on all four matched full seeds. CORE has small seed-level movement
with mixed signs.

## Arm Summary

| Arm | Runs | Seeds | Val BPB mean | CORE mean | Removed docs | Removed tokens |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| promoted SemDeDup eps0.03 | 5 | 42, 43, 44, 45, 46 | 0.750329 | 0.2477 | 506,761 | 591,983,518 |
| SemDeDup eps0.05 | 1 | 42 | 0.750040 | 0.2385 | 564,673 | 674,170,865 |
| SemDeDup eps0.07 reference | 6 discovered | 42, 43, 44 | 0.800211 | 0.2110 | 633,070 seed42 reference | 775,605,001 seed42 reference |
| SemDeDup eps0.09 | 1 | 42 | 0.750236 | 0.2648 | 745,890 | 949,184,575 |
| SemDeDup eps0.10 | 1 | 42 | 0.750802 | 0.2587 | 831,176 | 1,075,119,963 |
| random-drop doc-matched original removal | 5 | 42, 43, 44, 45, 46 | 0.750884 | 0.2422 | 633,070 | 647,840,433 |
| random-drop token-matched original removal | 1 | 42 | 0.750554 | 0.2492 | 756,313 | 775,604,971 |
| random-drop promoted-doc removal | 1 | 42 | 0.750486 | 0.2437 | 506,761 | 518,165,985 |

The eps0.07 aggregate row in the generated report mixes historical, pilot, and
repeat artifacts discovered under the quality root; use `final_report.md` for
the canonical eps0.07 three-seed analysis.

## BoolQ Analysis

| Seed | Baseline BoolQ | Promoted SemDeDup BoolQ | Delta |
| ---: | ---: | ---: | ---: |
| 42 | -0.1000 | -0.1500 | -0.0500 |
| 43 | -0.1838 | -0.1581 | +0.0258 |
| 44 | -0.1621 | -0.0993 | +0.0628 |
| 45 | -0.0768 | -0.0486 | +0.0282 |
| 46 | -0.0639 | -0.1379 | -0.0740 |

Removed/kept BoolQ-like heuristic rates vary by eps and are included in
`followup_report.generated.md`. They should be treated as triage only.

## Random-drop Controls

The controls separate semantic selection from generic data removal:

- Doc-matched random drop at the original eps0.07 removed-doc count:
  five-seed mean BPB `0.750884`, CORE `0.2422`.
- Random-drop seed sensitivity at the same removed-doc count:
  seed42 BPB `0.750518`, CORE `0.2354`.
- Token-matched random drop at the original eps0.07 removed-token count:
  seed42 BPB `0.750554`, CORE `0.2492`.
- Promoted eps0.03 removed-doc matched random drop:
  seed42 BPB `0.750486`, CORE `0.2437`.

The token-matched control is close to eps0.03 on CORE at seed42 but worse on
BPB. The doc-matched five-seed control is worse than eps0.03 on both mean BPB
and mean CORE.

## Audit Artifacts

Heuristic removed/kept audits and pair audit label sheets were generated for
all tested eps values.

| EPS | Removed/kept audit | Pair audit records | Label sheet |
| ---: | --- | ---: | --- |
| 0.03 | generated | 300 | `audits/eps0p03_pair_audit_label_sheet.csv` |
| 0.05 | generated | 100 | `audits/eps0p05_pair_audit_label_sheet.csv` |
| 0.07 | generated | 100 | `audits/eps0p07_pair_audit_label_sheet.csv` |
| 0.09 | generated | 100 | `audits/eps0p09_pair_audit_label_sheet.csv` |
| 0.10 | generated | 100 | `audits/eps0p10_pair_audit_label_sheet.csv` |

Labels are intentionally blank for human review. The expected label set is:
`exact_duplicate`, `near_duplicate`, `semantic_duplicate`,
`same_template_different_content`, `related_but_both_valuable`,
`false_positive`, and `unclear`.

## Runtime

Training runs took roughly 5,155-5,164 seconds per full 6,612-step run on 8x
B200, about 1.34M tokens/sec. The eps0.03 staged SemDeDup preprocessing time in
the aggregate report is about 420 seconds after reusing available embeddings and
staged artifacts; higher eps sweep values required longer preprocessing in the
follow-up run.

GPU monitor summaries are in the generated report. Existing historical runs
without monitor data are marked unavailable there.

## Interpretation

The original eps0.07 setting removed a large amount of FineWeb-EDU and showed a
small but promising three-seed result. This follow-up suggests that a less
aggressive eps0.03 threshold preserves more data while improving the seed42 BPB
selection criterion and maintaining competitive CORE.

The random-drop controls reduce the chance that the result is only "less data
is better." The promoted SemDeDup arm has better five-seed mean BPB and CORE
than the doc-matched random-drop control at the original eps0.07 removal count.
However, the token-matched seed42 control is close enough that the conclusion
should remain cautious until more token-matched seeds or manual audit labels are
available.

## Limitations

- Manual precision labeling is not completed.
- The generated aggregate discovers pilot and historical artifacts, so some
  generated aggregate rows need manual interpretation.
- Token-matched random-drop is only run for seed42 unless the threshold for
  expansion is triggered.
- Pair audit data may degrade to removed-only evidence where root pair
  information cannot be recovered from Curator artifacts.
- Runtime comparisons are fixed-budget training runs, not early-stop efficiency
  claims.

## Artifact Inventory

Repo artifacts:

- `reports/semdedup_quality/fineweb_edu/followup_report.md`: this curated
  follow-up report.
- `reports/semdedup_quality/fineweb_edu/followup_report.generated.md`: raw
  generated aggregate report.
- `reports/semdedup_quality/fineweb_edu/followup_report.json`: machine-readable
  aggregate summary.
- `reports/semdedup_quality/fineweb_edu/audits/*_pair_audit_label_sheet.csv`:
  human labeling sheets.

Generated run artifacts:

```text
$HOME/.cache/nanochat_b200_fineweb_edu/experiments/fineweb_edu_semdedup_followup/
```

Implementation entry points:

- `runs/fineweb_edu_task_delta_investigation_b200.sh`
- `scripts/aggregate_fineweb_edu_followup.py`
- `scripts/analyze_removed_samples.py`
- `scripts/extract_semdedup_pair_audit.py`
- `scripts/build_random_drop_dataset.py`
