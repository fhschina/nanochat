# FineWeb-EDU SemDeDup Follow-up Report

Manual precision not completed. Heuristic removed/kept audits are triage signals only.

## Status

Discovered `30` run summaries across QUALITY_ROOT and FOLLOWUP_ROOT.
Promoted eps by rule: `0.03`.

## EPS Sweep and Promotion

Promotion rule: versus eps0.07 seed42, require BPB improvement >= 0.0002 and CORE no worse than 0.005; choose lowest BPB, tie within 0.0001 goes to fewer removed tokens.

| EPS | Seed | Val BPB | CORE | Removed docs | Removed tokens | Eligible | Run |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.07 | 42 | 0.750224 | 0.2442 | 633,070 | 775,605,001 | reference | `full-n170-r9p5-eps0p07-20260622T192524Z_semdedup` |
| 0.03 | 42 | 0.749828 | 0.2497 | 506,761 | 591,983,518 | True | `full-semdedup-eps0p03-seed42-n170-r9p5` |
| 0.05 | 42 | 0.750040 | 0.2385 | 564,673 | 674,170,865 | False | `full-semdedup-eps0p05-seed42-n170-r9p5` |
| 0.09 | 42 | 0.750236 | 0.2648 | 745,890 | 949,184,575 | False | `full-semdedup-eps0p09-seed42-n170-r9p5` |
| 0.10 | 42 | 0.750802 | 0.2587 | 831,176 | 1,075,119,963 | False | `full-semdedup-eps0p10-seed42-n170-r9p5` |

## Arm Summary

| Arm | Runs | Seeds | Val BPB mean | CORE mean | Removed docs | Removed tokens |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| baseline | 6 | 42, 43, 44, 45, 46 | 0.784173 | 0.2231 | - | - |
| randomdrop_doc_matched_rd9001_target100 | 1 | 42 | - | - | 100 | - |
| randomdrop_doc_matched_rd9001_target506761 | 1 | 42 | 0.750486 | 0.2437 | 506761 | 518165985 |
| randomdrop_doc_matched_rd9001_target633070 | 5 | 42, 43, 44, 45, 46 | 0.750884 | 0.2422 | 633070 | 647840433 |
| randomdrop_doc_matched_rd9002_target633070 | 1 | 42 | 0.750518 | 0.2354 | 633070 | 651676999 |
| randomdrop_token_matched_rd9001_target20000 | 1 | 42 | - | - | 23 | 19841 |
| randomdrop_token_matched_rd9001_target775605001 | 1 | 42 | 0.750554 | 0.2492 | 756313 | 775604971 |
| semdedup_eps0.03 | 5 | 42, 43, 44, 45, 46 | 0.750329 | 0.2477 | 506761 | 591983518 |
| semdedup_eps0.05 | 1 | 42 | 0.750040 | 0.2385 | 564673 | 674170865 |
| semdedup_eps0.07 | 6 | 42, 43, 44 | 0.800211 | 0.2110 | 316823 | 582285743 |
| semdedup_eps0.09 | 1 | 42 | 0.750236 | 0.2648 | 745890 | 949184575 |
| semdedup_eps0.10 | 1 | 42 | 0.750802 | 0.2587 | 831176 | 1075119963 |

## Seed Expansion

| Seed | Baseline | Promoted SemDeDup | Delta BPB | Delta CORE |
| ---: | --- | --- | ---: | ---: |
| 42 | pilot-n8-i400-baseline | full-semdedup-eps0p03-seed42-n170-r9p5 | -0.200911 | 0.1412 |
| 43 | repeat-20260624T000845Z-seed43-n170-r9p5-eps0p07_baseline | full-semdedup-eps0p03-seed43-n170-r9p5 | -0.000503 | 0.0082 |
| 44 | repeat-20260624T000845Z-seed44-n170-r9p5-eps0p07_baseline | full-semdedup-eps0p03-seed44-n170-r9p5 | -0.000319 | 0.0038 |
| 45 | full-baseline-seed45-n170-r9p5 | full-semdedup-eps0p03-seed45-n170-r9p5 | -0.000492 | -0.0068 |
| 46 | full-baseline-seed46-n170-r9p5 | full-semdedup-eps0p03-seed46-n170-r9p5 | -0.000661 | 0.0000 |

## BoolQ Analysis

| Seed | Baseline BoolQ | Promoted SemDeDup BoolQ | Delta |
| ---: | ---: | ---: | ---: |
| 42 | -0.1000 | -0.1500 | -0.0500 |
| 43 | -0.1838 | -0.1581 | 0.0258 |
| 44 | -0.1621 | -0.0993 | 0.0628 |
| 45 | -0.0768 | -0.0486 | 0.0282 |
| 46 | -0.0639 | -0.1379 | -0.0740 |

Removed/kept BoolQ-like heuristic rates:

| Audit | Removed | Kept |
| --- | ---: | ---: |
| `removed_heuristic` | 0.1900 | 0.2130 |
| `removed_heuristic` | 0.1910 | 0.1970 |
| `removed_heuristic` | 0.2100 | 0.2220 |
| `removed_heuristic` | 0.1980 | 0.2070 |
| `removed_heuristic` | 0.2240 | 0.1910 |

## Random-drop Controls

| Run | Mode | RD seed | Train seed | Target docs | Target tokens | Val BPB | CORE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full-randomdrop-docmatched-drop633070-rdseed9001-seed42-n170-r9p5` | doc_matched | 9001 | 42 | 633,070 | - | 0.750473 | 0.2530 |
| `full-randomdrop-promoted-eps0p03-docmatched-drop506761-rdseed9001-seed42-n170-r9p5` | doc_matched | 9001 | 42 | 506,761 | - | 0.750486 | 0.2437 |
| `smoke-randomdrop-doc` | doc_matched | 9001 | 42 | 100 | - | - | - |
| `full-randomdrop-docmatched-drop633070-rdseed9001-seed43-n170-r9p5` | doc_matched | 9001 | 43 | 633,070 | - | 0.750855 | 0.2320 |
| `full-randomdrop-docmatched-drop633070-rdseed9001-seed44-n170-r9p5` | doc_matched | 9001 | 44 | 633,070 | - | 0.751300 | 0.2343 |
| `full-randomdrop-docmatched-drop633070-rdseed9001-seed45-n170-r9p5` | doc_matched | 9001 | 45 | 633,070 | - | 0.751109 | 0.2478 |
| `full-randomdrop-docmatched-drop633070-rdseed9001-seed46-n170-r9p5` | doc_matched | 9001 | 46 | 633,070 | - | 0.750684 | 0.2440 |
| `full-randomdrop-docmatched-sensitivity-drop633070-rdseed9002-seed42-n170-r9p5` | doc_matched | 9002 | 42 | 633,070 | - | 0.750518 | 0.2354 |
| `full-randomdrop-tokenmatched-droptok775605001-rdseed9001-seed42-n170-r9p5` | token_matched | 9001 | 42 | 291,374 | 775,605,001 | 0.750554 | 0.2492 |
| `smoke-randomdrop-token` | token_matched | 9001 | 42 | 291,374 | 20,000 | - | - |

## Audit Artifacts

Removed/kept heuristic audits: `5`. Pair audit sheets: `5`.

| Pair audit | Mode | Records | Degradation |
| --- | --- | ---: | --- |
| `pair_audit` | pair | 300 |  |
| `pair_audit` | pair | 100 |  |
| `pair_audit` | pair | 100 |  |
| `pair_audit` | pair | 100 |  |
| `pair_audit` | pair | 100 |  |

## Runtime

| Run | Arm | Seed | Preprocess sec | Train sec | tok/sec | step dt sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `full-n170-r9p5-eps0p07-20260622T192524Z_baseline` | baseline | 42 | - | 6157.2 | 1126028.8 | 0.931 |
| `pilot-n8-i400-baseline` | baseline | 42 | - | 307.2 | 1365333.3 | 0.768 |
| `repeat-20260624T000845Z-seed43-n170-r9p5-eps0p07_baseline` | baseline | 43 | - | 5987.4 | 1157962.5 | 0.906 |
| `repeat-20260624T000845Z-seed44-n170-r9p5-eps0p07_baseline` | baseline | 44 | - | 5323.8 | 1302300.0 | 0.805 |
| `full-baseline-seed45-n170-r9p5` | baseline | 45 | - | 5155.2 | 1344891.5 | 0.780 |
| `full-baseline-seed46-n170-r9p5` | baseline | 46 | - | 5160.6 | 1343484.2 | 0.780 |
| `full-randomdrop-docmatched-drop633070-rdseed9001-seed42-n170-r9p5` | randomdrop | 42 | - | 5158.2 | 1344109.3 | 0.780 |
| `full-randomdrop-docmatched-sensitivity-drop633070-rdseed9002-seed42-n170-r9p5` | randomdrop | 42 | - | 5155.8 | 1344735.0 | 0.780 |
| `full-randomdrop-promoted-eps0p03-docmatched-drop506761-rdseed9001-seed42-n170-r9p5` | randomdrop | 42 | - | 5156.4 | 1344578.5 | 0.780 |
| `full-randomdrop-tokenmatched-droptok775605001-rdseed9001-seed42-n170-r9p5` | randomdrop | 42 | - | 5155.2 | 1344891.5 | 0.780 |
| `full-randomdrop-docmatched-drop633070-rdseed9001-seed43-n170-r9p5` | randomdrop | 43 | - | 5159.4 | 1343796.7 | 0.780 |
| `full-randomdrop-docmatched-drop633070-rdseed9001-seed44-n170-r9p5` | randomdrop | 44 | - | 5163.0 | 1342859.7 | 0.781 |
| `full-randomdrop-docmatched-drop633070-rdseed9001-seed45-n170-r9p5` | randomdrop | 45 | - | 5161.8 | 1343171.9 | 0.781 |
| `full-randomdrop-docmatched-drop633070-rdseed9001-seed46-n170-r9p5` | randomdrop | 46 | - | 5160.6 | 1343484.2 | 0.780 |
| `full-semdedup-eps0p03-seed42-n170-r9p5` | semdedup eps0.03 | 42 | 419.9 | 5158.2 | 1344109.3 | 0.780 |
| `full-semdedup-eps0p03-seed43-n170-r9p5` | semdedup eps0.03 | 43 | 419.9 | 5158.2 | 1344109.3 | 0.780 |
| `full-semdedup-eps0p03-seed44-n170-r9p5` | semdedup eps0.03 | 44 | 419.9 | 5160.6 | 1343484.2 | 0.780 |
| `full-semdedup-eps0p03-seed45-n170-r9p5` | semdedup eps0.03 | 45 | 419.9 | 5164.2 | 1342547.6 | 0.781 |
| `full-semdedup-eps0p03-seed46-n170-r9p5` | semdedup eps0.03 | 46 | 419.9 | 5162.4 | 1343015.8 | 0.781 |
| `full-semdedup-eps0p05-seed42-n170-r9p5` | semdedup eps0.05 | 42 | 3750.2 | 5152.2 | 1345674.6 | 0.779 |
| `full-n170-r9p5-eps0p07-20260622T192524Z_semdedup` | semdedup eps0.07 | 42 | 875.1 | 5344.8 | 1297183.2 | 0.808 |
| `pilot-n8-i400-semdedup-eps0p07` | semdedup eps0.07 | 42 | 580.6 | 307.2 | 1365333.3 | 0.768 |
| `repeat-20260624T000845Z-seed43-n170-r9p5-eps0p07_semdedup` | semdedup eps0.07 | 43 | 875.1 | 5448.6 | 1272470.8 | 0.824 |
| `repeat-20260624T000845Z-seed44-n170-r9p5-eps0p07_semdedup` | semdedup eps0.07 | 44 | 875.1 | 5311.8 | 1305242.0 | 0.803 |
| `full-semdedup-eps0p09-seed42-n170-r9p5` | semdedup eps0.09 | 42 | 3794.8 | 5158.8 | 1343953.0 | 0.780 |
| `full-semdedup-eps0p10-seed42-n170-r9p5` | semdedup eps0.10 | 42 | 3750.2 | 5154.6 | 1345048.0 | 0.780 |

GPU dmon logs:
- `/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu/experiments/fineweb_edu_semdedup_followup/logs/dmon.all.20260705T034025Z.log` samples=29624 avg_sm=47.72 max_sm=100.00
- `/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu/experiments/fineweb_edu_semdedup_followup/logs/dmon.all.20260705T175443Z.log` samples=22792 avg_sm=61.87 max_sm=100.00
- `/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu/experiments/fineweb_edu_semdedup_followup/logs/dmon.all.20260705T175531Z.log` samples=22784 avg_sm=61.99 max_sm=100.00
- `/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu/experiments/fineweb_edu_semdedup_followup/logs/dmon.all.20260705T175925Z.log` samples=22752 avg_sm=62.21 max_sm=100.00
- `/home/nfs/hfang/.cache/nanochat_b200_fineweb_edu/experiments/fineweb_edu_semdedup_followup/logs/dmon.all.20260705T221924Z.log` samples=20672 avg_sm=67.49 max_sm=100.00

## Remaining Work

- No required EPS seed42 or baseline expansion summaries are missing from discovered artifacts.
