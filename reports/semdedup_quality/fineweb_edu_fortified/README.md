# FineWeb-EDU-Fortified deduplication

This directory is the dataset-level home for FineWeb-EDU-Fortified deduplication reports. It is parallel to `../climbmix/` and `../fineweb_edu/`; experiment-specific outputs live one level below it.

## Exact + fuzzy deduplication

- [`exact_fuzzy/final_report.md`](exact_fuzzy/final_report.md): full-corpus configuration, row/token retention, and integrity result.
- [`exact_fuzzy/exact_fuzzy_manifest.json`](exact_fuzzy/exact_fuzzy_manifest.json): machine-readable run configuration, timings, stage statistics, and integrity check.
- [`exact_fuzzy/fuzzy_audit/report.md`](exact_fuzzy/fuzzy_audit/report.md): connected-component and sampled removed-to-keeper audit summary.
- [`exact_fuzzy/fuzzy_audit/component_audit.json`](exact_fuzzy/fuzzy_audit/component_audit.json): machine-readable component statistics.
- [`exact_fuzzy/fuzzy_audit/removed_to_keeper_pairs.jsonl`](exact_fuzzy/fuzzy_audit/removed_to_keeper_pairs.jsonl): 519 sampled removed-to-keeper pairs with exact 24-character-ngram Jaccard scores.

The full run kept 204,557,770 of 322,250,000 documents (63.4780%). The exact stage found no additional exact duplicates in the already MD5-deduplicated published corpus; the fuzzy stage removed 117,692,230 documents (36.5220%).

## Reproduction code

- `runs/fineweb_edu_fortified_exact_fuzzy_b200.sh`: end-to-end materialization, exact/fuzzy deduplication, reporting, and smoke entrypoint.
- `scripts/materialize_fineweb_edu_fortified.py`: dataset materializer.
- `scripts/materialize_fineweb_edu_fortified_parallel.py`: parallel all-config materialization driver.
- `scripts/build_exact_fuzzy_dedup_dataset.py`: resumable exact/fuzzy identification, removal, statistics, and report driver.
- `scripts/make_exact_fuzzy_smoke_fixture.py`: deterministic local smoke fixture.
- `scripts/build_fuzzy_audit_artifacts.py` and `scripts/audit_fuzzy_components.py`: fuzzy-component audit preparation and analysis.

Raw parquet, Curator caches, duplicate-ID registries, and logs are intentionally not versioned. The committed reports and sampled audit artifacts are sufficient to review the aggregate result and the audited component behavior.
