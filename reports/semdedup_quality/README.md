# SemDeDup Quality Experiments

This directory keeps repo-level SemDeDup experiment documentation and final reports organized by dataset. Raw run outputs still live under each `$NANOCHAT_BASE_DIR/experiments/...` directory.

## Datasets

- `climbmix/`: ClimbMix canonical final report, runbook, branch report, task-delta analysis, commonsense QA follow-up, consolidated historical report, and ECDF artifacts.
- `fineweb_edu/`: FineWeb-EDU canonical final report, runbook, multi-seed repeat report, and ECDF artifacts.

Use `climbmix/final_report.md` and `fineweb_edu/final_report.md` as the main side-by-side reports. The other Markdown files are supporting detail.

## Entrypoints

- `runs/climbmix_semdedup_quality_b200.sh`: ClimbMix wrapper.
- `runs/fineweb_edu_semdedup_quality_b200.sh`: FineWeb-EDU wrapper.
- `runs/semdedup_quality_b200.sh`: shared baseline/SemDeDup/random-drop runner used by dataset wrappers.

## Shared Helpers

- `scripts/build_semdedup_dataset.py`: build a SemDeDup-reduced NanoChat parquet dataset.
- `scripts/build_random_drop_dataset.py`: build a random-drop control dataset.
- `scripts/parquet_data_stats.py`: compute tokenizer-based parquet stats.
- `scripts/compare_semdedup_experiments.py`: compare baseline vs SemDeDup run directories.
