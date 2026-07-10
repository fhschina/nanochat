# FineWeb-EDU Qwen3 SemDeDup Runbook

## Scope

This runbook launches the 3-seed FineWeb-EDU SemDeDup experiment with only the embedding model changed to `qwen3-embedding-8b`.

Fixed settings:

- dataset: `karpathy/fineweb-edu-100b-shuffle`
- train shards: `170`
- validation shard: `shard_01822.parquet`, copied to `shard_00170.parquet` in reduced data dirs
- `PARAM_DATA_RATIO=9.5`
- `SEMD_EPS=0.07`
- `SEMD_N_CLUSTERS=100`
- model depth, eval settings, and training horizon inherited from `runs/fineweb_edu_semdedup_quality_b200.sh`

## Artifact Roots

- Qwen3 run root: `$HOME/.cache/nanochat_b200_fineweb_edu/experiments/fineweb_edu_qwen3_semdedup_quality`
- Existing baseline and EmbeddingGemma comparison root: `$HOME/.cache/nanochat_b200_fineweb_edu/experiments/fineweb_edu_semdedup_quality`
- Final report: `reports/semdedup_quality/fineweb_edu/qwen3_final_report.md`
- ECDF plot: `reports/semdedup_quality/fineweb_edu/qwen3_semdedup_similarity_ecdf.svg`
- ECDF stats: `reports/semdedup_quality/fineweb_edu/qwen3_semdedup_similarity_ecdf.json`

## Commands

Preflight only:

```bash
bash runs/fineweb_edu_qwen3_semdedup_nohup_b200.sh dry-run
```

Full 3-seed run:

```bash
nohup bash runs/fineweb_edu_qwen3_semdedup_nohup_b200.sh all \
  > "$HOME/.cache/nanochat_b200_fineweb_edu/experiments/fineweb_edu_qwen3_semdedup_quality/qwen3_fwe.log" 2>&1 &
```

Seed 42 only, including Qwen3 SemDeDup dataset construction:

```bash
bash runs/fineweb_edu_qwen3_semdedup_nohup_b200.sh seed42
```

Seeds 43 and 44 only, reusing the seed42 Qwen3 deduped data:

```bash
bash runs/fineweb_edu_qwen3_semdedup_nohup_b200.sh repeats
```

Regenerate ECDF and final report after runs complete:

```bash
bash runs/fineweb_edu_qwen3_semdedup_nohup_b200.sh report
```

## Acceptance Checks

The runner checks that:

- Qwen3 dry-run resolves to `Qwen/Qwen3-Embedding-8B`.
- Qwen3 vLLM kwargs include `runner=pooling`, `convert=embed`, `dtype=bfloat16`, `enforce_eager=true`, and `attention_config.backend=TRITON_ATTN`.
- Seed 42 produces `semdedup_manifest.json`, `data_stats.json`, the deduped data dir, and pairwise results.
- Final report mode requires all three Qwen3 runs to have `run_summary.json`, `base_eval_core.csv`, `data_stats.json`, and `semdedup_manifest.json`.
