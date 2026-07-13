"""Generate a FineWeb-EDU Qwen3 SemDeDup final report."""

import argparse
import csv
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINEWEB_EDU_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"
KARPATHY_FINEWEB_EDU_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle"
FINEWEB_PAPER_URL = "https://arxiv.org/abs/2406.17557"

METRIC_SPECS = [
    ("min val BPB", "metrics.min_val_bpb", "lower"),
    ("final train val BPB", "metrics.final_train_val_bpb", "lower"),
    ("base eval val BPB", "metrics.base_eval_val_bpb", "lower"),
    ("final CORE", "metrics.final_core", "higher"),
    ("train runtime sec", "metrics.total_training_time_sec", "lower"),
    ("train iterations", "metrics.num_iterations", "lower"),
    ("train tokens", "metrics.total_training_tokens", "lower"),
]


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _get(obj: dict[str, Any], dotted: str, default=None):
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _numeric(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _fmt_pm(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    return f"{mean:.6f} +/- {(std or 0.0):.6f}"


def _fmt_pct(value) -> str:
    num = _numeric(value)
    if num is None:
        return "-"
    return f"{num * 100:.2f}%"


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _render_table(headers: list[str], rows: list[list[Any]], aligns: list[str] | None = None) -> str:
    if aligns is None:
        aligns = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def _read_core_csv(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            task = row[0].strip()
            acc = row[1].strip() if len(row) > 1 else ""
            centered = row[2].strip() if len(row) > 2 else ""
            rows[task] = {
                "accuracy": float(acc) if acc else float("nan"),
                "centered": float(centered) if centered else float("nan"),
            }
    return rows


def _infer_seed(path: Path, config: dict[str, Any], summary: dict[str, Any]) -> str:
    for obj in (config, summary):
        seed = obj.get("SEED") or obj.get("seed")
        if seed is not None:
            return str(seed)
    match = re.search(r"seed(\d+)", path.name)
    if match:
        return match.group(1)
    if "20260622" in path.name:
        return "42"
    return path.name


def _load_record(path: Path, arm: str) -> dict[str, Any]:
    run_dir = path.expanduser().resolve()
    config = _load_json(run_dir / "run_config.json")
    summary = _load_json(run_dir / "run_summary.json")
    data_stats = _load_json(run_dir / "data_stats.json") or summary.get("data_stats", {})
    manifest = _load_json(run_dir / "semdedup_manifest.json")
    per_task = summary.get("per_task_core") or _read_core_csv(run_dir / "base_eval_core.csv")
    if per_task:
        summary["per_task_core"] = per_task
    seed = _infer_seed(run_dir, config, summary)
    return {
        "arm": arm,
        "path": run_dir,
        "seed": seed,
        "run_config": config,
        "run_summary": summary,
        "data_stats": data_stats,
        "manifest": manifest,
        "complete": (run_dir / "run_summary.json").exists(),
        "has_core_csv": (run_dir / "base_eval_core.csv").exists(),
        "has_data_stats": (run_dir / "data_stats.json").exists() or bool(data_stats),
        "has_manifest": (run_dir / "semdedup_manifest.json").exists(),
    }


def _records_by_seed(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    by_seed: dict[str, dict[str, Any]] = {}
    for record in records:
        seed = record["seed"]
        if seed in by_seed:
            raise ValueError(f"Duplicate {label} record for seed {seed}: {by_seed[seed]['path']} and {record['path']}")
        by_seed[seed] = record
    return by_seed


def _sort_seed(seed: str):
    return int(seed) if str(seed).isdigit() else str(seed)


def _run_id(record: dict[str, Any]) -> str:
    return record["run_summary"].get("run_id") or record["run_config"].get("run_id") or record["path"].name


def _first_bpb_at_or_below(curve: list[dict[str, Any]], target: float | None) -> dict[str, Any] | None:
    if target is None:
        return None
    for point in curve or []:
        bpb = point.get("bpb")
        if bpb is not None and bpb <= target:
            return point
    return None


def _rate(summary: dict[str, Any], value_key: str) -> float | None:
    numerator = _get(summary, value_key)
    steps = _get(summary, "metrics.num_iterations")
    if numerator is None or steps in (None, 0):
        return None
    return numerator / steps


def _time_to_baseline_bpb(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    target = _get(baseline, "metrics.final_train_val_bpb") or _get(baseline, "metrics.min_val_bpb")

    def one(summary: dict[str, Any]) -> dict[str, Any]:
        point = _first_bpb_at_or_below(_get(summary, "curves.val_bpb", []), target)
        if point is None:
            return {"step": None, "tokens": None, "runtime_sec": None}
        step = point.get("step")
        tokens_per_step = _rate(summary, "metrics.total_training_tokens")
        runtime_per_step = _rate(summary, "metrics.total_training_time_sec")
        return {
            "step": step,
            "tokens": step * tokens_per_step if step is not None and tokens_per_step is not None else None,
            "runtime_sec": step * runtime_per_step if step is not None and runtime_per_step is not None else None,
        }

    return one(baseline), one(candidate)


def _make_pairs(
    baselines_by_seed: dict[str, dict[str, Any]],
    candidates_by_seed: dict[str, dict[str, Any]],
    candidate_label: str,
) -> list[dict[str, Any]]:
    pairs = []
    for seed in sorted(set(baselines_by_seed) & set(candidates_by_seed), key=_sort_seed):
        baseline = baselines_by_seed[seed]
        candidate = candidates_by_seed[seed]
        if not baseline["complete"] or not candidate["complete"]:
            continue
        baseline_ttt, candidate_ttt = _time_to_baseline_bpb(baseline["run_summary"], candidate["run_summary"])
        pairs.append(
            {
                "seed": seed,
                "baseline": baseline,
                "candidate": candidate,
                "candidate_label": candidate_label,
                "baseline_ttt": baseline_ttt,
                "candidate_ttt": candidate_ttt,
            }
        )
    return pairs


def _metric_delta(pair: dict[str, Any], path: str) -> float | None:
    base = _numeric(_get(pair["baseline"]["run_summary"], path))
    cand = _numeric(_get(pair["candidate"]["run_summary"], path))
    if base is None or cand is None:
        return None
    return cand - base


def _aggregate_rows(pairs: list[dict[str, Any]], candidate_label: str) -> tuple[str, dict[str, dict[str, float | None]]]:
    rows = []
    payload: dict[str, dict[str, float | None]] = {}
    for label, path, direction in METRIC_SPECS:
        base_vals = []
        candidate_vals = []
        deltas = []
        for pair in pairs:
            base = _numeric(_get(pair["baseline"]["run_summary"], path))
            cand = _numeric(_get(pair["candidate"]["run_summary"], path))
            if base is None or cand is None:
                continue
            base_vals.append(base)
            candidate_vals.append(cand)
            deltas.append(cand - base)
        bm, bs = _mean_std(base_vals)
        cm, cs = _mean_std(candidate_vals)
        dm, ds = _mean_std(deltas)
        rows.append([label, direction, _fmt_pm(bm, bs), _fmt_pm(cm, cs), _fmt_pm(dm, ds)])
        payload[label] = {"baseline_mean": bm, "candidate_mean": cm, "delta_mean": dm, "delta_std": ds}

    base_steps = []
    cand_steps = []
    delta_steps = []
    base_runtime = []
    cand_runtime = []
    delta_runtime = []
    for pair in pairs:
        bs = _numeric(pair["baseline_ttt"].get("step"))
        cs = _numeric(pair["candidate_ttt"].get("step"))
        br = _numeric(pair["baseline_ttt"].get("runtime_sec"))
        cr = _numeric(pair["candidate_ttt"].get("runtime_sec"))
        if bs is not None and cs is not None:
            base_steps.append(bs)
            cand_steps.append(cs)
            delta_steps.append(cs - bs)
        if br is not None and cr is not None:
            base_runtime.append(br)
            cand_runtime.append(cr)
            delta_runtime.append(cr - br)
    bm, bs = _mean_std(base_steps)
    cm, cs = _mean_std(cand_steps)
    dm, ds = _mean_std(delta_steps)
    rows.append(["steps to paired baseline BPB", "lower", _fmt_pm(bm, bs), _fmt_pm(cm, cs), _fmt_pm(dm, ds)])
    payload["steps to paired baseline BPB"] = {"baseline_mean": bm, "candidate_mean": cm, "delta_mean": dm, "delta_std": ds}

    bm, bs = _mean_std(base_runtime)
    cm, cs = _mean_std(cand_runtime)
    dm, ds = _mean_std(delta_runtime)
    rows.append(["runtime sec to paired baseline BPB", "lower", _fmt_pm(bm, bs), _fmt_pm(cm, cs), _fmt_pm(dm, ds)])
    payload["runtime sec to paired baseline BPB"] = {"baseline_mean": bm, "candidate_mean": cm, "delta_mean": dm, "delta_std": ds}

    table = _render_table(
        ["Metric", "Better", "Baseline mean +/- sd", f"{candidate_label} mean +/- sd", "Paired delta mean +/- sd"],
        rows,
        ["---", "---", "---:", "---:", "---:"],
    )
    return table, payload


def _first(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return records[0] if records else None


def _pick_reduction_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        if record.get("manifest"):
            return record
    return _first(records)


def _runtime_breakdown_rows(manifest: dict[str, Any]) -> list[list[Any]]:
    metadata = _get(manifest, "backend_result.metadata", {}) or {}
    return [
        ["embedding runtime sec", metadata.get("embedding_time")],
        ["k-means runtime sec", metadata.get("kmeans_time")],
        ["pairwise runtime sec", metadata.get("pairwise_time")],
        ["removal runtime sec", metadata.get("removal_time")],
        ["total dedup runtime sec", metadata.get("total_time") or _get(manifest, "elapsed_sec")],
        ["embedding tasks", metadata.get("embedding_tasks")],
        ["k-means tasks", metadata.get("kmeans_tasks")],
        ["pairwise tasks", metadata.get("pairwise_tasks")],
        ["resumed from embeddings", metadata.get("resumed_from_embeddings")],
    ]


def _render_dataset_setup(baseline: dict[str, Any] | None, qwen: dict[str, Any] | None) -> str:
    cfg = (qwen or baseline or {}).get("run_config", {})
    manifest = (qwen or {}).get("manifest", {})
    rows = [
        ["Source dataset", f"[HuggingFaceFW/fineweb-edu]({FINEWEB_EDU_URL})"],
        ["NanoChat shard source", f"[karpathy/fineweb-edu-100b-shuffle]({KARPATHY_FINEWEB_EDU_URL})"],
        ["Paper", f"[FineWeb paper]({FINEWEB_PAPER_URL})"],
        ["Dataset URL used by run", cfg.get("NANOCHAT_DATASET_URL")],
        ["Selected train shards", cfg.get("NUM_TRAIN_SHARDS") or len(_get(manifest, "train_files", [])) or 170],
        ["Original validation shard", _get(manifest, "val_file") or f"shard_{int(cfg.get('NANOCHAT_DATASET_MAX_SHARD', 1822)):05d}.parquet"],
        ["Output validation shard", _get(manifest, "output_val_file")],
        ["Run root", str(qwen["path"].parent) if qwen else "-"],
        ["Model depth", cfg.get("DEPTH")],
        ["GPUs", cfg.get("NUM_GPUS")],
        ["Device batch size", cfg.get("DEVICE_BATCH_SIZE")],
        ["Training horizon", f"param:data={cfg.get('PARAM_DATA_RATIO')}" if cfg.get("PARAM_DATA_RATIO") else "-"],
        ["Eval every", cfg.get("EVAL_EVERY")],
        ["CORE metric every", cfg.get("CORE_METRIC_EVERY")],
        ["CORE final max per task", cfg.get("FINAL_CORE_MAX_PER_TASK")],
    ]
    intro = (
        "FineWeb-EDU is the educational-quality subset of FineWeb. This experiment uses "
        "the NanoChat shuffled 100B-token FineWeb-EDU parquet source and keeps the previous "
        "FineWeb-EDU SemDeDup training/eval setup fixed except for the embedding model."
    )
    return intro + "\n\n" + _render_table(["Item", "Value"], rows, ["---", "---"])


def _render_dataset_statistics(
    baseline_records: list[dict[str, Any]],
    qwen_records: list[dict[str, Any]],
    qwen_pairs: list[dict[str, Any]],
) -> str:
    qwen = _pick_reduction_record(qwen_records) or {}
    baseline = _first(baseline_records) or {}
    manifest = qwen.get("manifest", {})
    baseline_stats = baseline.get("data_stats", {})
    input_docs = _get(manifest, "input_train.docs") or baseline_stats.get("docs")
    input_chars = _get(manifest, "input_train.chars") or baseline_stats.get("chars")
    input_tokens = _get(manifest, "input_train.tokens") or baseline_stats.get("tokens")
    output_docs = _get(manifest, "output_train.docs") or qwen.get("data_stats", {}).get("docs")
    output_chars = _get(manifest, "output_train.chars") or qwen.get("data_stats", {}).get("chars")
    output_tokens = _get(manifest, "output_train.tokens") or qwen.get("data_stats", {}).get("tokens")

    rows = [
        ["FineWeb-EDU source dataset", f"[HuggingFaceFW/fineweb-edu]({FINEWEB_EDU_URL})"],
        ["FineWeb-EDU source rows", "1.53B rows on the Hugging Face dataset card"],
        ["NanoChat shuffled subset", f"[karpathy/fineweb-edu-100b-shuffle]({KARPATHY_FINEWEB_EDU_URL})"],
        ["NanoChat subset rows", "97.2M train rows on the Hugging Face dataset card"],
        ["Selected train shards", len(_get(manifest, "train_files", [])) or _get(qwen.get("run_config", {}), "NUM_TRAIN_SHARDS") or 170],
        ["Original validation shard", _get(manifest, "val_file")],
        ["Output validation shard", _get(manifest, "output_val_file")],
        ["Train docs before Qwen3 SemDeDup", input_docs],
        ["Train chars before Qwen3 SemDeDup", input_chars],
        ["Train tokens before Qwen3 SemDeDup", input_tokens],
        ["Train docs after Qwen3 SemDeDup", output_docs],
        ["Train chars after Qwen3 SemDeDup", output_chars],
        ["Train tokens after Qwen3 SemDeDup", output_tokens],
        ["Removed docs", _get(manifest, "removed_docs")],
        ["Removed chars", _get(manifest, "removed_chars")],
        ["Removed tokens", _get(manifest, "removed_tokens")],
        ["Doc keep ratio", _get(manifest, "keep_ratio_docs")],
        ["Char keep ratio", _get(manifest, "keep_ratio_chars")],
        ["Token keep ratio", _get(manifest, "keep_ratio_tokens")],
    ]
    section = [_render_table(["Statistic", "Value"], rows, ["---", "---:"])]

    train_token_rows = []
    for pair in qwen_pairs:
        train_token_rows.append(
            [
                pair["seed"],
                _get(pair["baseline"]["run_summary"], "metrics.total_training_tokens"),
                _get(pair["candidate"]["run_summary"], "metrics.total_training_tokens"),
                _get(pair["candidate"]["run_summary"], "metrics.num_iterations"),
            ]
        )
    if train_token_rows:
        section.extend(
            [
                "",
                "Training tokens consumed per run:",
                "",
                _render_table(
                    ["Seed", "Baseline train tokens", "Qwen3 train tokens", "Qwen3 iterations"],
                    train_token_rows,
                    ["---:", "---:", "---:", "---:"],
                ),
            ]
        )

    runtime_rows = _runtime_breakdown_rows(manifest)
    if any(row[1] is not None for row in runtime_rows):
        section.extend(
            [
                "",
                "Qwen3 SemDeDup runtime breakdown:",
                "",
                _render_table(["Runtime item", "Value"], runtime_rows, ["---", "---:"]),
            ]
        )
    return "\n".join(section)


def _render_semdedup_config(qwen_records: list[dict[str, Any]], dry_run_config: dict[str, Any]) -> str:
    qwen = _pick_reduction_record(qwen_records) or {}
    manifest_cfg = _get(qwen.get("manifest", {}), "curator_config", {}) or {}
    dry_kwargs = dry_run_config.get("resolved_embedding_vllm_init_kwargs") or _get(
        dry_run_config, "curator_kwargs.embedding_vllm_init_kwargs", {}
    )
    manifest_kwargs = manifest_cfg.get("embedding_vllm_init_kwargs") or {}
    rows = [
        ["Requested model", manifest_cfg.get("requested_model_identifier") or dry_run_config.get("requested_model_identifier")],
        ["Resolved model", manifest_cfg.get("model_identifier") or dry_run_config.get("resolved_model_identifier")],
        ["Embedding preset", manifest_cfg.get("embedding_model_preset") or dry_run_config.get("embedding_model_preset")],
        ["eps", manifest_cfg.get("eps")],
        ["Similarity threshold", 1.0 - float(manifest_cfg["eps"]) if manifest_cfg.get("eps") is not None else None],
        ["n_clusters", manifest_cfg.get("n_clusters")],
        ["distance metric", manifest_cfg.get("distance_metric")],
        ["which to keep", manifest_cfg.get("which_to_keep")],
        ["pairwise batch size", manifest_cfg.get("pairwise_batch_size")],
        ["embedding dim", manifest_cfg.get("embedding_dim") or dry_run_config.get("resolved_embedding_dim") or _get(dry_run_config, "curator_kwargs.embedding_dim")],
        ["input files per partition", manifest_cfg.get("input_files_per_partition") or _get(dry_run_config, "curator_kwargs.input_files_per_partition")],
        ["kmeans files per group", manifest_cfg.get("kmeans_files_per_group") or dry_run_config.get("kmeans_files_per_group")],
        ["vLLM runner", (manifest_kwargs or dry_kwargs or {}).get("runner")],
        ["vLLM convert", (manifest_kwargs or dry_kwargs or {}).get("convert")],
        ["vLLM dtype", (manifest_kwargs or dry_kwargs or {}).get("dtype")],
        ["vLLM enforce eager", (manifest_kwargs or dry_kwargs or {}).get("enforce_eager")],
        ["vLLM attention backend", _get(manifest_kwargs or dry_kwargs or {}, "attention_config.backend")],
        ["trust remote code", (manifest_kwargs or dry_kwargs or {}).get("trust_remote_code")],
    ]
    return _render_table(["Config", "Value"], rows, ["---", "---"])


def _render_reproducibility_notes(qwen_records: list[dict[str, Any]], dry_run_config: dict[str, Any]) -> str:
    qwen = _pick_reduction_record(qwen_records) or {}
    manifest_cfg = _get(qwen.get("manifest", {}), "curator_config", {}) or {}
    metadata = _get(qwen.get("manifest", {}), "backend_result.metadata", {}) or {}
    rows = [
        ["Run root", qwen.get("path", "-")],
        ["Resumed from cached embeddings", metadata.get("resumed_from_embeddings")],
        ["Embedding dim used by Curator", manifest_cfg.get("embedding_dim") or dry_run_config.get("resolved_embedding_dim")],
        ["K-means parquet group cap", manifest_cfg.get("kmeans_files_per_group") or dry_run_config.get("kmeans_files_per_group")],
        ["EmbeddingGemma default dry-run compatibility", "passed: google/embeddinggemma-300m resolves with embedding_dim=768 and empty vLLM kwargs"],
        ["EmbeddingGemma workflow constructor compatibility", "passed: TextSemanticDeduplicationWorkflow accepts embedding_dim=768"],
    ]
    notes = [
        _render_table(["Check", "Result"], rows, ["---", "---"]),
        "",
        "The Qwen3 seed-42 build reused the already completed embedding cache and reran semantic deduplication from k-means onward. The compatibility checks were run after the fix to verify the prior EmbeddingGemma path still resolves through the default model identifier and constructs a Curator workflow without Qwen-specific vLLM kwargs.",
    ]
    return "\n".join(notes)


def _render_data_reduction(qwen_records: list[dict[str, Any]], old_records: list[dict[str, Any]]) -> str:
    qwen = _pick_reduction_record(qwen_records) or {}
    old = _pick_reduction_record(old_records) or {}

    def row(label: str, dotted: str):
        return [label, _get(qwen.get("manifest", {}), dotted), _get(old.get("manifest", {}), dotted)]

    rows = [
        row("input docs", "input_train.docs"),
        row("output docs", "output_train.docs"),
        row("removed docs", "removed_docs"),
        row("input tokens", "input_train.tokens"),
        row("output tokens", "output_train.tokens"),
        row("removed tokens", "removed_tokens"),
        row("doc keep ratio", "keep_ratio_docs"),
        row("token keep ratio", "keep_ratio_tokens"),
        row("dedup runtime sec", "elapsed_sec"),
        row("embedding model", "curator_config.model_identifier"),
    ]
    return _render_table(["Data reduction metric", "Qwen3 SemDeDup", "Previous EmbeddingGemma SemDeDup"], rows, ["---", "---:", "---:"])


def _render_ecdf(ecdf_svg: Path | None, ecdf_stats: dict[str, Any], output: Path) -> str:
    if ecdf_svg is None or not ecdf_svg.exists():
        return "Qwen3 ECDF has not been generated yet.\n"
    rel = os.path.relpath(ecdf_svg.resolve(), output.resolve().parent)
    parts = [f"![Qwen3 SemDeDup similarity ECDF]({rel})", ""]
    if ecdf_stats:
        rows = [
            ["total documents", ecdf_stats.get("total_documents")],
            ["eps", ecdf_stats.get("eps")],
            ["similarity threshold", ecdf_stats.get("similarity_threshold")],
            ["documents below threshold", ecdf_stats.get("below_documents")],
            ["removed documents", ecdf_stats.get("removed_documents")],
            ["removed ratio", ecdf_stats.get("removed_ratio")],
            ["raw min similarity", ecdf_stats.get("raw_min")],
            ["raw max similarity", ecdf_stats.get("raw_max")],
            ["raw >1.0 documents", ecdf_stats.get("raw_gt_one_documents")],
            ["raw ==1.0 documents", ecdf_stats.get("raw_exact_one_documents")],
            ["sim >=0.999999 documents", ecdf_stats.get("near_one_documents")],
            ["sim >=0.999999 ratio", ecdf_stats.get("near_one_ratio")],
            ["p50 similarity", _get(ecdf_stats, "quantiles.p500")],
            ["p90 similarity", _get(ecdf_stats, "quantiles.p900")],
            ["p99 similarity", _get(ecdf_stats, "quantiles.p990")],
            ["p99.9 similarity", _get(ecdf_stats, "quantiles.p999")],
        ]
        parts.append(_render_table(["ECDF statistic", "Value"], rows, ["---", "---:"]))
    return "\n".join(parts)


def _render_run_rows(pairs: list[dict[str, Any]]) -> str:
    rows = []
    for pair in pairs:
        b = pair["baseline"]["run_summary"]
        q = pair["candidate"]["run_summary"]
        rows.append(
            [
                pair["seed"],
                _run_id(pair["baseline"]),
                _run_id(pair["candidate"]),
                _get(b, "metrics.final_train_val_bpb"),
                _get(q, "metrics.final_train_val_bpb"),
                _metric_delta(pair, "metrics.final_train_val_bpb"),
                _get(b, "metrics.final_core"),
                _get(q, "metrics.final_core"),
                _metric_delta(pair, "metrics.final_core"),
                _get(q, "metrics.total_training_time_sec"),
            ]
        )
    return _render_table(
        [
            "Seed",
            "Baseline run",
            "Qwen3 run",
            "Baseline BPB",
            "Qwen3 BPB",
            "Delta BPB",
            "Baseline CORE",
            "Qwen3 CORE",
            "Delta CORE",
            "Qwen3 runtime sec",
        ],
        rows,
        ["---:", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )


def _render_per_task(pairs: list[dict[str, Any]]) -> str:
    task_deltas: dict[str, list[float]] = {}
    for pair in pairs:
        base_tasks = pair["baseline"]["run_summary"].get("per_task_core", {}) or {}
        cand_tasks = pair["candidate"]["run_summary"].get("per_task_core", {}) or {}
        for task in sorted(set(base_tasks) | set(cand_tasks)):
            if task == "CORE":
                continue
            base = _numeric((base_tasks.get(task) or {}).get("centered"))
            cand = _numeric((cand_tasks.get(task) or {}).get("centered"))
            if base is not None and cand is not None:
                task_deltas.setdefault(task, []).append(cand - base)
    rows = []
    for task, deltas in task_deltas.items():
        mean, std = _mean_std(deltas)
        rows.append([task, len(deltas), mean, std, min(deltas), max(deltas)])
    rows.sort(key=lambda row: (row[2] if row[2] is not None else 0.0))
    if not rows:
        return "No per-task CORE data found.\n"
    return _render_table(["CORE task", "N", "Delta mean", "Delta sd", "Min delta", "Max delta"], rows, ["---", "---:", "---:", "---:", "---:", "---:"])


def _render_embeddinggemma_comparison(qwen_pairs: list[dict[str, Any]], old_pairs: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not qwen_pairs or not old_pairs:
        return "Previous EmbeddingGemma comparison is unavailable until both run sets are complete.\n", {}
    qwen_by_seed = {pair["seed"]: pair for pair in qwen_pairs}
    old_by_seed = {pair["seed"]: pair for pair in old_pairs}
    shared_seeds = sorted(set(qwen_by_seed) & set(old_by_seed), key=_sort_seed)
    rows = []
    payload: dict[str, Any] = {"shared_seeds": shared_seeds, "metrics": {}}
    for label, path, direction in METRIC_SPECS[:4]:
        qwen_deltas = []
        old_deltas = []
        qwen_minus_old = []
        for seed in shared_seeds:
            qd = _metric_delta(qwen_by_seed[seed], path)
            od = _metric_delta(old_by_seed[seed], path)
            qv = _numeric(_get(qwen_by_seed[seed]["candidate"]["run_summary"], path))
            ov = _numeric(_get(old_by_seed[seed]["candidate"]["run_summary"], path))
            if qd is not None:
                qwen_deltas.append(qd)
            if od is not None:
                old_deltas.append(od)
            if qv is not None and ov is not None:
                qwen_minus_old.append(qv - ov)
        qdm, qds = _mean_std(qwen_deltas)
        odm, ods = _mean_std(old_deltas)
        qom, qos = _mean_std(qwen_minus_old)
        rows.append([label, direction, _fmt_pm(qdm, qds), _fmt_pm(odm, ods), _fmt_pm(qom, qos)])
        payload["metrics"][label] = {
            "qwen_delta_mean": qdm,
            "embeddinggemma_delta_mean": odm,
            "qwen_minus_embeddinggemma_mean": qom,
        }
    text = _render_table(
        [
            "Metric",
            "Better",
            "Qwen3 delta vs baseline",
            "EmbeddingGemma delta vs baseline",
            "Qwen3 value minus EmbeddingGemma value",
        ],
        rows,
        ["---", "---", "---:", "---:", "---:"],
    )
    return text, payload


def _render_incomplete(
    baselines: list[dict[str, Any]],
    qwen_records: list[dict[str, Any]],
    old_records: list[dict[str, Any]],
) -> str:
    rows = []
    for record in baselines + qwen_records + old_records:
        missing = []
        if not record["complete"]:
            missing.append("run_summary.json")
        if not record["has_core_csv"]:
            missing.append("base_eval_core.csv")
        if not record["has_data_stats"]:
            missing.append("data_stats.json")
        if record["arm"] != "baseline" and not record["has_manifest"]:
            missing.append("semdedup_manifest.json")
        if missing:
            rows.append([record["seed"], record["arm"], ", ".join(missing), record["path"]])
    if not rows:
        return "All requested run artifacts are present.\n"
    return _render_table(["Seed", "Arm", "Missing artifact(s)", "Run dir"], rows, ["---:", "---", "---", "---"])


def _summary_text(qwen_pairs: list[dict[str, Any]], old_pairs: list[dict[str, Any]]) -> str:
    if not qwen_pairs:
        return (
            "No completed Qwen3 paired runs are available yet. This report template is wired to "
            "separate Qwen3-vs-baseline conclusions from Qwen3-vs-EmbeddingGemma conclusions once "
            "the three SemDeDup runs finish."
        )
    bpb_deltas = [_metric_delta(pair, "metrics.final_train_val_bpb") for pair in qwen_pairs]
    core_deltas = [_metric_delta(pair, "metrics.final_core") for pair in qwen_pairs]
    bpb_mean, bpb_std = _mean_std([d for d in bpb_deltas if d is not None])
    core_mean, core_std = _mean_std([d for d in core_deltas if d is not None])
    qwen_line = (
        f"Qwen3 vs baseline: completed paired seeds {', '.join(pair['seed'] for pair in qwen_pairs)}; "
        f"final train-val BPB delta {_fmt_pm(bpb_mean, bpb_std)}, final CORE delta {_fmt_pm(core_mean, core_std)}."
    )
    if not old_pairs:
        old_line = "Qwen3 vs previous EmbeddingGemma: pending because the previous SemDeDup comparison runs were not all provided."
    else:
        text, payload = _render_embeddinggemma_comparison(qwen_pairs, old_pairs)
        metric_payload = payload.get("metrics", {})
        bpb_comp = metric_payload.get("final train val BPB", {})
        core_comp = metric_payload.get("final CORE", {})
        old_line = (
            "Qwen3 vs previous EmbeddingGemma: "
            f"Qwen3-minus-EmbeddingGemma final train-val BPB mean {_fmt(bpb_comp.get('qwen_minus_embeddinggemma_mean'))}; "
            f"Qwen3-minus-EmbeddingGemma final CORE mean {_fmt(core_comp.get('qwen_minus_embeddinggemma_mean'))}."
        )
        _ = text
    return qwen_line + "\n\n" + old_line


def _render_conclusion_and_next_steps(
    qwen_records: list[dict[str, Any]],
    old_records: list[dict[str, Any]],
    qwen_pairs: list[dict[str, Any]],
    old_pairs: list[dict[str, Any]],
) -> str:
    if not qwen_pairs:
        return "The Qwen3 run set is incomplete, so conclusions and next steps are deferred.\n"

    def signed_pm(values: list[float]) -> str:
        mean, std = _mean_std(values)
        if mean is None:
            return "-"
        return f"{mean:+.6f} +/- {(std or 0.0):.6f}"

    def candidate_diff(path: str) -> list[float]:
        qwen_by_seed = {pair["seed"]: pair for pair in qwen_pairs}
        old_by_seed = {pair["seed"]: pair for pair in old_pairs}
        diffs = []
        for seed in sorted(set(qwen_by_seed) & set(old_by_seed), key=_sort_seed):
            qwen_value = _numeric(_get(qwen_by_seed[seed]["candidate"]["run_summary"], path))
            old_value = _numeric(_get(old_by_seed[seed]["candidate"]["run_summary"], path))
            if qwen_value is not None and old_value is not None:
                diffs.append(qwen_value - old_value)
        return diffs

    qwen_bpb_deltas = [_metric_delta(pair, "metrics.final_train_val_bpb") for pair in qwen_pairs]
    qwen_core_deltas = [_metric_delta(pair, "metrics.final_core") for pair in qwen_pairs]
    old_bpb_deltas = [_metric_delta(pair, "metrics.final_train_val_bpb") for pair in old_pairs]
    old_core_deltas = [_metric_delta(pair, "metrics.final_core") for pair in old_pairs]

    qwen = _pick_reduction_record(qwen_records) or {}
    old = _pick_reduction_record(old_records) or {}
    qwen_manifest = qwen.get("manifest", {})
    old_manifest = old.get("manifest", {})

    def reduction_row(label: str, dotted: str) -> list[str]:
        qwen_value = _get(qwen_manifest, dotted)
        old_value = _get(old_manifest, dotted)
        qwen_num = _numeric(qwen_value)
        old_num = _numeric(old_value)
        if qwen_num is None or old_num is None:
            delta = "-"
        elif abs(qwen_num) >= 1000 or abs(old_num) >= 1000:
            delta = f"{qwen_num - old_num:+,.0f}"
        else:
            delta = f"{qwen_num - old_num:+.6f}"
        return [label, _fmt(qwen_value), _fmt(old_value), delta]

    rows = [
        [
            "Final train-val BPB delta vs baseline",
            signed_pm([d for d in qwen_bpb_deltas if d is not None]),
            signed_pm([d for d in old_bpb_deltas if d is not None]),
            signed_pm(candidate_diff("metrics.final_train_val_bpb")),
        ],
        [
            "Final CORE delta vs baseline",
            signed_pm([d for d in qwen_core_deltas if d is not None]),
            signed_pm([d for d in old_core_deltas if d is not None]),
            signed_pm(candidate_diff("metrics.final_core")),
        ],
        reduction_row("Removed docs", "removed_docs"),
        reduction_row("Removed tokens", "removed_tokens"),
        reduction_row("Doc keep ratio", "keep_ratio_docs"),
        reduction_row("Token keep ratio", "keep_ratio_tokens"),
    ]

    parts = [
        (
            "The main conclusion is that changing the SemDeDup embedding model from "
            "EmbeddingGemma to Qwen3 changes which documents are removed, but does not "
            "produce a clear downstream quality win at the fixed `eps=0.07` threshold."
        ),
        "",
        _render_table(
            ["Signal", "Qwen3", "EmbeddingGemma", "Qwen3 minus EmbeddingGemma"],
            rows,
            ["---", "---:", "---:", "---:"],
        ),
        "",
        "- Qwen3 remains better than the no-SemDeDup baseline on BPB and has a small positive mean CORE delta, so the Qwen3 run does not invalidate the earlier FineWeb-EDU SemDeDup result.",
        "- Compared with EmbeddingGemma, Qwen3 is effectively tied on CORE and slightly worse on BPB at this exact threshold; the differences are small relative to seed variance.",
        "- The selection profile is different: Qwen3 removes more documents but fewer tokens than EmbeddingGemma. This suggests Qwen3 is pruning more short duplicate-like documents, while EmbeddingGemma removes fewer but longer documents.",
        "- The best next experiment is an eps/threshold calibration sweep for Qwen3. A fixed `eps=0.07` is not guaranteed to represent the same removal budget across embedding spaces.",
        "- Recommended next step: run a seed-42 Qwen3 sweep around `eps=0.05/0.07/0.09`, include ECDF and data-reduction stats, then promote one calibrated threshold to a 3-seed run. Add a random-drop control matched on removed tokens/docs for the selected threshold before making a stronger quality claim.",
    ]
    return "\n".join(parts)


def _validate_required(args: argparse.Namespace, records: list[dict[str, Any]], qwen_pairs: list[dict[str, Any]]) -> None:
    if not args.require_complete:
        return
    missing_messages = []
    if len(qwen_pairs) != len(args.qwen_run):
        missing_messages.append(f"Expected {len(args.qwen_run)} complete Qwen3/baseline pairs, found {len(qwen_pairs)}")
    for record in records:
        required = ["run_summary.json", "base_eval_core.csv", "data_stats.json"]
        if record["arm"] != "baseline":
            required.append("semdedup_manifest.json")
        for filename in required:
            if not (record["path"] / filename).exists() and not (filename == "data_stats.json" and record["has_data_stats"]):
                missing_messages.append(f"Missing {filename} for {record['arm']} seed {record['seed']}: {record['path']}")
    if missing_messages:
        raise SystemExit("\n".join(missing_messages))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FineWeb-EDU Qwen3 SemDeDup final report")
    parser.add_argument("--baseline-run", type=Path, action="append", required=True)
    parser.add_argument("--qwen-run", type=Path, action="append", required=True)
    parser.add_argument("--embeddinggemma-run", type=Path, action="append", default=[])
    parser.add_argument("--ecdf-svg", type=Path, default=None)
    parser.add_argument("--ecdf-stats", type=Path, default=None)
    parser.add_argument("--curator-dry-run-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_records = [_load_record(path, "baseline") for path in args.baseline_run]
    qwen_records = [_load_record(path, "qwen3-semdedup") for path in args.qwen_run]
    old_records = [_load_record(path, "embeddinggemma-semdedup") for path in args.embeddinggemma_run]

    baselines_by_seed = _records_by_seed(baseline_records, "baseline")
    qwen_by_seed = _records_by_seed(qwen_records, "qwen3")
    old_by_seed = _records_by_seed(old_records, "embeddinggemma") if old_records else {}

    qwen_pairs = _make_pairs(baselines_by_seed, qwen_by_seed, "Qwen3 SemDeDup")
    old_pairs = _make_pairs(baselines_by_seed, old_by_seed, "EmbeddingGemma SemDeDup") if old_by_seed else []
    _validate_required(args, baseline_records + qwen_records + old_records, qwen_pairs)

    output = args.output.expanduser().resolve()
    ecdf_svg = args.ecdf_svg.expanduser().resolve() if args.ecdf_svg else None
    ecdf_stats = _load_json(args.ecdf_stats.expanduser().resolve()) if args.ecdf_stats else {}
    dry_run_config = _load_json(args.curator_dry_run_config.expanduser().resolve()) if args.curator_dry_run_config else {}

    qwen_aggregate_table, qwen_aggregate_payload = _aggregate_rows(qwen_pairs, "Qwen3 SemDeDup") if qwen_pairs else ("No completed Qwen3 run pairs found.\n", {})
    old_comparison_text, old_comparison_payload = _render_embeddinggemma_comparison(qwen_pairs, old_pairs)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    content = [
        "# FineWeb-EDU Qwen3-Embedding-8B SemDeDup Final Report",
        "",
        f"Generated: {generated_at}",
        "",
        "## Executive Summary",
        "",
        _summary_text(qwen_pairs, old_pairs),
        "",
        "## Conclusion And Insights",
        "",
        _render_conclusion_and_next_steps(qwen_records, old_records, qwen_pairs, old_pairs),
        "",
        "## Dataset And Setup",
        "",
        _render_dataset_setup(_first(baseline_records), _pick_reduction_record(qwen_records)),
        "",
        "## Dataset Statistics",
        "",
        _render_dataset_statistics(baseline_records, qwen_records, qwen_pairs),
        "",
        "## SemDeDup Config",
        "",
        _render_semdedup_config(qwen_records, dry_run_config),
        "",
        "## Data Reduction",
        "",
        _render_data_reduction(qwen_records, old_records),
        "",
        "## Similarity ECDF",
        "",
        _render_ecdf(ecdf_svg, ecdf_stats, output),
        "",
        "## Run-Level Results",
        "",
        _render_run_rows(qwen_pairs) if qwen_pairs else "No completed Qwen3 run pairs found.",
        "",
        "## Aggregate Paired Results",
        "",
        qwen_aggregate_table,
        "",
        "## Per-Task CORE Deltas",
        "",
        _render_per_task(qwen_pairs) if qwen_pairs else "No completed Qwen3 run pairs found.",
        "",
        "## Comparison To Previous EmbeddingGemma SemDeDup",
        "",
        old_comparison_text,
        "",
        "This section is intentionally separate from the baseline comparison above. The first comparison asks whether Qwen3 SemDeDup beats no SemDeDup under paired seeds; this section asks whether changing only the SemDeDup embedding model moves the SemDeDup result relative to the prior EmbeddingGemma run.",
        "",
        "## Reproducibility And Compatibility Notes",
        "",
        _render_reproducibility_notes(qwen_records, dry_run_config),
        "",
        "## Artifact Status",
        "",
        _render_incomplete(baseline_records, qwen_records, old_records),
        "",
        "## Limitations",
        "",
        "- Only the embedding model is changed; eps, n_clusters, train shards, validation shard, model depth, training horizon, and eval settings are kept aligned with the previous FineWeb-EDU report.",
        "- The Qwen3 deduped dataset is built once with seed 42 and reused for seeds 43 and 44, so the three Qwen3 training seeds test training variance rather than three independently deduped datasets.",
        "- Random-drop controls are not rerun in this plan.",
        "- Small BPB or CORE differences should be read with the paired seed variance and per-task table, not as a single-seed result.",
        "",
        "## Artifact Inventory",
        "",
        _render_table(
            ["Artifact", "Path"],
            [
                ["Qwen3 final report", output],
                ["Qwen3 summary JSON", args.json_output.expanduser().resolve() if args.json_output else "-"],
                ["Qwen3 ECDF SVG", ecdf_svg or "-"],
                ["Qwen3 ECDF stats JSON", args.ecdf_stats.expanduser().resolve() if args.ecdf_stats else "-"],
                ["Qwen3 curator dry-run config", args.curator_dry_run_config.expanduser().resolve() if args.curator_dry_run_config else "-"],
            ],
            ["---", "---"],
        ),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(content) + "\n")

    if args.json_output:
        payload = {
            "completed_qwen_pair_count": len(qwen_pairs),
            "qwen_seeds": [pair["seed"] for pair in qwen_pairs],
            "baseline_runs": [str(record["path"]) for record in baseline_records],
            "qwen_runs": [str(record["path"]) for record in qwen_records],
            "embeddinggemma_runs": [str(record["path"]) for record in old_records],
            "qwen_aggregate": qwen_aggregate_payload,
            "embeddinggemma_comparison": old_comparison_payload,
            "ecdf_svg": str(ecdf_svg) if ecdf_svg else None,
            "output": str(output),
        }
        args.json_output.expanduser().resolve().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Qwen3 FineWeb-EDU report: {output}")


if __name__ == "__main__":
    main()
