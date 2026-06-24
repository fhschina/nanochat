"""Generate a multi-seed FineWeb-EDU SemDeDup comparison report."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


FILTERS = {
    "DATASET_TAG": "fineweb_edu",
    "NUM_TRAIN_SHARDS": "170",
    "PARAM_DATA_RATIO": "9.5",
    "SEMD_EPS": "0.07",
}

COMMON_PARAM_KEYS = [
    "DATASET_TAG",
    "NANOCHAT_DATASET_URL",
    "NANOCHAT_DATASET_MAX_SHARD",
    "NUM_TRAIN_SHARDS",
    "PARAM_DATA_RATIO",
    "DEPTH",
    "NUM_GPUS",
    "DEVICE_BATCH_SIZE",
    "SEED",
    "CORE_EVAL_SEED",
    "EVAL_EVERY",
    "EVAL_TOKENS",
    "CORE_METRIC_EVERY",
    "CORE_METRIC_MAX_PER_TASK",
    "FINAL_CORE_MAX_PER_TASK",
    "BASE_EVAL_MODES",
    "SEMD_BACKEND",
    "SEMD_MODEL",
    "SEMD_EPS",
    "SEMD_N_CLUSTERS",
    "SEMD_DISTANCE_METRIC",
    "SEMD_WHICH_TO_KEEP",
    "SEMD_PAIRWISE_BATCH_SIZE",
]

METRICS = [
    ("min val BPB", "metrics.min_val_bpb", "lower"),
    ("final train val BPB", "metrics.final_train_val_bpb", "lower"),
    ("base eval val BPB", "metrics.base_eval_val_bpb", "lower"),
    ("final CORE", "metrics.final_core", "higher"),
    ("train runtime sec", "metrics.total_training_time_sec", "lower"),
    ("train iterations", "metrics.num_iterations", "lower"),
    ("train tokens", "metrics.total_training_tokens", "lower"),
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _get(obj: dict[str, Any], dotted: str, default=None):
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "-"
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


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _numeric(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


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


def _time_to_baseline_bpb(baseline: dict[str, Any], semdedup: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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

    return one(baseline), one(semdedup)


def _matches(cfg: dict[str, Any]) -> bool:
    for key, expected in FILTERS.items():
        if str(cfg.get(key, "")) != expected:
            return False
    kind = str(cfg.get("run_kind", ""))
    run_id = str(cfg.get("run_id", ""))
    if kind not in {"baseline", "semdedup"}:
        return False
    return run_id.endswith(f"_{kind}")


def _discover(run_root: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], list[Path]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    ignored_duplicates: list[Path] = []
    for config_path in sorted(run_root.glob("*/run_config.json")):
        cfg = _load_json(config_path)
        if not _matches(cfg):
            continue
        seed = str(cfg.get("SEED"))
        kind = str(cfg.get("run_kind"))
        run_dir = config_path.parent
        summary_path = run_dir / "run_summary.json"
        entry = {
            "run_dir": run_dir,
            "run_config": cfg,
            "run_summary": _load_json(summary_path),
            "run_summary_path": summary_path,
            "complete": summary_path.exists(),
        }
        current = grouped.setdefault(seed, {}).get(kind)
        if current is None:
            grouped[seed][kind] = entry
            continue
        current_mtime = current["run_summary_path"].stat().st_mtime if current["run_summary_path"].exists() else 0
        new_mtime = summary_path.stat().st_mtime if summary_path.exists() else 0
        if new_mtime >= current_mtime:
            ignored_duplicates.append(current["run_dir"])
            grouped[seed][kind] = entry
        else:
            ignored_duplicates.append(run_dir)
    return grouped, ignored_duplicates


def _render_table(headers: list[str], rows: list[list[Any]], aligns: list[str] | None = None) -> str:
    if aligns is None:
        aligns = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def _completed_pairs(grouped: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    pairs = []
    for seed in sorted(grouped, key=lambda s: int(s) if s.isdigit() else s):
        by_kind = grouped[seed]
        base = by_kind.get("baseline")
        semd = by_kind.get("semdedup")
        if not base or not semd or not base["complete"] or not semd["complete"]:
            continue
        base_summary = base["run_summary"]
        semd_summary = semd["run_summary"]
        base_ttt, semd_ttt = _time_to_baseline_bpb(base_summary, semd_summary)
        pairs.append({
            "seed": seed,
            "baseline": base,
            "semdedup": semd,
            "baseline_ttt": base_ttt,
            "semdedup_ttt": semd_ttt,
        })
    return pairs


def _render_common_params(pair: dict[str, Any] | None) -> str:
    if pair is None:
        return "No completed pairs found.\n"
    base_cfg = pair["baseline"]["run_config"]
    semd_cfg = pair["semdedup"]["run_config"]
    rows = []
    for key in COMMON_PARAM_KEYS:
        base_val = base_cfg.get(key, "-")
        semd_val = semd_cfg.get(key, "-")
        if key == "SEED":
            rows.append([key, "varies by repeat", "varies by repeat"])
        else:
            rows.append([key, base_val, semd_val])
    return _render_table(["Parameter", "Baseline", "SemDeDup"], rows, ["---", "---", "---"])


def _render_run_rows(pairs: list[dict[str, Any]]) -> str:
    rows = []
    for pair in pairs:
        b = pair["baseline"]["run_summary"]
        s = pair["semdedup"]["run_summary"]
        rows.append([
            pair["seed"],
            b.get("run_id"),
            s.get("run_id"),
            _get(b, "metrics.final_train_val_bpb"),
            _get(s, "metrics.final_train_val_bpb"),
            (_get(s, "metrics.final_train_val_bpb") or 0) - (_get(b, "metrics.final_train_val_bpb") or 0),
            _get(b, "metrics.final_core"),
            _get(s, "metrics.final_core"),
            (_get(s, "metrics.final_core") or 0) - (_get(b, "metrics.final_core") or 0),
            (_get(s, "metrics.total_training_time_sec") or 0) - (_get(b, "metrics.total_training_time_sec") or 0),
        ])
    return _render_table(
        ["Seed", "Baseline run", "SemDeDup run", "Baseline BPB", "SemDeDup BPB", "Delta BPB", "Baseline CORE", "SemDeDup CORE", "Delta CORE", "Delta runtime sec"],
        rows,
        ["---:", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )


def _render_aggregate(pairs: list[dict[str, Any]]) -> str:
    rows = []
    for label, path, direction in METRICS:
        base_vals = []
        semd_vals = []
        deltas = []
        for pair in pairs:
            b = _numeric(_get(pair["baseline"]["run_summary"], path))
            s = _numeric(_get(pair["semdedup"]["run_summary"], path))
            if b is None or s is None:
                continue
            base_vals.append(b)
            semd_vals.append(s)
            deltas.append(s - b)
        bm, bs = _mean_std(base_vals)
        sm, ss = _mean_std(semd_vals)
        dm, ds = _mean_std(deltas)
        rows.append([label, direction, _fmt_pm(bm, bs), _fmt_pm(sm, ss), _fmt_pm(dm, ds)])

    base_ttt_runtime = []
    semd_ttt_runtime = []
    delta_ttt_runtime = []
    base_ttt_steps = []
    semd_ttt_steps = []
    delta_ttt_steps = []
    for pair in pairs:
        br = _numeric(pair["baseline_ttt"].get("runtime_sec"))
        sr = _numeric(pair["semdedup_ttt"].get("runtime_sec"))
        bs = _numeric(pair["baseline_ttt"].get("step"))
        ss = _numeric(pair["semdedup_ttt"].get("step"))
        if br is not None and sr is not None:
            base_ttt_runtime.append(br)
            semd_ttt_runtime.append(sr)
            delta_ttt_runtime.append(sr - br)
        if bs is not None and ss is not None:
            base_ttt_steps.append(bs)
            semd_ttt_steps.append(ss)
            delta_ttt_steps.append(ss - bs)
    bm, bs = _mean_std(base_ttt_steps)
    sm, ss = _mean_std(semd_ttt_steps)
    dm, ds = _mean_std(delta_ttt_steps)
    rows.append(["steps to paired baseline BPB", "lower", _fmt_pm(bm, bs), _fmt_pm(sm, ss), _fmt_pm(dm, ds)])
    bm, bs = _mean_std(base_ttt_runtime)
    sm, ss = _mean_std(semd_ttt_runtime)
    dm, ds = _mean_std(delta_ttt_runtime)
    rows.append(["runtime sec to paired baseline BPB", "lower", _fmt_pm(bm, bs), _fmt_pm(sm, ss), _fmt_pm(dm, ds)])

    return _render_table(["Metric", "Better", "Baseline mean +/- sd", "SemDeDup mean +/- sd", "Paired delta mean +/- sd"], rows, ["---", "---", "---:", "---:", "---:"])


def _render_semdedup_stats(pair: dict[str, Any] | None) -> str:
    if pair is None:
        return "No SemDeDup manifest found.\n"
    manifest = _load_json(pair["semdedup"]["run_dir"] / "semdedup_manifest.json")
    rows = [
        ["input docs", _get(manifest, "input_train.docs")],
        ["output docs", _get(manifest, "output_train.docs")],
        ["removed docs", _get(manifest, "removed_docs")],
        ["input tokens", _get(manifest, "input_train.tokens")],
        ["output tokens", _get(manifest, "output_train.tokens")],
        ["removed tokens", _get(manifest, "removed_tokens")],
        ["doc keep ratio", _get(manifest, "keep_ratio_docs")],
        ["token keep ratio", _get(manifest, "keep_ratio_tokens")],
        ["dedup runtime sec", _get(manifest, "elapsed_sec")],
        ["embedding model", _get(manifest, "curator_config.model_identifier")],
        ["eps", _get(manifest, "curator_config.eps")],
        ["n clusters", _get(manifest, "curator_config.n_clusters")],
        ["distance metric", _get(manifest, "curator_config.distance_metric")],
        ["which to keep", _get(manifest, "curator_config.which_to_keep")],
    ]
    return _render_table(["SemDeDup statistic", "Value"], rows, ["---", "---:"])


def _render_per_task(pairs: list[dict[str, Any]]) -> str:
    task_deltas: dict[str, list[float]] = {}
    for pair in pairs:
        base_tasks = pair["baseline"]["run_summary"].get("per_task_core", {}) or {}
        semd_tasks = pair["semdedup"]["run_summary"].get("per_task_core", {}) or {}
        for task in sorted(set(base_tasks) | set(semd_tasks)):
            b = _numeric((base_tasks.get(task) or {}).get("centered"))
            s = _numeric((semd_tasks.get(task) or {}).get("centered"))
            if b is not None and s is not None:
                task_deltas.setdefault(task, []).append(s - b)
    rows = []
    for task, deltas in task_deltas.items():
        mean, std = _mean_std(deltas)
        rows.append([task, len(deltas), mean, std, min(deltas), max(deltas)])
    rows.sort(key=lambda row: (row[2] if row[2] is not None else 0.0))
    return _render_table(["CORE task", "N", "Delta mean", "Delta sd", "Min delta", "Max delta"], rows, ["---", "---:", "---:", "---:", "---:", "---:"])


def _render_incomplete(grouped: dict[str, dict[str, dict[str, Any]]]) -> str:
    rows = []
    for seed in sorted(grouped, key=lambda s: int(s) if s.isdigit() else s):
        kinds = grouped[seed]
        for kind in ("baseline", "semdedup"):
            entry = kinds.get(kind)
            if entry is None:
                rows.append([seed, kind, "missing", "-"])
            elif not entry["complete"]:
                rows.append([seed, kind, "running/incomplete", entry["run_dir"]])
    if not rows:
        return "All discovered repeat pairs are complete.\n"
    return _render_table(["Seed", "Kind", "Status", "Run dir"], rows, ["---:", "---", "---", "---"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-seed FineWeb-EDU SemDeDup report")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ecdf-svg", type=Path, default=None)
    parser.add_argument("--ecdf-stats", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    grouped, ignored_duplicates = _discover(run_root)
    pairs = _completed_pairs(grouped)
    first_pair = pairs[0] if pairs else None

    ecdf_svg = args.ecdf_svg.expanduser().resolve() if args.ecdf_svg else None
    ecdf_stats = _load_json(args.ecdf_stats.expanduser().resolve()) if args.ecdf_stats else {}
    ecdf_rel = os.path.relpath(ecdf_svg, args.output.expanduser().resolve().parent) if ecdf_svg and ecdf_svg.exists() else None

    content = [
        "# FineWeb-EDU SemDeDup Multi-Seed Repeat Report",
        "",
        "## Scope",
        "",
        f"Completed paired seeds: {len(pairs)}",
        "",
        "This report compares FineWeb-EDU baseline vs FineWeb-EDU SemDeDup only. Pilot/smoke runs are excluded. Repeated runs share the same dataset, tokenizer, shard order, validation shard, model depth, training horizon, and CORE evaluation settings; the model seed changes by repeat.",
        "",
        "## Common Parameters",
        "",
        _render_common_params(first_pair),
        "",
        "## Run-Level Results",
        "",
        _render_run_rows(pairs) if pairs else "No completed run pairs found.",
        "",
        "## Aggregate Paired Results",
        "",
        _render_aggregate(pairs) if pairs else "No completed run pairs found.",
        "",
        "## SemDeDup Data Reduction",
        "",
        _render_semdedup_stats(first_pair),
        "",
    ]
    if ecdf_rel:
        content.extend([
            "## SemDeDup Similarity ECDF",
            "",
            f"![SemDeDup similarity ECDF]({ecdf_rel})",
            "",
        ])
        if ecdf_stats:
            content.extend([
                _render_table(
                    ["ECDF statistic", "Value"],
                    [
                        ["total documents", ecdf_stats.get("total_documents")],
                        ["eps", ecdf_stats.get("eps")],
                        ["similarity threshold", ecdf_stats.get("similarity_threshold")],
                        ["documents below threshold", ecdf_stats.get("below_documents")],
                        ["removed documents", ecdf_stats.get("removed_documents")],
                        ["removed ratio", ecdf_stats.get("removed_ratio")],
                        ["raw max similarity", ecdf_stats.get("raw_max")],
                        ["raw >1.0 documents", ecdf_stats.get("raw_gt_one_documents")],
                        ["raw >1.0 ratio", ecdf_stats.get("raw_gt_one_ratio")],
                        ["raw ==1.0 documents", ecdf_stats.get("raw_exact_one_documents")],
                        ["sim >=0.999999 documents", ecdf_stats.get("near_one_documents")],
                        ["sim >=0.999999 ratio", ecdf_stats.get("near_one_ratio")],
                    ],
                    ["---", "---:"],
                ),
                "",
            ])
    content.extend([
        "## Per-Task CORE Delta",
        "",
        _render_per_task(pairs) if pairs else "No completed run pairs found.",
        "",
        "## Incomplete Or Missing Runs",
        "",
        _render_incomplete(grouped),
        "",
        "## Notes",
        "",
        "- BPB is the primary convergence-efficiency metric; lower is better.",
        "- CORE is the downstream-quality metric; interpret small average deltas together with per-task deltas and seed variance.",
        "- The ECDF x-axis is cosine similarity. For cosine distance SemDeDup, eps maps to similarity cutoff 1 - eps.",
        "- The visible jump near cosine similarity 1.0 is a point mass of exact/near-exact duplicates. Raw Curator scores can be slightly above 1.0 from floating-point roundoff; the plotted ECDF clips scores to [0, 1] and reports the affected count.",
        "- Manual audit files remain in each SemDeDup run directory: removed_samples.jsonl and kept_samples.jsonl. Repeated SemDeDup training runs that reuse the seed42 deduped data inherit the same data audit artifacts.",
    ])
    if ignored_duplicates:
        content.extend([
            "",
            "## Ignored Duplicate Runs",
            "",
            "The report kept the newest completed run per seed/kind and ignored:",
            "",
            "\n".join(f"- {p}" for p in ignored_duplicates),
        ])

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(content) + "\n")

    if args.json_output:
        payload = {
            "completed_pair_count": len(pairs),
            "seeds": [pair["seed"] for pair in pairs],
            "run_root": str(run_root),
            "output": str(output),
            "ecdf_svg": str(ecdf_svg) if ecdf_svg else None,
        }
        args.json_output.expanduser().resolve().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote multi-seed report: {output}")


if __name__ == "__main__":
    main()
