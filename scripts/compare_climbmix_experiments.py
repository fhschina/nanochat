"""
Compare a no-SemDeDup ClimbMix run with a SemDeDup ClimbMix run.

The script is intentionally tolerant of partial smoke runs: missing logs or
metrics are rendered as "-" rather than failing, while missing run directories
still fail fast.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _load_core_csv(path: Path) -> dict[str, dict[str, float]]:
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


def _first_bpb_at_or_below(curve: list[dict[str, Any]], target: float | None) -> dict[str, Any] | None:
    if target is None:
        return None
    for point in curve:
        bpb = point.get("bpb")
        if bpb is not None and bpb <= target:
            return point
    return None


def _tokens_per_step(summary: dict[str, Any]) -> float | None:
    tokens = _get(summary, "metrics.total_training_tokens")
    steps = _get(summary, "metrics.num_iterations")
    if tokens is None or steps in (None, 0):
        return None
    return tokens / steps


def _runtime_per_step(summary: dict[str, Any]) -> float | None:
    runtime = _get(summary, "metrics.total_training_time_sec")
    steps = _get(summary, "metrics.num_iterations")
    if runtime is None or steps in (None, 0):
        return None
    return runtime / steps


def _time_to_target(summary: dict[str, Any], target: float | None) -> dict[str, Any]:
    point = _first_bpb_at_or_below(_get(summary, "curves.val_bpb", []), target)
    if point is None:
        return {"step": None, "tokens": None, "runtime_sec": None}
    step = point.get("step")
    tokens_per_step = _tokens_per_step(summary)
    runtime_per_step = _runtime_per_step(summary)
    return {
        "step": step,
        "tokens": step * tokens_per_step if step is not None and tokens_per_step is not None else None,
        "runtime_sec": step * runtime_per_step if step is not None and runtime_per_step is not None else None,
    }


def _metric_rows(baseline: dict[str, Any], semdedup: dict[str, Any], semd_manifest: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    baseline_target_bpb = _get(baseline, "metrics.final_val_bpb") or _get(baseline, "metrics.min_val_bpb")
    baseline_ttt = _time_to_target(baseline, baseline_target_bpb)
    semd_ttt = _time_to_target(semdedup, baseline_target_bpb)
    rows = [
        ("run id", baseline.get("run_id"), semdedup.get("run_id")),
        ("model tag", baseline.get("model_tag"), semdedup.get("model_tag")),
        ("data docs", _get(baseline, "data_stats.docs"), _get(semdedup, "data_stats.docs")),
        ("data tokens", _get(baseline, "data_stats.tokens"), _get(semdedup, "data_stats.tokens")),
        ("data chars", _get(baseline, "data_stats.chars"), _get(semdedup, "data_stats.chars")),
        ("SemDeDup input docs", "-", _get(semd_manifest, "input_train.docs")),
        ("SemDeDup output docs", "-", _get(semd_manifest, "output_train.docs")),
        ("SemDeDup input tokens", "-", _get(semd_manifest, "input_train.tokens")),
        ("SemDeDup output tokens", "-", _get(semd_manifest, "output_train.tokens")),
        ("doc keep ratio", "-", _get(semd_manifest, "keep_ratio_docs")),
        ("token keep ratio", "-", _get(semd_manifest, "keep_ratio_tokens")),
        ("removed docs", "-", _get(semd_manifest, "removed_docs")),
        ("removed tokens", "-", _get(semd_manifest, "removed_tokens")),
        ("dedup runtime sec", "-", _get(semd_manifest, "elapsed_sec")),
        ("embedding model", "-", _get(semd_manifest, "curator_config.model_identifier")),
        ("eps", "-", _get(semd_manifest, "curator_config.eps")),
        ("n clusters", "-", _get(semd_manifest, "curator_config.n_clusters")),
        ("train iterations", _get(baseline, "metrics.num_iterations"), _get(semdedup, "metrics.num_iterations")),
        ("train tokens", _get(baseline, "metrics.total_training_tokens"), _get(semdedup, "metrics.total_training_tokens")),
        ("train runtime sec", _get(baseline, "metrics.total_training_time_sec"), _get(semdedup, "metrics.total_training_time_sec")),
        ("min val BPB", _get(baseline, "metrics.min_val_bpb"), _get(semdedup, "metrics.min_val_bpb")),
        ("final train val BPB", _get(baseline, "metrics.final_train_val_bpb"), _get(semdedup, "metrics.final_train_val_bpb")),
        ("base eval val BPB", _get(baseline, "metrics.base_eval_val_bpb"), _get(semdedup, "metrics.base_eval_val_bpb")),
        ("final CORE", _get(baseline, "metrics.final_core"), _get(semdedup, "metrics.final_core")),
        (f"steps to baseline BPB <= {_fmt(baseline_target_bpb)}", baseline_ttt["step"], semd_ttt["step"]),
        ("tokens to baseline BPB", baseline_ttt["tokens"], semd_ttt["tokens"]),
        ("runtime sec to baseline BPB", baseline_ttt["runtime_sec"], semd_ttt["runtime_sec"]),
    ]
    return rows


def _render_table(rows: list[tuple[str, Any, Any]]) -> str:
    lines = ["| Metric | No SemDeDup | SemDeDup | Delta |", "|---|---:|---:|---:|"]
    for metric, base, semd in rows:
        delta = "-"
        if isinstance(base, (int, float)) and isinstance(semd, (int, float)):
            delta = _fmt(semd - base)
        lines.append(f"| {metric} | {_fmt(base)} | {_fmt(semd)} | {delta} |")
    return "\n".join(lines)


def _render_core_delta(baseline_csv: dict[str, dict[str, float]], semd_csv: dict[str, dict[str, float]]) -> str:
    tasks = sorted(set(baseline_csv) | set(semd_csv))
    if not tasks:
        return "No per-task CORE CSVs found.\n"
    lines = ["| CORE Task | No SemDeDup Centered | SemDeDup Centered | Delta |", "|---|---:|---:|---:|"]
    for task in tasks:
        base = baseline_csv.get(task, {}).get("centered")
        semd = semd_csv.get(task, {}).get("centered")
        delta = semd - base if isinstance(base, float) and isinstance(semd, float) else None
        lines.append(f"| {task} | {_fmt(base)} | {_fmt(semd)} | {_fmt(delta)} |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ClimbMix no-SemDeDup and SemDeDup run directories")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--semdedup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = args.baseline.expanduser().resolve()
    semdedup_dir = args.semdedup.expanduser().resolve()
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline run dir not found: {baseline_dir}")
    if not semdedup_dir.is_dir():
        raise FileNotFoundError(f"SemDeDup run dir not found: {semdedup_dir}")

    baseline = _load_json(baseline_dir / "run_summary.json")
    semdedup = _load_json(semdedup_dir / "run_summary.json")
    baseline["data_stats"] = _load_json(baseline_dir / "data_stats.json")
    semdedup["data_stats"] = _load_json(semdedup_dir / "data_stats.json")
    semd_manifest = _load_json(semdedup_dir / "semdedup_manifest.json")

    baseline_csv = _load_core_csv(baseline_dir / "base_eval_core.csv")
    semd_csv = _load_core_csv(semdedup_dir / "base_eval_core.csv")

    content = [
        "# ClimbMix SemDeDup Experiment Comparison",
        "",
        "## Summary Table",
        "",
        _render_table(_metric_rows(baseline, semdedup, semd_manifest)),
        "",
        "## Per-Task CORE Delta",
        "",
        _render_core_delta(baseline_csv, semd_csv),
        "",
        "## Notes",
        "",
        "- BPB is the primary convergence-efficiency curve; lower is better.",
        "- CORE is the downstream-quality metric; inspect per-task deltas before reading small average deltas as meaningful.",
        "- SemDeDup quality should also be checked through removed/kept sample audit files in the SemDeDup run directory.",
    ]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(content) + "\n")
    print(f"Wrote comparison report: {output}")


if __name__ == "__main__":
    main()
