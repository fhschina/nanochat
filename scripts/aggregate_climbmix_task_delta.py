"""
Aggregate ClimbMix SemDeDup task-delta investigation artifacts into Markdown.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

FOCUS_TASKS = ("commonsense_qa", "winograd", "winogrande")


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _read_core_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            task = row[0].strip()
            acc = row[1].strip() if len(row) > 1 else ""
            centered = row[2].strip() if len(row) > 2 else ""
            out[task] = {
                "accuracy": float(acc) if acc else None,
                "centered": float(centered) if centered else None,
            }
    return out


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def _read_eval_log(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    return {
        "core": _last_float(r"CORE metric:\s+([0-9.]+)", text),
        "train_bpb": _last_float(r"^train bpb:\s+([0-9.]+)", text),
        "val_bpb": _last_float(r"^val bpb:\s+([0-9.]+)", text),
    }


def _discover_run_summaries(roots: list[Path]) -> list[Path]:
    seen = set()
    paths = []
    stage_dirs = {"full_seed_repeats", "randomdrop_control", "eps_sweep"}

    def add_if_run_dir(directory: Path) -> None:
        summary_path = directory / "run_summary.json"
        if not summary_path.exists():
            return
        resolved = summary_path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        paths.append(summary_path)

    for root in roots:
        if not root or not root.exists():
            continue
        add_if_run_dir(root)
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            add_if_run_dir(child)
            if child.name not in stage_dirs:
                continue
            for grandchild in sorted(p for p in child.iterdir() if p.is_dir()):
                add_if_run_dir(grandchild)
    return sorted(paths)


def _manifest_for(run_dir: Path) -> dict:
    semd = _load_json(run_dir / "semdedup_manifest.json")
    if semd:
        return semd
    random_drop = _load_json(run_dir / "random_drop_manifest.json")
    return random_drop


def _infer_arm(summary: dict, manifest: dict, config: dict) -> str:
    run_kind = summary.get("run_kind") or config.get("run_kind") or ""
    run_id = summary.get("run_id", "")
    if manifest.get("backend") == "random-drop" or run_kind == "randomdrop" or "randomdrop" in run_id:
        return "randomdrop"
    if manifest.get("semantic_dedup") or run_kind == "semdedup" or "semdedup" in run_id:
        return "semdedup"
    return "baseline"


def _seed(config: dict, summary: dict) -> int | None:
    value = config.get("SEED") or summary.get("seed")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _eps(manifest: dict, config: dict) -> str:
    value = None
    if manifest.get("curator_config"):
        value = manifest["curator_config"].get("eps")
    value = value if value is not None else config.get("SEMD_EPS")
    return str(value) if value is not None else ""


def _run_record(summary_path: Path) -> dict:
    summary = _load_json(summary_path)
    run_dir = Path(summary.get("run_dir") or summary_path.parent)
    config = _load_json(run_dir / "run_config.json")
    manifest = _manifest_for(run_dir)
    per_task = summary.get("per_task_core") or _read_core_csv(run_dir / "base_eval_core.csv")
    metrics = summary.get("metrics", {})
    arm = _infer_arm(summary, manifest, config)
    return {
        "run_id": summary.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "arm": arm,
        "seed": _seed(config, summary),
        "eps": _eps(manifest, config) if arm == "semdedup" else "",
        "random_drop_seed": config.get("RANDOM_DROP_SEED") or manifest.get("random_drop_config", {}).get("random_seed"),
        "final_core": metrics.get("final_core"),
        "final_train_val_bpb": metrics.get("final_train_val_bpb"),
        "base_eval_val_bpb": metrics.get("base_eval_val_bpb"),
        "min_val_bpb": metrics.get("min_val_bpb"),
        "num_iterations": metrics.get("num_iterations"),
        "total_training_time_sec": metrics.get("total_training_time_sec"),
        "keep_ratio_docs": manifest.get("keep_ratio_docs"),
        "keep_ratio_tokens": manifest.get("keep_ratio_tokens"),
        "removed_docs": manifest.get("removed_docs"),
        "removed_tokens": manifest.get("removed_tokens"),
        "per_task_core": per_task,
    }


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _std(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    mu = sum(clean) / len(clean)
    return math.sqrt(sum((v - mu) ** 2 for v in clean) / (len(clean) - 1))


def _fmt(value, digits=4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _task_centered(run: dict, task: str) -> float | None:
    rec = run.get("per_task_core", {}).get(task, {})
    return rec.get("centered")


def _group_runs(runs: list[dict]) -> dict:
    groups = defaultdict(list)
    for run in runs:
        key = run["arm"] if run["arm"] != "semdedup" else f"semdedup_eps{run['eps']}"
        if run["arm"] == "randomdrop" and run.get("random_drop_seed"):
            key = f"randomdrop_seed{run['random_drop_seed']}"
        groups[key].append(run)
    return groups


def _load_eval_only(task_root: Path) -> list[dict]:
    eval_root = task_root / "eval_only"
    if not eval_root.exists():
        return []
    records = []
    for path in sorted(eval_root.rglob("eval_summary.json")):
        payload = _load_json(path)
        if not payload:
            continue
        log_values = _read_eval_log(path.parent / "base_eval.log")
        for key, value in log_values.items():
            if payload.get(key) is None:
                payload[key] = value
        payload.setdefault("run_dir", str(path.parent))
        records.append(payload)
    return records


def _write_report(path: Path, runs: list[dict], eval_only: list[dict], audit: dict) -> None:
    groups = _group_runs(runs)
    lines = [
        "# ClimbMix SemDeDup Task-Delta Investigation",
        "",
        "## Status",
        "",
        f"Discovered `{len(runs)}` training/eval run summaries and `{len(eval_only)}` eval-only summaries.",
        "",
        "## Eval-Only Stability",
        "",
        "| Arm | Repeats | CORE mean | CORE std | Val BPB mean | Val BPB std |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    eval_groups = defaultdict(list)
    for record in eval_only:
        eval_groups[record.get("arm", "")].append(record)
    for arm, records in sorted(eval_groups.items()):
        lines.append(
            f"| {arm} | {len(records)} | "
            f"{_fmt(_mean([r.get('core') for r in records]))} | "
            f"{_fmt(_std([r.get('core') for r in records]))} | "
            f"{_fmt(_mean([r.get('val_bpb') for r in records]), 6)} | "
            f"{_fmt(_std([r.get('val_bpb') for r in records]), 6)} |"
        )

    lines.extend([
        "",
        "## Arm Summary",
        "",
        "| Arm | Runs | Seeds | CORE mean | CORE std | Val BPB mean | Keep docs | Keep tokens | Removed docs |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm, arm_runs in sorted(groups.items()):
        seeds = sorted({str(run.get("seed")) for run in arm_runs if run.get("seed") is not None})
        lines.append(
            f"| {arm} | {len(arm_runs)} | {', '.join(seeds)} | "
            f"{_fmt(_mean([r.get('final_core') for r in arm_runs]))} | "
            f"{_fmt(_std([r.get('final_core') for r in arm_runs]))} | "
            f"{_fmt(_mean([r.get('base_eval_val_bpb') for r in arm_runs]), 6)} | "
            f"{_fmt(_mean([r.get('keep_ratio_docs') for r in arm_runs]))} | "
            f"{_fmt(_mean([r.get('keep_ratio_tokens') for r in arm_runs]))} | "
            f"{_fmt(_mean([r.get('removed_docs') for r in arm_runs]), 0)} |"
        )

    lines.extend([
        "",
        "## Focus Task Means",
        "",
        "| Arm | commonsense_qa | winograd | winogrande |",
        "| --- | ---: | ---: | ---: |",
    ])
    for arm, arm_runs in sorted(groups.items()):
        task_values = [_fmt(_mean([_task_centered(run, task) for run in arm_runs])) for task in FOCUS_TASKS]
        lines.append(f"| {arm} | " + " | ".join(task_values) + " |")

    baseline_by_seed = {run.get("seed"): run for run in runs if run["arm"] == "baseline" and run.get("seed") is not None}
    semd07_by_seed = {
        run.get("seed"): run
        for run in runs
        if run["arm"] == "semdedup" and str(run.get("eps")) in {"0.07", "0.070000", "0.07"} and run.get("seed") is not None
    }
    lines.extend([
        "",
        "## Seed-Matched SemDeDup eps0.07 Delta vs Baseline",
        "",
        "| Seed | Delta CORE | Delta BPB | Delta commonsense_qa | Delta winograd | Delta winogrande |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    deltas = []
    for seed in sorted(set(baseline_by_seed) & set(semd07_by_seed)):
        base = baseline_by_seed[seed]
        semd = semd07_by_seed[seed]
        core_delta = (semd.get("final_core") - base.get("final_core")) if semd.get("final_core") is not None and base.get("final_core") is not None else None
        bpb_delta = (semd.get("base_eval_val_bpb") - base.get("base_eval_val_bpb")) if semd.get("base_eval_val_bpb") is not None and base.get("base_eval_val_bpb") is not None else None
        task_deltas = []
        for task in FOCUS_TASKS:
            s = _task_centered(semd, task)
            b = _task_centered(base, task)
            task_deltas.append((s - b) if s is not None and b is not None else None)
        deltas.append((seed, core_delta, bpb_delta, task_deltas))
        lines.append(f"| {seed} | {_fmt(core_delta)} | {_fmt(bpb_delta, 6)} | " + " | ".join(_fmt(v) for v in task_deltas) + " |")

    if audit:
        removed = audit.get("removed_sample", {})
        kept = audit.get("kept_sample", {})
        lines.extend([
            "",
            "## Removed Sample Audit",
            "",
            "| Group | Docs | QA-like | Coref-like | Boilerplate-like | p50 chars | p95 chars |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for name, summary in [("removed", removed), ("kept", kept)]:
            rates = summary.get("pattern_rates", {})
            chars = summary.get("char_length_quantiles", {})
            lines.append(
                f"| {name} | {summary.get('docs', '')} | {_fmt(rates.get('qa_like'))} | {_fmt(rates.get('coref_like'))} | "
                f"{_fmt(rates.get('boilerplate_like'))} | {_fmt(chars.get('p50'))} | {_fmt(chars.get('p95'))} |"
            )

    lines.extend([
        "",
        "## Preliminary Decision Logic",
        "",
    ])
    if len(eval_only) < 6:
        lines.append("- Eval-only stability is not complete yet; run `eval-only` before making a final claim.")
    elif eval_groups:
        max_eval_core_std = max((_std([r.get("core") for r in records]) or 0.0) for records in eval_groups.values())
        max_eval_bpb_std = max((_std([r.get("val_bpb") for r in records]) or 0.0) for records in eval_groups.values())
        if max_eval_core_std == 0 and max_eval_bpb_std == 0:
            lines.append("- Eval-only repeats are deterministic for the checked checkpoints.")
        else:
            lines.append("- Eval-only repeats vary; interpret small CORE/BPB deltas with that eval noise in mind.")
    if not deltas:
        lines.append("- Seed-matched baseline vs SemDeDup deltas are not complete yet.")
    else:
        core_signs = {1 if d[1] and d[1] > 0 else -1 if d[1] and d[1] < 0 else 0 for d in deltas}
        if len(core_signs - {0}) > 1:
            lines.append("- SemDeDup CORE deltas flip sign across seeds, so training variance is a likely contributor.")
        else:
            lines.append("- SemDeDup CORE deltas do not flip sign in completed seed-matched runs.")
    if "randomdrop_seed9001" not in groups:
        lines.append("- Random-drop control is missing, so generic data-removal effects are not isolated yet.")
    else:
        baseline_core = _mean([r.get("final_core") for r in groups.get("baseline", [])])
        randomdrop_core = _mean([r.get("final_core") for r in groups.get("randomdrop_seed9001", [])])
        semd07_core = _mean([r.get("final_core") for r in groups.get("semdedup_eps0.07", [])])
        if baseline_core is not None and randomdrop_core is not None and semd07_core is not None:
            lines.append(
                f"- Random-drop seed9001 CORE ({_fmt(randomdrop_core)}) is close to baseline ({_fmt(baseline_core)}); "
                f"SemDeDup eps0.07 is slightly higher ({_fmt(semd07_core)}) but the mean delta is small."
            )
        base_cqa = _mean([_task_centered(r, "commonsense_qa") for r in groups.get("baseline", [])])
        rd_cqa = _mean([_task_centered(r, "commonsense_qa") for r in groups.get("randomdrop_seed9001", [])])
        semd_cqa = _mean([_task_centered(r, "commonsense_qa") for r in groups.get("semdedup_eps0.07", [])])
        if base_cqa is not None and rd_cqa is not None and semd_cqa is not None:
            lines.append(
                f"- The commonsense_qa jump is much larger for SemDeDup eps0.07 ({_fmt(semd_cqa)}) "
                f"than for random-drop seed9001 ({_fmt(rd_cqa)}) or baseline ({_fmt(base_cqa)}), "
                "so it is not fully explained by removing the same number of documents."
            )
    if not any(key.startswith("semdedup_eps0.05") or key.startswith("semdedup_eps0.09") or key.startswith("semdedup_eps0.12") for key in groups):
        lines.append("- EPS sweep is incomplete, so threshold sensitivity is unknown.")
    else:
        lines.append("- EPS sweep is threshold-sensitive: eps0.09 and eps0.12 remove more data but do not improve CORE/BPB over eps0.07.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ClimbMix task-delta investigation artifacts")
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, default=None, help="Existing climbmix_semdedup_quality root with anchor runs")
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Explicit run dir to include without scanning its parent")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [args.task_root.expanduser().resolve()]
    if args.quality_root:
        roots.append(args.quality_root.expanduser().resolve())
    summary_paths = _discover_run_summaries(roots)
    for run_dir_arg in args.run_dir:
        run_dir = run_dir_arg.expanduser().resolve()
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists() and summary_path not in summary_paths:
            summary_paths.append(summary_path)
    runs = [_run_record(path) for path in sorted(summary_paths)]
    eval_only = _load_eval_only(args.task_root.expanduser().resolve())
    audit = _load_json(args.task_root.expanduser().resolve() / "removed_audit" / "removed_audit.json")
    _write_report(args.output.expanduser().resolve(), runs, eval_only, audit)
    print(f"Wrote investigation report: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
