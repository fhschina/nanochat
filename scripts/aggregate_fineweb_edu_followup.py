"""Aggregate FineWeb-EDU SemDeDup follow-up artifacts into a Markdown report."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


EPS_VALUES = ("0.03", "0.05", "0.07", "0.09", "0.10")
DESIRED_SEEDS = (42, 43, 44, 45, 46)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _get(obj: dict[str, Any], dotted: str, default=None):
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_float(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _as_int(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt(value, digits=6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _std(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    mu = sum(clean) / len(clean)
    return math.sqrt(sum((v - mu) ** 2 for v in clean) / (len(clean) - 1))


def _read_core_csv(path: Path) -> dict[str, dict[str, float | None]]:
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
            out[task] = {"accuracy": _as_float(acc), "centered": _as_float(centered)}
    return out


def _manifest_for(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "semdedup_manifest.json") or _load_json(run_dir / "random_drop_manifest.json")


def _infer_arm(summary: dict[str, Any], config: dict[str, Any], manifest: dict[str, Any]) -> str:
    kind = str(summary.get("run_kind") or config.get("run_kind") or "")
    run_id = str(summary.get("run_id") or config.get("run_id") or "")
    if kind == "randomdrop" or manifest.get("backend") == "random-drop" or "randomdrop" in run_id:
        return "randomdrop"
    if kind == "semdedup" or manifest.get("semantic_dedup") or "semdedup" in run_id:
        return "semdedup"
    return "baseline"


def _eps(manifest: dict[str, Any], config: dict[str, Any]) -> str:
    value = _get(manifest, "curator_config.eps")
    if value is None:
        value = config.get("SEMD_EPS")
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _seed(config: dict[str, Any], summary: dict[str, Any]) -> int | None:
    return _as_int(config.get("SEED") or summary.get("seed"))


def _run_record(summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    run_dir = Path(summary.get("run_dir") or summary_path.parent)
    config = _load_json(run_dir / "run_config.json")
    manifest = _manifest_for(run_dir)
    metrics = summary.get("metrics") or {}
    runtime = _as_float(metrics.get("total_training_time_sec"))
    tokens = _as_float(metrics.get("total_training_tokens"))
    iterations = _as_float(metrics.get("num_iterations"))
    arm = _infer_arm(summary, config, manifest)
    random_cfg = manifest.get("random_drop_config") or {}
    selection_mode = random_cfg.get("selection_mode")
    target_removed_docs = random_cfg.get("target_removed_docs") or config.get("RANDOM_DROP_REMOVED_DOCS")
    target_removed_tokens = random_cfg.get("target_removed_tokens") or config.get("RANDOM_DROP_REMOVED_TOKENS")
    if arm == "randomdrop" and not selection_mode:
        selection_mode = "token_matched" if target_removed_tokens else "doc_matched"
    return {
        "run_id": summary.get("run_id") or run_dir.name,
        "run_dir": str(run_dir),
        "arm": arm,
        "seed": _seed(config, summary),
        "eps": _eps(manifest, config) if arm == "semdedup" else "",
        "random_drop_seed": _as_int(random_cfg.get("random_seed") or config.get("RANDOM_DROP_SEED")),
        "random_selection_mode": selection_mode,
        "target_removed_docs": _as_int(target_removed_docs),
        "target_removed_tokens": _as_int(target_removed_tokens),
        "val_bpb": _as_float(metrics.get("base_eval_val_bpb")),
        "train_val_bpb": _as_float(metrics.get("final_train_val_bpb")),
        "min_val_bpb": _as_float(metrics.get("min_val_bpb")),
        "core": _as_float(metrics.get("final_core")),
        "runtime_sec": runtime,
        "tokens": tokens,
        "iterations": iterations,
        "tok_per_sec": (tokens / runtime if tokens is not None and runtime else None),
        "step_dt_sec": (runtime / iterations if runtime is not None and iterations else None),
        "removed_docs": _as_int(manifest.get("removed_docs")),
        "removed_tokens": _as_int(manifest.get("removed_tokens")),
        "keep_ratio_docs": _as_float(manifest.get("keep_ratio_docs")),
        "keep_ratio_tokens": _as_float(manifest.get("keep_ratio_tokens")),
        "preprocess_elapsed_sec": _as_float(manifest.get("elapsed_sec")),
        "preprocess_metadata": _get(manifest, "backend_result.metadata", {}),
        "per_task_core": summary.get("per_task_core") or _read_core_csv(run_dir / "base_eval_core.csv"),
    }


def _discover_runs(roots: list[Path]) -> list[dict[str, Any]]:
    seen = set()
    records = []
    for root in roots:
        if not root.exists():
            continue
        for summary_path in sorted(root.rglob("run_summary.json")):
            resolved = summary_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records.append(_run_record(summary_path))
    return records


def _select_promoted_eps(runs: list[dict[str, Any]]) -> dict[str, Any]:
    seed42 = [r for r in runs if r["arm"] == "semdedup" and r.get("seed") == 42 and r.get("val_bpb") is not None]
    ref = next((r for r in seed42 if r.get("eps") == "0.07"), None)
    candidates = []
    if ref and ref.get("core") is not None:
        for run in seed42:
            if run.get("eps") == "0.07":
                continue
            bpb = run.get("val_bpb")
            core = run.get("core")
            if bpb is None or core is None:
                continue
            eligible = bpb <= ref["val_bpb"] - 0.0002 and core >= ref["core"] - 0.005
            run = dict(run)
            run["promotion_eligible"] = eligible
            candidates.append(run)
    eligible = [r for r in candidates if r.get("promotion_eligible")]
    promoted = "0.07"
    promoted_run = ref
    if eligible:
        best = eligible[0]
        for run in eligible[1:]:
            if run["val_bpb"] < best["val_bpb"] - 0.0001:
                best = run
            elif abs(run["val_bpb"] - best["val_bpb"]) < 0.0001:
                if (run.get("removed_tokens") or 10**30) < (best.get("removed_tokens") or 10**30):
                    best = run
        promoted = best.get("eps") or "0.07"
        promoted_run = best
    return {"promoted_eps": promoted, "reference": ref, "candidates": candidates, "promoted_run": promoted_run}


def _task_value(run: dict[str, Any], task_pattern: str) -> float | None:
    per_task = run.get("per_task_core") or {}
    for task, payload in per_task.items():
        if task_pattern.lower() in task.lower():
            return _as_float(payload.get("centered") if isinstance(payload, dict) else None)
    return None


def _dmon_summary(log_dir: Path) -> list[dict[str, Any]]:
    out = []
    if not log_dir.exists():
        return out
    for path in sorted(log_dir.glob("dmon.*.log")):
        cols = None
        sm_values = []
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                maybe = stripped.lstrip("#").split()
                if "sm" in maybe:
                    cols = maybe
                continue
            parts = stripped.split()
            if cols and len(parts) >= len(cols):
                try:
                    sm_idx = cols.index("sm")
                    sm_values.append(float(parts[sm_idx]))
                except (ValueError, IndexError):
                    pass
            else:
                numeric = [float(x) for x in parts if re.fullmatch(r"-?\d+(\.\d+)?", x)]
                if len(numeric) >= 2:
                    sm_values.append(numeric[-2])
        out.append({
            "path": str(path),
            "samples": len(sm_values),
            "avg_sm_util": _mean(sm_values),
            "max_sm_util": max(sm_values) if sm_values else None,
        })
    return out


def _load_audits(followup_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    removed = []
    pairs = []
    if followup_root.exists():
        for path in sorted(followup_root.rglob("removed_audit.json")):
            payload = _load_json(path)
            if payload:
                payload["path"] = str(path)
                removed.append(payload)
        for path in sorted(followup_root.rglob("pair_audit_summary.json")):
            payload = _load_json(path)
            if payload:
                payload["path"] = str(path)
                pairs.append(payload)
    return removed, pairs


def _run_key(run: dict[str, Any]) -> str:
    if run["arm"] == "semdedup":
        return f"semdedup_eps{run.get('eps')}"
    if run["arm"] == "randomdrop":
        target = run.get("target_removed_tokens") if run.get("random_selection_mode") == "token_matched" else run.get("target_removed_docs")
        return f"randomdrop_{run.get('random_selection_mode')}_rd{run.get('random_drop_seed')}_target{target}"
    return run["arm"]


def _is_formal_run(run: dict[str, Any]) -> bool:
    run_id = str(run.get("run_id") or "").lower()
    run_dir = str(run.get("run_dir") or "").lower()
    return not any(marker in run_id or marker in run_dir for marker in ("pilot", "smoke"))


def _prefer_formal_by_seed(runs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for run in runs:
        seed = run.get("seed")
        if seed is None:
            continue
        current = out.get(seed)
        if current is None:
            out[seed] = run
            continue
        if _is_formal_run(run) and not _is_formal_run(current):
            out[seed] = run
    return out


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    runs = payload["runs"]
    formal_runs = [run for run in runs if _is_formal_run(run)]
    promotion = payload["promotion"]
    removed_audits = payload["removed_audits"]
    pair_audits = payload["pair_audits"]
    dmon = payload["dmon"]
    groups = defaultdict(list)
    for run in formal_runs:
        groups[_run_key(run)].append(run)

    lines = [
        "# FineWeb-EDU SemDeDup Follow-up Report",
        "",
        "Manual precision not completed. Heuristic removed/kept audits are triage signals only.",
        "",
        "## Status",
        "",
        f"Discovered `{len(runs)}` run summaries across QUALITY_ROOT and FOLLOWUP_ROOT.",
        f"Promoted eps by rule: `{promotion['promoted_eps']}`.",
        "",
        "## EPS Sweep and Promotion",
        "",
        "Promotion rule: versus eps0.07 seed42, require BPB improvement >= 0.0002 and CORE no worse than 0.005; choose lowest BPB, tie within 0.0001 goes to fewer removed tokens.",
        "",
        "| EPS | Seed | Val BPB | CORE | Removed docs | Removed tokens | Eligible | Run |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    ref = promotion.get("reference")
    if ref:
        lines.append(f"| 0.07 | 42 | {_fmt(ref.get('val_bpb'))} | {_fmt(ref.get('core'), 4)} | {_fmt(ref.get('removed_docs'))} | {_fmt(ref.get('removed_tokens'))} | reference | `{ref.get('run_id')}` |")
    for run in sorted(promotion.get("candidates") or [], key=lambda r: r.get("eps") or ""):
        lines.append(
            f"| {run.get('eps')} | {run.get('seed')} | {_fmt(run.get('val_bpb'))} | {_fmt(run.get('core'), 4)} | "
            f"{_fmt(run.get('removed_docs'))} | {_fmt(run.get('removed_tokens'))} | {run.get('promotion_eligible')} | `{run.get('run_id')}` |"
        )

    lines.extend(["", "## Arm Summary", "", "| Arm | Runs | Seeds | Val BPB mean | CORE mean | Removed docs | Removed tokens |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"])
    for key, group in sorted(groups.items()):
        seeds = sorted({str(r.get("seed")) for r in group if r.get("seed") is not None})
        lines.append(
            f"| {key} | {len(group)} | {', '.join(seeds)} | {_fmt(_mean([r.get('val_bpb') for r in group]))} | "
            f"{_fmt(_mean([r.get('core') for r in group]), 4)} | {_fmt(_mean([r.get('removed_docs') for r in group]), 0)} | {_fmt(_mean([r.get('removed_tokens') for r in group]), 0)} |"
        )

    baseline_by_seed = _prefer_formal_by_seed([r for r in runs if r["arm"] == "baseline"])
    promoted_by_seed = _prefer_formal_by_seed([r for r in runs if r["arm"] == "semdedup" and r.get("eps") == promotion["promoted_eps"]])
    lines.extend(["", "## Seed Expansion", "", "| Seed | Baseline | Promoted SemDeDup | Delta BPB | Delta CORE |", "| ---: | --- | --- | ---: | ---: |"])
    for seed in DESIRED_SEEDS:
        base = baseline_by_seed.get(seed)
        semd = promoted_by_seed.get(seed)
        delta_bpb = (semd.get("val_bpb") - base.get("val_bpb")) if base and semd and base.get("val_bpb") is not None and semd.get("val_bpb") is not None else None
        delta_core = (semd.get("core") - base.get("core")) if base and semd and base.get("core") is not None and semd.get("core") is not None else None
        lines.append(f"| {seed} | {base.get('run_id') if base else 'missing'} | {semd.get('run_id') if semd else 'missing'} | {_fmt(delta_bpb)} | {_fmt(delta_core, 4)} |")

    lines.extend(["", "## BoolQ Analysis", "", "| Seed | Baseline BoolQ | Promoted SemDeDup BoolQ | Delta |", "| ---: | ---: | ---: | ---: |"])
    for seed in DESIRED_SEEDS:
        base = baseline_by_seed.get(seed)
        semd = promoted_by_seed.get(seed)
        b = _task_value(base, "boolq") if base else None
        s = _task_value(semd, "boolq") if semd else None
        lines.append(f"| {seed} | {_fmt(b, 4)} | {_fmt(s, 4)} | {_fmt((s - b) if s is not None and b is not None else None, 4)} |")
    if removed_audits:
        lines.extend(["", "Removed/kept BoolQ-like heuristic rates:", "", "| Audit | Removed | Kept |", "| --- | ---: | ---: |"])
        for audit in removed_audits:
            removed_rate = _get(audit, "removed_sample.pattern_rates.boolq_like")
            kept_rate = _get(audit, "kept_sample.pattern_rates.boolq_like")
            lines.append(f"| `{Path(audit['path']).parent.name}` | {_fmt(removed_rate, 4)} | {_fmt(kept_rate, 4)} |")

    lines.extend(["", "## Random-drop Controls", "", "| Run | Mode | RD seed | Train seed | Target docs | Target tokens | Val BPB | CORE |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for run in sorted([r for r in runs if r["arm"] == "randomdrop"], key=lambda r: (r.get("random_selection_mode") or "", r.get("random_drop_seed") or 0, r.get("seed") or 0)):
        lines.append(
            f"| `{run.get('run_id')}` | {run.get('random_selection_mode')} | {run.get('random_drop_seed')} | {run.get('seed')} | "
            f"{_fmt(run.get('target_removed_docs'))} | {_fmt(run.get('target_removed_tokens'))} | {_fmt(run.get('val_bpb'))} | {_fmt(run.get('core'), 4)} |"
        )

    lines.extend(["", "## Audit Artifacts", "", f"Removed/kept heuristic audits: `{len(removed_audits)}`. Pair audit sheets: `{len(pair_audits)}`."])
    if pair_audits:
        lines.extend(["", "| Pair audit | Mode | Records | Degradation |", "| --- | --- | ---: | --- |"])
        for audit in pair_audits:
            lines.append(f"| `{Path(audit['path']).parent.name}` | {audit.get('mode')} | {_fmt(audit.get('record_count'))} | {audit.get('degradation_reason') or ''} |")

    lines.extend(["", "## Runtime", "", "| Run | Arm | Seed | Preprocess sec | Train sec | tok/sec | step dt sec |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for run in sorted(runs, key=lambda r: (r.get("arm"), r.get("eps") or "", r.get("seed") or -1, r.get("run_id") or "")):
        if run.get("runtime_sec") is None and run.get("preprocess_elapsed_sec") is None:
            continue
        lines.append(
            f"| `{run.get('run_id')}` | {run.get('arm')}{(' eps' + run.get('eps')) if run.get('eps') else ''} | {run.get('seed')} | "
            f"{_fmt(run.get('preprocess_elapsed_sec'), 1)} | {_fmt(run.get('runtime_sec'), 1)} | {_fmt(run.get('tok_per_sec'), 1)} | {_fmt(run.get('step_dt_sec'), 3)} |"
        )
    lines.extend(["", "GPU dmon logs:"])
    if not dmon:
        lines.append("- unavailable for existing runs; new follow-up runner will write dmon logs when long tasks start.")
    else:
        for item in dmon:
            lines.append(f"- `{item['path']}` samples={item['samples']} avg_sm={_fmt(item.get('avg_sm_util'), 2)} max_sm={_fmt(item.get('max_sm_util'), 2)}")

    missing = []
    for eps in EPS_VALUES:
        if not any(r for r in runs if r["arm"] == "semdedup" and r.get("eps") == eps and r.get("seed") == 42):
            missing.append(f"eps {eps} seed42")
    for seed in (45, 46):
        if seed not in baseline_by_seed:
            missing.append(f"baseline seed{seed}")
    lines.extend(["", "## Remaining Work", ""])
    if missing:
        for item in missing:
            lines.append(f"- Missing or not yet summarized: {item}.")
    else:
        lines.append("- No required EPS seed42 or baseline expansion summaries are missing from discovered artifacts.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FineWeb-EDU SemDeDup follow-up")
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--followup-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_root = args.quality_root.expanduser().resolve()
    followup_root = args.followup_root.expanduser().resolve()
    runs = _discover_runs([quality_root, followup_root])
    promotion = _select_promoted_eps(runs)
    removed_audits, pair_audits = _load_audits(followup_root)
    payload = {
        "quality_root": str(quality_root),
        "followup_root": str(followup_root),
        "runs": runs,
        "promotion": promotion,
        "removed_audits": removed_audits,
        "pair_audits": pair_audits,
        "dmon": _dmon_summary(followup_root / "logs"),
    }
    output = args.output.expanduser().resolve()
    _write_report(output, payload)
    json_output = args.json_output.expanduser().resolve() if args.json_output else output.with_suffix(".json")
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote follow-up report: {output}")
    print(f"Wrote follow-up JSON: {json_output}")


if __name__ == "__main__":
    main()
