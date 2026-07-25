"""Generate a single-seed, fuzzy-only pioneer report for the Fortified NanoChat experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.compare_fortified_fuzzy_ab import (
    EXPECTED_STEPS,
    EXPECTED_TOKENS,
    _fmt,
    _plot_metric,
    _save_figure,
    _table,
    atomic_json,
    load_json,
    plot_bpb_time,
    plot_dataset_audits,
    sha256_file,
    utc_now,
)


def plot_final_core(record: dict[str, Any], output: Path) -> None:
    value = record["metrics"].get("final_core")
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.bar(["fuzzy seed 42"], [value], color="#1f77b4", alpha=0.82)
    ax.scatter([0], [value], color="black", s=22, zorder=3)
    ax.set_ylabel("Final full CORE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)


def plot_task_scores(record: dict[str, Any], output: Path) -> None:
    tasks = record.get("core_tasks", {})
    ordered = sorted(
        ((task, float(values["centered"])) for task, values in tasks.items()),
        key=lambda row: row[1],
    )
    if not ordered:
        return
    y = np.arange(len(ordered))
    values = [row[1] for row in ordered]
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]
    fig, ax = plt.subplots(figsize=(9, max(5, len(ordered) * 0.25)))
    ax.hlines(y, 0, values, color=colors)
    ax.scatter(values, y, color=colors, s=20)
    ax.set_yticks(y, [row[0] for row in ordered])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Centered accuracy — fuzzy seed 42")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)


def generate_pioneer_report(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "full" / "fuzzy" / f"seed_{args.seed}" / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Pioneer run is incomplete: {summary_path}")
    record = load_json(summary_path)
    if record.get("arm") != "fuzzy" or int(record.get("seed", -1)) != args.seed:
        raise RuntimeError("Pioneer summary does not describe the requested fuzzy seed")
    metrics = record["metrics"]
    if metrics.get("num_iterations") != EXPECTED_STEPS:
        raise RuntimeError(f"Expected {EXPECTED_STEPS} optimization steps")
    if metrics.get("training_tokens") != EXPECTED_TOKENS:
        raise RuntimeError(f"Expected {EXPECTED_TOKENS} training tokens")
    val_curve = record["curves"].get("val_bpb", [])
    if not val_curve or val_curve[0]["step"] != 0 or val_curve[-1]["step"] != EXPECTED_STEPS:
        raise RuntimeError("Pioneer validation curve is incomplete")

    records = [record]
    _plot_metric(records, "val_bpb", "bpb", "Validation BPB", output / "val_bpb_vs_tokens.svg")
    _plot_metric(records, "train_loss", "loss", "Training loss EMA (diagnostic)", output / "train_loss_vs_tokens.svg")
    _plot_metric(records, "online_core", "core", "Online CORE (max 500/task)", output / "online_core_vs_tokens.svg")
    plot_bpb_time(records, output / "val_bpb_vs_time.svg")
    plot_final_core(record, output / "final_core.svg")
    plot_task_scores(record, output / "core_task_scores.svg")
    plot_dataset_audits(args.component_audit, args.pair_audit, output)

    fuzzy_manifest = load_json(args.fuzzy_manifest)
    dedup_manifest = load_json(args.dedup_manifest)
    selection = fuzzy_manifest.get("selection", {})
    full_fuzzy = dedup_manifest.get("stats", {}).get("post_fuzzy", {})
    subset_rows = [
        [subset, f"{values.get('docs', 0):,}", f"{values.get('tokens', 0):,}"]
        for subset, values in sorted(fuzzy_manifest.get("by_subset", {}).items())
    ]
    length_rows = [
        [
            label,
            f"{fuzzy_manifest.get('by_length', {}).get(label, {}).get('docs', 0):,}",
            f"{fuzzy_manifest.get('by_length', {}).get(label, {}).get('tokens', 0):,}",
        ]
        for label in ("lt128", "128_511", "512_2047", "ge2048")
    ]
    qualitative_rows = []
    if args.pair_audit.is_file():
        for line in args.pair_audit.read_text(encoding="utf-8").splitlines()[:5]:
            row = json.loads(line)
            clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")[:180]
            qualitative_rows.append([
                _fmt(row.get("char_ngram_jaccard"), 3),
                row.get("component_size"),
                clean(row.get("removed_preview", "")),
                clean(row.get("keeper_preview", "")),
            ])

    final_bpb = metrics.get("final_train_val_bpb")
    final_eval_bpb = metrics.get("final_eval_val_bpb")
    final_core = metrics.get("final_core")
    minimum_bpb = min(float(point["bpb"]) for point in val_curve)
    result = {
        "generated_at": utc_now(),
        "report_type": "pioneer_fuzzy_only",
        "seed": args.seed,
        "comparison_available": False,
        "run": record,
        "metrics": {
            "final_train_val_bpb": final_bpb,
            "final_eval_val_bpb": final_eval_bpb,
            "minimum_validation_bpb": minimum_bpb,
            "final_core": final_core,
            "training_time_sec": metrics.get("training_time_sec"),
        },
        "dataset": {"fuzzy": fuzzy_manifest, "full_dedup": dedup_manifest},
    }
    atomic_json(output / "results.json", result)
    with (output / "run_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "arm", "seed", "steps", "training_tokens", "final_train_val_bpb",
            "final_eval_val_bpb", "minimum_validation_bpb", "final_core", "training_time_sec",
        ])
        writer.writerow([
            "fuzzy", args.seed, EXPECTED_STEPS, EXPECTED_TOKENS, final_bpb,
            final_eval_bpb, minimum_bpb, final_core, metrics.get("training_time_sec"),
        ])

    lines = [
        "# FineWeb-EDU-Fortified fuzzy dedup × NanoChat d24 — Pioneer report",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "> This is a fuzzy-only, single-seed pioneer report. It validates the completed training and evaluation pipeline, but does not estimate the causal effect of fuzzy deduplication. The paired raw comparison will appear in the final report.",
        "",
        "## Executive summary",
        "",
        f"Completed arm/seed: **fuzzy / {args.seed}**.",
        f"Optimization steps and training tokens: **{EXPECTED_STEPS:,} / {EXPECTED_TOKENS:,}**.",
        f"Final in-training validation BPB: **{_fmt(final_bpb)}**; minimum observed BPB: **{_fmt(minimum_bpb)}**.",
        f"Independent final validation BPB: **{_fmt(final_eval_bpb)}**.",
        f"Final full CORE: **{_fmt(final_core, 4)}**.",
        f"Optimization wall time: **{_fmt((metrics.get('training_time_sec') or 0) / 3600, 2)} hours**.",
        "",
        "## Run results",
        "",
        _table(
            ["Arm", "Seed", "Steps", "Training tokens", "Final BPB", "Eval BPB", "Final CORE"],
            [["fuzzy", args.seed, f"{EXPECTED_STEPS:,}", f"{EXPECTED_TOKENS:,}", _fmt(final_bpb), _fmt(final_eval_bpb), _fmt(final_core, 4)]],
        ),
        "",
        "## Quantitative Analysis",
        "",
        "![Validation BPB](val_bpb_vs_tokens.png)",
        "",
        "![Training loss diagnostic](train_loss_vs_tokens.png)",
        "",
        "![Online CORE](online_core_vs_tokens.png)",
        "",
        "![Validation BPB versus optimization time](val_bpb_vs_time.png)",
        "",
        "## LM Eval Harness Results",
        "",
        "![Final CORE](final_core.png)",
        "",
        "![Per-task centered accuracy](core_task_scores.png)",
        "",
        "## Dataset Analysis",
        "",
        _table(
            ["Selected docs", "Selected source tokens", "Excluded validation-overlap docs", "Excluded validation-overlap tokens"],
            [[
                f"{selection.get('docs', 0):,}",
                f"{selection.get('tokens', 0):,}",
                f"{selection.get('excluded_validation_overlap_docs', 0):,}",
                f"{selection.get('excluded_validation_overlap_tokens', 0):,}",
            ]],
        ),
        "",
        f"Full post-fuzzy corpus: **{int(full_fuzzy.get('docs', 0)):,} documents / {int(full_fuzzy.get('nanochat_tokens', 0)):,} NanoChat tokens**.",
        "",
        "### Selected fuzzy data by Common Crawl snapshot",
        "",
        _table(["Subset", "Documents", "Source tokens"], subset_rows),
        "",
        "### Selected fuzzy data by document length",
        "",
        _table(["Token bin", "Documents", "Source tokens"], length_rows),
        "",
        "![Component sizes](component_size_histogram.png)",
        "",
        "![Jaccard ECDF](jaccard_ecdf.png)",
        "",
        "### Removed → keeper examples",
        "",
        _table(["Jaccard", "Component size", "Removed preview", "Keeper preview"], qualitative_rows),
        "",
        "## Reproducibility",
        "",
        f"Validation checksum: {record['config']['validation_sha256']}.",
        f"Fuzzy view manifest checksum: {sha256_file(args.fuzzy_manifest)}.",
        f"Run summary checksum: {sha256_file(summary_path)}.",
        f"Git commit recorded by run: {record['config'].get('git_commit', 'unknown')}.",
        "",
        "## Interpretation",
        "",
        "- Validation BPB, training loss, online CORE, wall time, and final full CORE are reported with the same definitions used by the final paired report.",
        "- Training loss is diagnostic only.",
        "- This report contains one model seed and no raw NanoChat arm; it must not be used to claim that fuzzy dedup improves or harms model quality.",
        "- The final report will add paired BPB deltas, token-efficiency targets, raw removed-token exposure, cross-seed mean ± SD, and per-task paired CORE deltas.",
    ]
    report_text = "\n".join(lines) + "\n"
    (output / "pioneer_report.md").write_text(report_text, encoding="utf-8")
    (output / "final_report.md").write_text(report_text, encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fuzzy-manifest", type=Path, required=True)
    parser.add_argument("--dedup-manifest", type=Path, required=True)
    parser.add_argument("--component-audit", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    generate_pioneer_report(parser().parse_args())


if __name__ == "__main__":
    main()
