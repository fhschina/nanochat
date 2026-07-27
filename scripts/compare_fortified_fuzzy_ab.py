"""Record, summarize, and report the Fortified fuzzy-dedup NanoChat A/B."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

TOTAL_BATCH_SIZE = 1_048_576
EXPECTED_STEPS = 6_612
EXPECTED_TOKENS = TOTAL_BATCH_SIZE * EXPECTED_STEPS
ARMS = ("fuzzy", "raw")
SEEDS = (42, 43, 44)
COLORS = {"fuzzy": "#1f77b4", "raw": "#d62728"}


def _save_figure(fig, output: Path) -> None:
    """Save the required SVG plus a broadly-renderable PNG companion."""
    fig.savefig(output)
    if output.suffix.lower() == ".svg":
        fig.savefig(output.with_suffix(".png"), dpi=160)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def record_config(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.train_manifest.expanduser().resolve()
    validation = args.validation_file.expanduser().resolve()
    payload = {
        "created_at": utc_now(),
        "arm": args.arm,
        "seed": args.seed,
        "kind": args.kind,
        "model_tag": args.model_tag,
        "git_commit": git_commit(),
        "train_manifest": str(manifest),
        "train_manifest_sha256": sha256_file(manifest),
        "validation_file": str(validation),
        "validation_sha256": sha256_file(validation),
        "training": {
            "depth": args.depth,
            "num_iterations": args.num_iterations,
            "total_batch_size": args.total_batch_size,
            "max_seq_len": args.max_seq_len,
            "device_batch_size": args.device_batch_size,
            "eval_every": args.eval_every,
            "eval_tokens": args.eval_tokens,
            "core_metric_every": args.core_metric_every,
            "core_metric_max_per_task": args.core_metric_max_per_task,
            "core_eval_seed": args.core_eval_seed,
            "fp8": args.fp8,
            "window_pattern": args.window_pattern,
        },
    }
    atomic_json(run_dir / "run_config.json", payload)


def parse_core_csv(path: Path) -> tuple[float | None, dict[str, dict[str, float]]]:
    if not path.is_file():
        return None, {}
    tasks: dict[str, dict[str, float]] = {}
    core = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            task = row[0].strip()
            if task == "CORE":
                core = float(row[2]) if len(row) > 2 and row[2].strip() else None
            else:
                tasks[task] = {
                    "accuracy": float(row[1]) if len(row) > 1 and row[1].strip() else float("nan"),
                    "centered": float(row[2]) if len(row) > 2 and row[2].strip() else float("nan"),
                }
    return core, tasks


def summarize_run(run_dir: Path) -> dict[str, Any]:
    config = load_json(run_dir / "run_config.json")
    train_log = (run_dir / "train.log").read_text(errors="replace")
    eval_log = (run_dir / "base_eval.log").read_text(errors="replace") if (run_dir / "base_eval.log").is_file() else ""
    val_curve = [
        {"step": int(step), "tokens": int(step) * TOTAL_BATCH_SIZE, "bpb": float(value)}
        for step, value in re.findall(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([0-9.]+)", train_log)
    ]
    core_curve = [
        {"step": int(step), "tokens": int(step) * TOTAL_BATCH_SIZE, "core": float(value)}
        for step, value in re.findall(r"Step\s+(\d+)\s+\|\s+CORE metric:\s+([+-]?[0-9.eE]+)", train_log)
    ]
    train_curve = [
        {
            "step": int(step),
            "tokens": (int(step) + 1) * TOTAL_BATCH_SIZE,
            "loss": float(loss),
            "runtime_sec": float(minutes) * 60,
        }
        for step, loss, minutes in re.findall(
            r"step\s+(\d+)/\d+.*?\| loss:\s+([0-9.eE+-]+).*?\| total time:\s+([0-9.]+)m",
            train_log,
        )
    ]
    final_exposure = None
    exposure_matches = re.findall(r"FINAL_DATA_EXPOSURE\s+(\{[^\n]+\})", train_log)
    if exposure_matches:
        final_exposure = json.loads(exposure_matches[-1])
    loader_stats = None
    loader_matches = re.findall(r"FINAL_DATA_LOADER_STATS\s+(\{[^\n]+\})", train_log)
    if loader_matches:
        loader_stats = json.loads(loader_matches[-1])
    iterations = re.findall(r"Using user-provided number of iterations:\s+([0-9,]+)", train_log)
    total_tokens = re.findall(r"Total number of training tokens:\s+([0-9,]+)", train_log)
    times = re.findall(r"Total training time:\s+([0-9.]+)m", train_log)
    peak = re.findall(r"Peak memory usage:\s+([0-9.]+)MiB", train_log)
    eval_bpb = re.findall(r"val bpb:\s+([0-9.]+)", eval_log)
    core, tasks = parse_core_csv(run_dir / "base_eval_core.csv")
    if core is None:
        values = re.findall(r"CORE metric:\s+([+-]?[0-9.eE]+)", eval_log)
        core = float(values[-1]) if values else (core_curve[-1]["core"] if core_curve else None)
    summary = {
        "completed_at": utc_now(),
        "arm": config["arm"],
        "seed": config["seed"],
        "kind": config["kind"],
        "run_dir": str(run_dir),
        "config": config,
        "curves": {"val_bpb": val_curve, "train_loss": train_curve, "online_core": core_curve},
        "metrics": {
            "num_iterations": int(iterations[-1].replace(",", "")) if iterations else None,
            "training_tokens": int(total_tokens[-1].replace(",", "")) if total_tokens else None,
            "final_train_val_bpb": val_curve[-1]["bpb"] if val_curve else None,
            "final_eval_val_bpb": float(eval_bpb[-1]) if eval_bpb else None,
            "final_core": core,
            "training_time_sec": float(times[-1]) * 60 if times else None,
            "peak_memory_mib": float(peak[-1]) if peak else None,
            "final_data_exposure": final_exposure,
            "final_loader_stats": loader_stats,
        },
        "core_tasks": tasks,
    }
    if config["kind"] == "full":
        if summary["metrics"]["num_iterations"] != EXPECTED_STEPS:
            raise RuntimeError(f"{run_dir}: expected {EXPECTED_STEPS} steps")
        if summary["metrics"]["training_tokens"] != EXPECTED_TOKENS:
            raise RuntimeError(f"{run_dir}: expected {EXPECTED_TOKENS} training tokens")
        if not val_curve or val_curve[0]["step"] != 0 or val_curve[-1]["step"] != EXPECTED_STEPS:
            raise RuntimeError(f"{run_dir}: incomplete validation curve")
    atomic_json(run_dir / "run_summary.json", summary)
    return summary


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _fmt(value: float | None, digits: int = 6) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _curves_by_arm(records, name):
    return {
        arm: {int(row["seed"]): row["curves"][name] for row in records if row["arm"] == arm}
        for arm in ARMS
    }


def _plot_metric(records, curve_name, value_key, ylabel, output, secondary_step=True):
    curves = _curves_by_arm(records, curve_name)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for arm in ARMS:
        arm_curves = curves[arm]
        for seed, curve in sorted(arm_curves.items()):
            if not curve:
                continue
            ax.plot([x["tokens"] / 1e9 for x in curve], [x[value_key] for x in curve], color=COLORS[arm], alpha=0.22, linewidth=1)
        if arm_curves:
            common = sorted(set.intersection(*(set(x["step"] for x in curve) for curve in arm_curves.values())))
            matrix = np.asarray([[next(x[value_key] for x in curve if x["step"] == step) for step in common] for curve in arm_curves.values()])
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean)
            tokens = np.asarray(common) * TOTAL_BATCH_SIZE / 1e9
            ax.plot(tokens, mean, color=COLORS[arm], label=arm, linewidth=2.4)
            ax.fill_between(tokens, mean - sd, mean + sd, color=COLORS[arm], alpha=0.13)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    if secondary_step:
        sec = ax.secondary_xaxis("top", functions=(lambda x: x * 1e9 / TOTAL_BATCH_SIZE, lambda x: x * TOTAL_BATCH_SIZE / 1e9))
        sec.set_xlabel("Optimization step")
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)


def first_crossing(curve: list[dict], target: float) -> float | None:
    best = float("inf")
    previous = None
    for point in curve:
        current = min(best, float(point["bpb"]))
        if current <= target:
            if previous is None or previous[1] <= target or previous[1] == current:
                return float(point["tokens"])
            fraction = (previous[1] - target) / (previous[1] - current)
            return previous[0] + fraction * (float(point["tokens"]) - previous[0])
        best = current
        previous = (float(point["tokens"]), current)
    return None


def plot_delta(pairs, output):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    all_deltas = []
    for seed, fuzzy, raw in pairs:
        f = {row["step"]: row["bpb"] for row in fuzzy["curves"]["val_bpb"]}
        r = {row["step"]: row["bpb"] for row in raw["curves"]["val_bpb"]}
        steps = sorted(set(f) & set(r))
        deltas = np.asarray([f[step] - r[step] for step in steps])
        all_deltas.append((steps, deltas))
        ax.plot(np.asarray(steps) * TOTAL_BATCH_SIZE / 1e9, deltas, alpha=0.25, color="#9467bd")
    if all_deltas:
        common = sorted(set.intersection(*(set(steps) for steps, _ in all_deltas)))
        matrix = np.asarray([[dict(zip(steps, values, strict=True))[step] for step in common] for steps, values in all_deltas])
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean)
        x = np.asarray(common) * TOTAL_BATCH_SIZE / 1e9
        ax.plot(x, mean, color="#9467bd", linewidth=2.4)
        ax.fill_between(x, mean - sd, mean + sd, color="#9467bd", alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Training tokens (billions)", ylabel="Validation BPB delta (fuzzy - raw)")
    ax.grid(alpha=0.25)
    fig.tight_layout(); _save_figure(fig, output); plt.close(fig)



def _validation_with_runtime(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validation step/token/BPB arrays with interpolated optimization time."""
    curve = record["curves"]["val_bpb"]
    steps = np.asarray([float(point["step"]) for point in curve])
    tokens = np.asarray([float(point["tokens"]) for point in curve])
    bpb = np.asarray([float(point["bpb"]) for point in curve])
    train_curve = record["curves"].get("train_loss", [])
    runtime_steps = np.asarray([float(point["step"]) for point in train_curve])
    runtime_seconds = np.asarray([float(point["runtime_sec"]) for point in train_curve])
    final_seconds = record.get("metrics", {}).get("training_time_sec")
    if runtime_steps.size == 0:
        runtime_steps = np.asarray([0.0, float(steps[-1])])
        runtime_seconds = np.asarray([0.0, float(final_seconds or 0.0)])
    elif final_seconds is not None and steps[-1] > runtime_steps[-1]:
        runtime_steps = np.append(runtime_steps, steps[-1])
        runtime_seconds = np.append(runtime_seconds, float(final_seconds))
    runtime_hours = np.interp(steps, runtime_steps, runtime_seconds) / 3600.0
    return steps, tokens, bpb, runtime_hours


def plot_interactive_validation(pairs, output: Path) -> None:
    """Write a self-contained Plotly dashboard for validation BPB and paired deltas."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Validation BPB vs training tokens",
            "Paired BPB delta vs training tokens",
            "Validation BPB vs optimization time",
            "Paired BPB delta vs optimization time",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )
    trace_kinds: list[str] = []

    def add(trace, row: int, col: int, kind: str) -> None:
        fig.add_trace(trace, row=row, col=col)
        trace_kinds.append(kind)

    arm_rgba = {"fuzzy": "rgba(31,119,180,0.14)", "raw": "rgba(214,39,40,0.14)"}
    delta_color = "#9467bd"
    delta_rgba = "rgba(148,103,189,0.15)"
    seed_symbols = {42: "circle", 43: "triangle-up", 44: "square"}
    records = [record for _, fuzzy, raw in pairs for record in (fuzzy, raw)]

    # Individual validation curves against tokens and wall time.
    for record in records:
        arm, seed = record["arm"], int(record["seed"])
        steps, tokens, bpb, hours = _validation_with_runtime(record)
        token_custom = np.column_stack([steps, hours])
        time_custom = np.column_stack([steps, tokens / 1e9])
        add(
            go.Scatter(
                x=tokens / 1e9,
                y=bpb,
                mode="lines+markers",
                name=f"{arm} seed {seed}",
                legendgroup=f"{arm}-seed-{seed}",
                line=dict(color=COLORS[arm], width=1.2),
                marker=dict(color=COLORS[arm], size=4, symbol=seed_symbols.get(seed, "circle")),
                opacity=0.35,
                customdata=token_custom,
                hovertemplate=(
                    f"arm={arm}<br>seed={seed}<br>step=%{{customdata[0]:,.0f}}"
                    "<br>training tokens=%{x:.3f}B<br>wall time=%{customdata[1]:.3f}h"
                    "<br>validation BPB=%{y:.6f}<extra></extra>"
                ),
                showlegend=True,
            ),
            1,
            1,
            "seed",
        )
        add(
            go.Scatter(
                x=hours,
                y=bpb,
                mode="lines+markers",
                name=f"{arm} seed {seed}",
                legendgroup=f"{arm}-seed-{seed}",
                line=dict(color=COLORS[arm], width=1.2),
                marker=dict(color=COLORS[arm], size=4, symbol=seed_symbols.get(seed, "circle")),
                opacity=0.35,
                customdata=time_custom,
                hovertemplate=(
                    f"arm={arm}<br>seed={seed}<br>wall time=%{{x:.3f}}h"
                    "<br>step=%{customdata[0]:,.0f}<br>training tokens=%{customdata[1]:.3f}B"
                    "<br>validation BPB=%{y:.6f}<extra></extra>"
                ),
                showlegend=False,
            ),
            2,
            1,
            "seed",
        )

    # Cross-seed validation means and ±1 SD bands.
    for arm in ARMS:
        arm_records = [record for record in records if record["arm"] == arm]
        step_maps = []
        for record in arm_records:
            steps, tokens, bpb, hours = _validation_with_runtime(record)
            step_maps.append({int(step): (token, value, hour) for step, token, value, hour in zip(steps, tokens, bpb, hours, strict=True)})
        common_steps = sorted(set.intersection(*(set(mapping) for mapping in step_maps)))
        matrix = np.asarray([[mapping[step][1] for step in common_steps] for mapping in step_maps])
        hour_matrix = np.asarray([[mapping[step][2] for step in common_steps] for mapping in step_maps])
        token_x = np.asarray([step_maps[0][step][0] for step in common_steps]) / 1e9
        mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1])
        mean_hours = hour_matrix.mean(axis=0)
        add(go.Scatter(x=token_x, y=mean + sd, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False), 1, 1, "mean")
        add(go.Scatter(x=token_x, y=mean - sd, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=arm_rgba[arm], hoverinfo="skip", showlegend=False), 1, 1, "mean")
        add(
            go.Scatter(
                x=token_x,
                y=mean,
                mode="lines",
                name=f"{arm} mean ±1 SD",
                legendgroup=f"{arm}-mean",
                line=dict(color=COLORS[arm], width=3),
                customdata=np.column_stack([common_steps, mean_hours]),
                hovertemplate=(
                    f"arm={arm}<br>cross-seed mean<br>step=%{{customdata[0]:,.0f}}"
                    "<br>training tokens=%{x:.3f}B<br>mean wall time=%{customdata[1]:.3f}h"
                    "<br>mean validation BPB=%{y:.6f}<extra></extra>"
                ),
                showlegend=True,
            ),
            1,
            1,
            "mean",
        )

        time_series = [_validation_with_runtime(record) for record in arm_records]
        max_common_hour = min(series[3][-1] for series in time_series)
        time_grid = np.linspace(0.0, max_common_hour, 180)
        time_matrix = np.asarray([np.interp(time_grid, series[3], series[2]) for series in time_series])
        step_matrix = np.asarray([np.interp(time_grid, series[3], series[0]) for series in time_series])
        time_mean = time_matrix.mean(axis=0)
        time_sd = time_matrix.std(axis=0, ddof=1) if len(time_series) > 1 else np.zeros_like(time_mean)
        mean_steps = step_matrix.mean(axis=0)
        add(go.Scatter(x=time_grid, y=time_mean + time_sd, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False), 2, 1, "mean")
        add(go.Scatter(x=time_grid, y=time_mean - time_sd, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=arm_rgba[arm], hoverinfo="skip", showlegend=False), 2, 1, "mean")
        add(
            go.Scatter(
                x=time_grid,
                y=time_mean,
                mode="lines",
                name=f"{arm} mean ±1 SD",
                legendgroup=f"{arm}-mean",
                line=dict(color=COLORS[arm], width=3),
                customdata=np.column_stack([mean_steps, mean_steps * TOTAL_BATCH_SIZE / 1e9]),
                hovertemplate=(
                    f"arm={arm}<br>cross-seed mean<br>wall time=%{{x:.3f}}h"
                    "<br>approx. step=%{customdata[0]:,.0f}<br>approx. tokens=%{customdata[1]:.3f}B"
                    "<br>mean validation BPB=%{y:.6f}<extra></extra>"
                ),
                showlegend=False,
            ),
            2,
            1,
            "mean",
        )

    # Paired deltas at matched token budgets.
    token_delta_rows = []
    time_pair_series = []
    for seed, fuzzy, raw in pairs:
        f_steps, f_tokens, f_bpb, f_hours = _validation_with_runtime(fuzzy)
        r_steps, r_tokens, r_bpb, r_hours = _validation_with_runtime(raw)
        f_map = {int(step): (token, value, hour) for step, token, value, hour in zip(f_steps, f_tokens, f_bpb, f_hours, strict=True)}
        r_map = {int(step): (token, value, hour) for step, token, value, hour in zip(r_steps, r_tokens, r_bpb, r_hours, strict=True)}
        steps = sorted(set(f_map) & set(r_map))
        token_x = np.asarray([f_map[step][0] for step in steps]) / 1e9
        delta = np.asarray([f_map[step][1] - r_map[step][1] for step in steps])
        token_delta_rows.append((steps, token_x, delta, f_map, r_map))
        add(
            go.Scatter(
                x=token_x,
                y=delta,
                mode="lines+markers",
                name=f"paired seed {seed}",
                legendgroup=f"paired-seed-{seed}",
                line=dict(color=delta_color, width=1.2),
                marker=dict(color=delta_color, size=4, symbol=seed_symbols.get(seed, "circle")),
                opacity=0.38,
                customdata=np.column_stack([
                    steps,
                    [f_map[step][2] for step in steps],
                    [r_map[step][2] for step in steps],
                ]),
                hovertemplate=(
                    f"paired seed={seed}<br>step=%{{customdata[0]:,.0f}}<br>training tokens=%{{x:.3f}}B"
                    "<br>fuzzy time=%{customdata[1]:.3f}h<br>raw time=%{customdata[2]:.3f}h"
                    "<br>BPB delta (fuzzy − raw)=%{y:+.6f}<extra></extra>"
                ),
                showlegend=False,
            ),
            1,
            2,
            "seed",
        )

        max_common_hour = min(f_hours[-1], r_hours[-1])
        time_grid = np.linspace(0.0, max_common_hour, 180)
        time_delta = np.interp(time_grid, f_hours, f_bpb) - np.interp(time_grid, r_hours, r_bpb)
        fuzzy_step = np.interp(time_grid, f_hours, f_steps)
        raw_step = np.interp(time_grid, r_hours, r_steps)
        time_pair_series.append((max_common_hour, f_hours, f_bpb, f_steps, r_hours, r_bpb, r_steps))
        add(
            go.Scatter(
                x=time_grid,
                y=time_delta,
                mode="lines",
                name=f"paired seed {seed}",
                legendgroup=f"paired-seed-{seed}",
                line=dict(color=delta_color, width=1.2),
                opacity=0.38,
                customdata=np.column_stack([fuzzy_step, raw_step]),
                hovertemplate=(
                    f"paired seed={seed}<br>wall time=%{{x:.3f}}h"
                    "<br>fuzzy approx. step=%{customdata[0]:,.0f}<br>raw approx. step=%{customdata[1]:,.0f}"
                    "<br>BPB delta (fuzzy − raw)=%{y:+.6f}<extra></extra>"
                ),
                showlegend=False,
            ),
            2,
            2,
            "seed",
        )

    common_steps = sorted(set.intersection(*(set(row[0]) for row in token_delta_rows)))
    token_matrix = np.asarray([[dict(zip(row[0], row[2], strict=True))[step] for step in common_steps] for row in token_delta_rows])
    token_mean = token_matrix.mean(axis=0)
    token_sd = token_matrix.std(axis=0, ddof=1) if len(token_delta_rows) > 1 else np.zeros_like(token_mean)
    token_x = np.asarray(common_steps) * TOTAL_BATCH_SIZE / 1e9
    add(go.Scatter(x=token_x, y=token_mean + token_sd, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False), 1, 2, "mean")
    add(go.Scatter(x=token_x, y=token_mean - token_sd, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=delta_rgba, hoverinfo="skip", showlegend=False), 1, 2, "mean")
    add(
        go.Scatter(
            x=token_x,
            y=token_mean,
            mode="lines",
            name="paired mean ±1 SD",
            legendgroup="paired-mean",
            line=dict(color=delta_color, width=3),
            customdata=np.asarray(common_steps),
            hovertemplate=(
                "cross-seed paired mean<br>step=%{customdata:,.0f}<br>training tokens=%{x:.3f}B"
                "<br>mean BPB delta (fuzzy − raw)=%{y:+.6f}<extra></extra>"
            ),
            showlegend=True,
        ),
        1,
        2,
        "mean",
    )

    max_common_hour = min(series[0] for series in time_pair_series)
    time_grid = np.linspace(0.0, max_common_hour, 180)
    time_matrix = np.asarray([
        np.interp(time_grid, f_hours, f_bpb) - np.interp(time_grid, r_hours, r_bpb)
        for _, f_hours, f_bpb, _, r_hours, r_bpb, _ in time_pair_series
    ])
    time_mean = time_matrix.mean(axis=0)
    time_sd = time_matrix.std(axis=0, ddof=1) if len(time_pair_series) > 1 else np.zeros_like(time_mean)
    fuzzy_steps = np.asarray([np.interp(time_grid, row[1], row[3]) for row in time_pair_series]).mean(axis=0)
    raw_steps = np.asarray([np.interp(time_grid, row[4], row[6]) for row in time_pair_series]).mean(axis=0)
    add(go.Scatter(x=time_grid, y=time_mean + time_sd, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False), 2, 2, "mean")
    add(go.Scatter(x=time_grid, y=time_mean - time_sd, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=delta_rgba, hoverinfo="skip", showlegend=False), 2, 2, "mean")
    add(
        go.Scatter(
            x=time_grid,
            y=time_mean,
            mode="lines",
            name="paired mean ±1 SD",
            legendgroup="paired-mean",
            line=dict(color=delta_color, width=3),
            customdata=np.column_stack([fuzzy_steps, raw_steps]),
            hovertemplate=(
                "cross-seed paired mean<br>wall time=%{x:.3f}h"
                "<br>fuzzy approx. step=%{customdata[0]:,.0f}<br>raw approx. step=%{customdata[1]:,.0f}"
                "<br>mean BPB delta (fuzzy − raw)=%{y:+.6f}<extra></extra>"
            ),
            showlegend=False,
        ),
        2,
        2,
        "mean",
    )

    fig.add_hline(y=0, line_color="#444", line_width=1, row=1, col=2)
    fig.add_hline(y=0, line_color="#444", line_width=1, row=2, col=2)
    fig.update_xaxes(title_text="Training tokens (billions)", row=1, col=1)
    fig.update_xaxes(title_text="Training tokens (billions)", row=1, col=2)
    fig.update_xaxes(title_text="Optimization time (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Optimization time (hours)", row=2, col=2)
    fig.update_yaxes(title_text="Validation BPB", range=[0.74, 1.08], row=1, col=1)
    fig.update_yaxes(title_text="Fuzzy − raw BPB", row=1, col=2)
    fig.update_yaxes(title_text="Validation BPB", range=[0.74, 1.08], row=2, col=1)
    fig.update_yaxes(title_text="Fuzzy − raw BPB", row=2, col=2)

    all_mask = [True] * len(trace_kinds)
    mean_mask = [kind == "mean" for kind in trace_kinds]
    seed_mask = [kind == "seed" for kind in trace_kinds]
    fig.update_layout(
        title=None,
        template="plotly_white",
        autosize=True,
        height=900,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.17,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=80, r=40, t=120, b=150),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                xanchor="center",
                y=1.08,
                yanchor="bottom",
                font=dict(size=12),
                buttons=[
                    dict(label="Means + seeds", method="update", args=[{"visible": all_mask}]),
                    dict(label="Means ±1 SD", method="update", args=[{"visible": mean_mask}]),
                    dict(label="Individual seeds", method="update", args=[{"visible": seed_mask}]),
                ],
            )
        ],
    )
    fig.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        div_id="fortified-validation-dashboard",
        default_width="100%",
        default_height="900px",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )
    header = """
<header class="dashboard-header">
  <h1>FineWeb-EDU-Fortified Fuzzy Dedup × NanoChat d24</h1>
  <p>Interactive validation analysis · ΔBPB = fuzzy − raw · Negative favors fuzzy; positive favors raw</p>
</header>
"""
    styles = """
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FineWeb-EDU-Fortified Fuzzy Dedup × NanoChat d24</title>
<style>
  html, body { margin: 0; padding: 0; }
  .dashboard-header {
    box-sizing: border-box;
    width: 100%;
    padding: 24px 24px 4px;
    text-align: center;
    color: #2a3f5f;
    font-family: Arial, Helvetica, sans-serif;
  }
  .dashboard-header h1 {
    margin: 0;
    font-size: clamp(22px, 2.2vw, 34px);
    font-weight: 500;
    line-height: 1.2;
  }
  .dashboard-header p {
    margin: 6px 0 0;
    font-size: clamp(14px, 1.2vw, 18px);
    font-weight: 400;
    line-height: 1.35;
  }
  @media (max-width: 720px) {
    .dashboard-header { padding: 16px 12px 0; }
  }
</style>
"""
    page = output.read_text(encoding="utf-8")
    if page.count("</head>") != 1 or page.count("<body>") != 1:
        raise RuntimeError("Unexpected Plotly HTML shell")
    page = page.replace("</head>", f"{styles}</head>", 1)
    page = page.replace("<body>", f"<body>{header}", 1)
    output.write_text(page, encoding="utf-8")


def plot_bpb_time(records, output):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for row in records:
        runtime = {point["step"]: point["runtime_sec"] for point in row["curves"]["train_loss"]}
        if not runtime:
            continue
        steps = sorted(runtime)
        xs, ys = [], []
        for point in row["curves"]["val_bpb"]:
            prior = [step for step in steps if step <= point["step"]]
            elapsed = runtime[prior[-1]] if prior else 0.0
            xs.append(elapsed / 3600); ys.append(point["bpb"])
        ax.plot(xs, ys, color=COLORS[row["arm"]], alpha=0.3, label=row["arm"] if int(row["seed"]) == 42 else None)
    ax.set(xlabel="Optimization time (hours)", ylabel="Validation BPB")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); _save_figure(fig, output); plt.close(fig)

def plot_final_core(records, output):
    values = [[row["metrics"]["final_core"] for row in records if row["arm"] == arm and row["metrics"]["final_core"] is not None] for arm in ARMS]
    means = [np.mean(row) if row else np.nan for row in values]
    sds = [np.std(row, ddof=1) if len(row) > 1 else 0 for row in values]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.bar(ARMS, means, yerr=sds, color=[COLORS[arm] for arm in ARMS], alpha=0.8, capsize=5)
    for index, row in enumerate(values):
        ax.scatter([index] * len(row), row, color="black", s=18, zorder=3)
    ax.set_ylabel("Final full CORE"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); _save_figure(fig, output); plt.close(fig)


def plot_task_delta(pairs, output):
    deltas: dict[str, list[tuple[int, float]]] = {}
    for seed, fuzzy, raw in pairs:
        for task in set(fuzzy["core_tasks"]) & set(raw["core_tasks"]):
            delta = fuzzy["core_tasks"][task]["centered"] - raw["core_tasks"][task]["centered"]
            deltas.setdefault(task, []).append((int(seed), float(delta)))
    if not deltas:
        return

    rows = []
    for task, seed_values in deltas.items():
        values = np.asarray([value for _, value in seed_values], dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        consistent = bool(np.all(values > 0) or np.all(values < 0))
        rows.append((task, seed_values, mean, sd, consistent))
    rows.sort(key=lambda row: row[2])

    fig, ax = plt.subplots(figsize=(10.5, max(6.5, len(rows) * 0.38)))
    y = np.arange(len(rows), dtype=float)
    means = np.asarray([row[2] for row in rows])
    sds = np.asarray([row[3] for row in rows])
    colors = np.asarray(["#2ca02c" if value >= 0 else "#d62728" for value in means])
    for position, mean, sd, color in zip(y, means, sds, colors, strict=True):
        ax.hlines(position, 0, mean, color=color, alpha=0.55, linewidth=1.5, zorder=1)
        ax.errorbar(
            mean,
            position,
            xerr=sd,
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            alpha=0.9,
            zorder=2,
        )

    seed_order = sorted({seed for row in rows for seed, _ in row[1]})
    seed_offsets = np.linspace(-0.14, 0.14, max(len(seed_order), 1))
    seed_markers = ["o", "^", "s"]
    seed_colors = ["#1f77b4", "#9467bd", "#ff7f0e"]
    for index, seed in enumerate(seed_order):
        xs, ys = [], []
        for position, (_, seed_values, _, _, _) in zip(y, rows, strict=True):
            value_by_seed = dict(seed_values)
            if seed in value_by_seed:
                xs.append(value_by_seed[seed])
                ys.append(position + seed_offsets[index])
        ax.scatter(
            xs,
            ys,
            s=38,
            marker=seed_markers[index % len(seed_markers)],
            color=seed_colors[index % len(seed_colors)],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.82,
            label=f"seed {seed} paired Δ",
            zorder=3,
        )

    for position, (_, _, mean, _, consistent), color in zip(y, rows, colors, strict=True):
        ax.scatter(
            mean,
            position,
            s=58 if consistent else 48,
            marker="D" if consistent else "o",
            facecolor=color if consistent else "white",
            edgecolor=color,
            linewidth=1.5,
            zorder=4,
        )
    ax.scatter([], [], s=58, marker="D", color="#555555", label="paired mean; all 3 seeds agree")
    ax.scatter([], [], s=48, marker="o", facecolor="white", edgecolor="#555555", label="paired mean; mixed direction")

    ax.set_yticks(y, [row[0] for row in rows])
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Centered-accuracy delta (fuzzy − raw); left favors raw, right favors fuzzy")
    ax.set_title(
        "Final full CORE per-task paired deltas\n"
        "Each benchmark: seeds 42/43/44 + paired mean ±1 SD"
    )
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    _save_figure(fig, output)
    plt.close(fig)


def plot_interactive_core(pairs, output: Path) -> None:
    """Write an interactive final-CORE and per-task paired-delta dashboard."""
    task_sets = [
        set(fuzzy.get("core_tasks", {})) & set(raw.get("core_tasks", {}))
        for _, fuzzy, raw in pairs
    ]
    if not task_sets:
        return
    tasks = sorted(set.intersection(*task_sets))
    if not tasks:
        return

    rows = []
    for task in tasks:
        seed_values = []
        for seed, fuzzy, raw in pairs:
            fuzzy_value = float(fuzzy["core_tasks"][task]["centered"])
            raw_value = float(raw["core_tasks"][task]["centered"])
            seed_values.append((int(seed), fuzzy_value, raw_value, fuzzy_value - raw_value))
        values = np.asarray([row[3] for row in seed_values], dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        consistent = bool(np.all(values > 0) or np.all(values < 0))
        rows.append((task, seed_values, mean, sd, consistent))
    rows.sort(key=lambda row: row[2])

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.27, 0.73],
        vertical_spacing=0.14,
        subplot_titles=(
            "Final full CORE by paired seed",
            "Per-task centered-accuracy paired deltas",
        ),
    )
    trace_kinds: list[str] = []

    def add(trace, row: int, kind: str) -> None:
        fig.add_trace(trace, row=row, col=1)
        trace_kinds.append(kind)

    seed_colors = {42: "#1f77b4", 43: "#9467bd", 44: "#ff7f0e"}
    seed_symbols = {42: "circle", 43: "triangle-up", 44: "square"}

    # Final full CORE: preserve the pairing by connecting fuzzy/raw for each seed.
    for seed, fuzzy, raw in pairs:
        fuzzy_core = float(fuzzy["metrics"]["final_core"])
        raw_core = float(raw["metrics"]["final_core"])
        paired_delta = fuzzy_core - raw_core
        add(
            go.Scatter(
                x=["Fuzzy", "Raw"],
                y=[fuzzy_core, raw_core],
                mode="lines+markers",
                name=f"seed {seed}",
                legendgroup=f"core-seed-{seed}",
                line=dict(color=seed_colors.get(int(seed), "#777"), width=1.5),
                marker=dict(
                    color=seed_colors.get(int(seed), "#777"),
                    size=8,
                    symbol=seed_symbols.get(int(seed), "circle"),
                ),
                opacity=0.58,
                customdata=np.asarray([
                    [int(seed), paired_delta],
                    [int(seed), paired_delta],
                ]),
                hovertemplate=(
                    "arm=%{x}<br>seed=%{customdata[0]:.0f}<br>final full CORE=%{y:.6f}"
                    "<br>paired CORE delta (fuzzy − raw)=%{customdata[1]:+.6f}<extra></extra>"
                ),
            ),
            1,
            "seed",
        )

    arm_values = {
        "Fuzzy": np.asarray([float(fuzzy["metrics"]["final_core"]) for _, fuzzy, _ in pairs]),
        "Raw": np.asarray([float(raw["metrics"]["final_core"]) for _, _, raw in pairs]),
    }
    arm_means = np.asarray([arm_values["Fuzzy"].mean(), arm_values["Raw"].mean()])
    arm_sds = np.asarray([
        arm_values["Fuzzy"].std(ddof=1) if len(arm_values["Fuzzy"]) > 1 else 0.0,
        arm_values["Raw"].std(ddof=1) if len(arm_values["Raw"]) > 1 else 0.0,
    ])
    add(
        go.Scatter(
            x=["Fuzzy", "Raw"],
            y=arm_means,
            mode="lines+markers",
            name="cross-seed mean ±1 SD",
            legendgroup="core-mean",
            line=dict(color="#333", width=3),
            marker=dict(color=[COLORS["fuzzy"], COLORS["raw"]], size=12, symbol="diamond"),
            error_y=dict(type="data", array=arm_sds, visible=True, thickness=1.6, width=5),
            customdata=np.column_stack([arm_sds, np.repeat(len(pairs), 2)]),
            hovertemplate=(
                "arm=%{x}<br>cross-seed mean CORE=%{y:.6f}"
                "<br>sample SD=%{customdata[0]:.6f}<br>n=%{customdata[1]:.0f}<extra></extra>"
            ),
        ),
        1,
        "mean",
    )

    # Per-task paired deltas: one point per seed plus paired mean ±1 sample SD.
    y = np.arange(len(rows), dtype=float)
    seed_order = sorted({seed for _, seed_values, _, _, _ in rows for seed, *_ in seed_values})
    seed_offsets = dict(zip(seed_order, np.linspace(-0.16, 0.16, max(len(seed_order), 1)), strict=True))
    for seed in seed_order:
        xs, ys, custom = [], [], []
        for position, (task, seed_values, _, _, _) in zip(y, rows, strict=True):
            value = next((value for value in seed_values if value[0] == seed), None)
            if value is None:
                continue
            _, fuzzy_value, raw_value, delta = value
            xs.append(delta)
            ys.append(position + seed_offsets[seed])
            custom.append([task, fuzzy_value, raw_value])
        add(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=f"seed {seed} paired Δ",
                legendgroup=f"core-seed-{seed}",
                marker=dict(
                    color=seed_colors.get(seed, "#777"),
                    size=9,
                    symbol=seed_symbols.get(seed, "circle"),
                    line=dict(color="white", width=0.7),
                ),
                opacity=0.78,
                customdata=np.asarray(custom, dtype=object),
                hovertemplate=(
                    f"task=%{{customdata[0]}}<br>paired seed={seed}"
                    "<br>fuzzy centered accuracy=%{customdata[1]:.6f}"
                    "<br>raw centered accuracy=%{customdata[2]:.6f}"
                    "<br>delta (fuzzy − raw)=%{x:+.6f}<extra></extra>"
                ),
                showlegend=False,
            ),
            2,
            "seed",
        )

    line_x, line_y = [], []
    for position, (_, _, mean, _, _) in zip(y, rows, strict=True):
        line_x.extend([0.0, mean, None])
        line_y.extend([position, position, None])
    add(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color="rgba(70,70,70,0.48)", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ),
        2,
        "mean",
    )

    means = np.asarray([row[2] for row in rows])
    sds = np.asarray([row[3] for row in rows])
    mean_colors = ["#2ca02c" if value >= 0 else "#d62728" for value in means]
    mean_symbols = ["diamond" if row[4] else "circle-open" for row in rows]
    mean_custom = np.asarray([
        [task, sd, "all seeds agree" if agrees else "mixed directions", len(seed_values)]
        for task, seed_values, _, sd, agrees in rows
    ], dtype=object)
    add(
        go.Scatter(
            x=means,
            y=y,
            mode="markers",
            name="paired mean ±1 SD",
            legendgroup="task-mean",
            marker=dict(
                color=mean_colors,
                size=12,
                symbol=mean_symbols,
                line=dict(width=1.6),
            ),
            error_x=dict(type="data", array=sds, visible=True, color="#555", thickness=1.3, width=4),
            customdata=mean_custom,
            hovertemplate=(
                "task=%{customdata[0]}<br>mean paired delta=%{x:+.6f}"
                "<br>sample SD=%{customdata[1]:.6f}<br>%{customdata[2]}"
                "<br>n=%{customdata[3]:.0f}<extra></extra>"
            ),
        ),
        2,
        "mean",
    )

    fig.add_vline(x=0, line_color="#444", line_width=1, row=2, col=1)
    fig.update_xaxes(title_text="Arm", row=1, col=1)
    fig.update_yaxes(title_text="Final full CORE", row=1, col=1)
    fig.update_xaxes(
        title_text="Centered-accuracy Δ (fuzzy − raw); negative favors raw, positive favors fuzzy",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y,
        ticktext=[row[0] for row in rows],
        row=2,
        col=1,
    )

    all_mask = [True] * len(trace_kinds)
    mean_mask = [kind == "mean" for kind in trace_kinds]
    seed_mask = [kind == "seed" for kind in trace_kinds]
    height = max(1050, 510 + len(rows) * 31)
    fig.update_layout(
        title=None,
        template="plotly_white",
        autosize=True,
        height=height,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=190, r=55, t=125, b=125),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                xanchor="center",
                y=1.07,
                yanchor="bottom",
                font=dict(size=12),
                buttons=[
                    dict(label="Means + seeds", method="update", args=[{"visible": all_mask}]),
                    dict(label="Means ±1 SD", method="update", args=[{"visible": mean_mask}]),
                    dict(label="Individual seeds", method="update", args=[{"visible": seed_mask}]),
                ],
            )
        ],
    )
    fig.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        div_id="fortified-core-dashboard",
        default_width="100%",
        default_height=f"{height}px",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )
    header = """
<header class="dashboard-header">
  <h1>FineWeb-EDU-Fortified Fuzzy Dedup × NanoChat d24</h1>
  <p>Interactive LM Eval Harness analysis · Final full CORE and per-task paired centered-accuracy deltas</p>
</header>
"""
    styles = """
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FineWeb-EDU-Fortified Fuzzy Dedup × NanoChat d24 · LM Eval</title>
<style>
  html, body { margin: 0; padding: 0; }
  .dashboard-header {
    box-sizing: border-box;
    width: 100%;
    padding: 24px 24px 4px;
    text-align: center;
    color: #2a3f5f;
    font-family: Arial, Helvetica, sans-serif;
  }
  .dashboard-header h1 {
    margin: 0;
    font-size: clamp(22px, 2.2vw, 34px);
    font-weight: 500;
    line-height: 1.2;
  }
  .dashboard-header p {
    margin: 6px 0 0;
    font-size: clamp(14px, 1.2vw, 18px);
    font-weight: 400;
    line-height: 1.35;
  }
  @media (max-width: 720px) {
    .dashboard-header { padding: 16px 12px 0; }
  }
</style>
"""
    page = output.read_text(encoding="utf-8")
    if page.count("</head>") != 1 or page.count("<body>") != 1:
        raise RuntimeError("Unexpected Plotly HTML shell")
    page = page.replace("</head>", f"{styles}</head>", 1)
    page = page.replace("<body>", f"<body>{header}", 1)
    output.write_text(page, encoding="utf-8")


def plot_dataset_audits(audit_json: Path, pairs_jsonl: Path, output_dir: Path):
    if audit_json.is_file():
        audit = load_json(audit_json)
        hist = audit.get("components", {}).get("component_size_histogram", {})
        if hist:
            sizes = np.asarray([int(key) for key in hist]); counts = np.asarray([int(value) for value in hist.values()])
            order = np.argsort(sizes)
            fig, ax = plt.subplots(figsize=(7.5, 4.8)); ax.loglog(sizes[order], counts[order], marker=".", linestyle="none")
            ax.set(xlabel="Component size", ylabel="Number of components"); ax.grid(alpha=0.25)
            fig.tight_layout(); _save_figure(fig, output_dir / "component_size_histogram.svg"); plt.close(fig)
    if pairs_jsonl.is_file():
        values = []
        for line in pairs_jsonl.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); value = row.get("char_ngram_jaccard")
            if value is not None: values.append(float(value))
        if values:
            x = np.sort(values); y = np.arange(1, len(x) + 1) / len(x)
            fig, ax = plt.subplots(figsize=(7.5, 4.8)); ax.plot(x, y)
            ax.set(xlabel="Exact 24-character-ngram Jaccard", ylabel="ECDF"); ax.grid(alpha=0.25)
            fig.tight_layout(); _save_figure(fig, output_dir / "jaccard_ecdf.svg"); plt.close(fig)


def generate_report(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    records = []
    for arm in ARMS:
        for seed in SEEDS:
            path = run_root / "full" / arm / f"seed_{seed}" / "run_summary.json"
            if path.is_file(): records.append(load_json(path))
    by_key = {(row["arm"], int(row["seed"])): row for row in records}
    pairs = [(seed, by_key[("fuzzy", seed)], by_key[("raw", seed)]) for seed in SEEDS if ("fuzzy", seed) in by_key and ("raw", seed) in by_key]
    if not args.allow_incomplete and len(pairs) != 3:
        raise RuntimeError(f"Expected 3 completed pairs, found {len(pairs)}")
    if pairs:
        for seed, fuzzy, raw in pairs:
            if fuzzy["config"]["validation_sha256"] != raw["config"]["validation_sha256"]:
                raise RuntimeError(f"seed {seed}: validation checksum mismatch")
            f0, r0 = fuzzy["curves"]["val_bpb"][0]["bpb"], raw["curves"]["val_bpb"][0]["bpb"]
            if abs(f0 - r0) >= 1e-6:
                raise RuntimeError(f"seed {seed}: step-0 BPB mismatch {f0} vs {r0}")
        _plot_metric(records, "val_bpb", "bpb", "Validation BPB", output / "val_bpb_vs_tokens.svg")
        _plot_metric(records, "train_loss", "loss", "Training loss EMA (diagnostic)", output / "train_loss_vs_tokens.svg")
        _plot_metric(records, "online_core", "core", "Online CORE (max 500/task)", output / "online_core_vs_tokens.svg")
        plot_delta(pairs, output / "bpb_delta_vs_tokens.svg")
        plot_interactive_validation(pairs, output / "interactive_validation_analysis.html")
        plot_bpb_time(records, output / "val_bpb_vs_time.svg")
        plot_final_core(records, output / "final_core.svg")
        plot_task_delta(pairs, output / "core_task_delta.svg")
        plot_interactive_core(pairs, output / "interactive_core_analysis.html")
    plot_dataset_audits(args.component_audit, args.pair_audit, output)

    run_rows = []
    final_bpb_deltas, final_core_deltas, exposure_values = [], [], []
    token_efficiency = []
    for seed, fuzzy, raw in pairs:
        fb, rb = fuzzy["metrics"]["final_train_val_bpb"], raw["metrics"]["final_train_val_bpb"]
        fc, rc = fuzzy["metrics"]["final_core"], raw["metrics"]["final_core"]
        final_bpb_deltas.append(fb - rb); final_core_deltas.append(fc - rc)
        target = rb
        fuzzy_tokens = first_crossing(fuzzy["curves"]["val_bpb"], target)
        raw_tokens = first_crossing(raw["curves"]["val_bpb"], target)
        token_efficiency.append({"seed": seed, "target_bpb": target, "fuzzy_tokens": fuzzy_tokens, "raw_tokens": raw_tokens})
        exposure = (raw["metrics"].get("final_data_exposure") or {}).get("fuzzy_removed_target_fraction")
        if exposure is not None: exposure_values.append(float(exposure))
        run_rows.append([seed, _fmt(fb), _fmt(rb), _fmt(fb-rb), _fmt(fc,4), _fmt(rc,4), _fmt(fc-rc,4), _fmt(exposure,4)])
    bpb_mean, bpb_sd = _mean_sd(final_bpb_deltas); core_mean, core_sd = _mean_sd(final_core_deltas); exp_mean, exp_sd = _mean_sd(exposure_values)

    fuzzy_manifest = load_json(args.fuzzy_manifest) if args.fuzzy_manifest.is_file() else {}
    raw_manifest = load_json(args.raw_manifest) if args.raw_manifest.is_file() else {}
    dedup_manifest = load_json(args.dedup_manifest) if args.dedup_manifest.is_file() else {}
    dataset_rows = []
    for arm, manifest in (("fuzzy", fuzzy_manifest), ("raw", raw_manifest)):
        selection = manifest.get("selection", {})
        dataset_rows.append([arm, f"{selection.get('docs', 0):,}", f"{selection.get('tokens', 0):,}", selection.get("excluded_validation_overlap_docs", 0), _fmt(selection.get("removed_token_fraction"),4)])
    subset_rows = []
    for subset in sorted(set(raw_manifest.get("by_subset", {})) | set(fuzzy_manifest.get("by_subset", {}))):
        raw_values = raw_manifest.get("by_subset", {}).get(subset, {})
        fuzzy_values = fuzzy_manifest.get("by_subset", {}).get(subset, {})
        raw_tokens = int(raw_values.get("tokens", 0)); fuzzy_tokens = int(fuzzy_values.get("tokens", 0))
        subset_rows.append([subset, raw_values.get("docs", 0), fuzzy_values.get("docs", 0), _fmt(fuzzy_tokens / raw_tokens if raw_tokens else None, 4)])
    length_rows = []
    for label in ("lt128", "128_511", "512_2047", "ge2048"):
        raw_values = raw_manifest.get("by_length", {}).get(label, {})
        fuzzy_values = fuzzy_manifest.get("by_length", {}).get(label, {})
        raw_tokens = int(raw_values.get("tokens", 0)); fuzzy_tokens = int(fuzzy_values.get("tokens", 0))
        length_rows.append([label, raw_values.get("docs", 0), fuzzy_values.get("docs", 0), _fmt(fuzzy_tokens / raw_tokens if raw_tokens else None, 4)])
    qualitative_rows = []
    if args.pair_audit.is_file():
        for line in args.pair_audit.read_text(encoding="utf-8").splitlines()[:5]:
            row = json.loads(line)
            clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")[:180]
            qualitative_rows.append([_fmt(row.get("char_ngram_jaccard"), 3), row.get("component_size"), clean(row.get("removed_preview", "")), clean(row.get("keeper_preview", ""))])
    full_raw = dedup_manifest.get("stats", {}).get("raw", {})
    full_fuzzy = dedup_manifest.get("stats", {}).get("post_fuzzy", {})
    conclusions = {
        "fuzzy_reaches_raw_target_earlier_seeds": sum(1 for row in token_efficiency if row["fuzzy_tokens"] is not None and row["raw_tokens"] is not None and row["fuzzy_tokens"] < row["raw_tokens"]),
        "final_bpb_delta_mean": bpb_mean,
        "final_bpb_delta_sd": bpb_sd,
        "final_core_delta_mean": core_mean,
        "final_core_delta_sd": core_sd,
        "raw_removed_target_fraction_mean": exp_mean,
        "raw_removed_target_fraction_sd": exp_sd,
    }
    result = {"generated_at": utc_now(), "completed_pairs": len(pairs), "runs": records, "token_efficiency": token_efficiency, "conclusions": conclusions, "dataset": {"fuzzy": fuzzy_manifest, "raw": raw_manifest, "full_dedup": dedup_manifest}}
    atomic_json(output / "results.json", result)
    with (output / "run_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["seed","fuzzy_bpb","raw_bpb","delta_bpb","fuzzy_core","raw_core","delta_core","raw_removed_target_fraction"]); writer.writerows(run_rows)

    lines = [
        "# FineWeb-EDU-Fortified fuzzy dedup × NanoChat d24",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Executive summary",
        "",
        f"Completed paired seeds: **{len(pairs)}**.",
        f"Mean paired final BPB delta (fuzzy - raw): **{_fmt(bpb_mean)} ± {_fmt(bpb_sd)}**; lower is better.",
        f"Mean paired final CORE delta: **{_fmt(core_mean,4)} ± {_fmt(core_sd,4)}**; higher is better.",
        f"Fuzzy reached the paired raw target earlier in **{conclusions['fuzzy_reaches_raw_target_earlier_seeds']}/{len(pairs)}** seeds.",
        f"Mean raw target-token exposure to fuzzy-removed documents: **{_fmt(exp_mean,4)} ± {_fmt(exp_sd,4)}**.",
        "",
        "## Run results",
        "",
        _table(["Seed","Fuzzy BPB","Raw BPB","Δ BPB","Fuzzy CORE","Raw CORE","Δ CORE","Raw removed-target fraction"], run_rows),
        "",
        "## Quantitative Analysis",
        "",
        "![Validation BPB](val_bpb_vs_tokens.png)",
        "",
        "[Open the interactive validation dashboard](https://fhschina.github.io/nanochat/reports/semdedup_quality/fineweb_edu_fortified/fuzzy_ab/interactive_validation_analysis.html) — zoom, hover, toggle individual seeds versus cross-seed means, and inspect paired deltas versus tokens or wall time.",
        "",
        "![Paired BPB delta](bpb_delta_vs_tokens.png)",
        "",
        "![Training loss diagnostic](train_loss_vs_tokens.png)",
        "",
        "![Online CORE](online_core_vs_tokens.png)",
        "",
        "![Validation BPB versus optimization time](val_bpb_vs_time.png)",
        "",
        "## LM Eval Harness Results",
        "",
        "[Open the interactive LM Eval dashboard](https://fhschina.github.io/nanochat/reports/semdedup_quality/fineweb_edu_fortified/fuzzy_ab/interactive_core_analysis.html) — hover over final CORE and per-task paired deltas, and toggle cross-seed means versus individual seeds.",
        "",
        "![Final CORE](final_core.png)",
        "",
        "![Per-task CORE delta](core_task_delta.png)",
        "",
        "Each benchmark row contains exactly three individual points: the paired centered-accuracy deltas for seeds 42, 43, and 44 (`fuzzy - raw` within the same seed). The colored diamond/circle is their paired mean and the horizontal error bar is ±1 sample SD. A filled diamond means all three seeds agree in direction; an open circle means mixed directions. No significance p-value is reported for n=3.",
        "",
        "## Dataset Analysis",
        "",
        _table(["Arm","Selected docs","Selected source tokens","Excluded val-overlap docs","Removed token fraction"], dataset_rows),
        "",
        f"Full corpus raw/fuzzy documents: **{int(full_raw.get('docs', 0)):,} / {int(full_fuzzy.get('docs', 0)):,}**.",
        f"Full corpus raw/fuzzy NanoChat tokens: **{int(full_raw.get('nanochat_tokens', 0)):,} / {int(full_fuzzy.get('nanochat_tokens', 0)):,}**.",
        "",
        "### Retention by Common Crawl snapshot",
        "",
        _table(["Subset","Raw sampled docs","Fuzzy sampled docs","Token keep ratio"], subset_rows),
        "",
        "### Retention by document length",
        "",
        _table(["Token bin","Raw sampled docs","Fuzzy sampled docs","Token keep ratio"], length_rows),
        "",
        "![Component sizes](component_size_histogram.png)",
        "",
        "![Jaccard ECDF](jaccard_ecdf.png)",
        "",
        "### Removed → keeper examples",
        "",
        _table(["Jaccard","Component size","Removed preview","Keeper preview"], qualitative_rows),
        "",
        "## Reproducibility",
        "",
        f"Validation checksum: `{records[0]['config']['validation_sha256'] if records else '-'}`.",
        f"Fuzzy view manifest checksum: `{sha256_file(args.fuzzy_manifest) if args.fuzzy_manifest.is_file() else '-'}`.",
        f"Raw view manifest checksum: `{sha256_file(args.raw_manifest) if args.raw_manifest.is_file() else '-'}`.",
        "",
        "## Interpretation",
        "",
        "- Token efficiency is evaluated against each paired raw run's final BPB using a cumulative-min validation curve.",
        "- Final BPB and CORE deltas are paired by model seed; all individual seeds are retained because n=3 is too small for a meaningful significance test.",
        "- Training loss is diagnostic only because the two arms see different document distributions.",
        "- Online CORE uses at most 500 examples per task; final CORE is the full evaluation.",
    ]
    (output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__); commands = root.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record-config")
    record.add_argument("--run-dir", type=Path, required=True); record.add_argument("--arm", choices=ARMS, required=True); record.add_argument("--seed", type=int, required=True); record.add_argument("--kind", choices=("smoke","full"), required=True); record.add_argument("--model-tag", required=True); record.add_argument("--train-manifest", type=Path, required=True); record.add_argument("--validation-file", type=Path, required=True)
    record.add_argument("--depth", type=int, default=24); record.add_argument("--num-iterations", type=int, required=True); record.add_argument("--total-batch-size", type=int, default=TOTAL_BATCH_SIZE); record.add_argument("--max-seq-len", type=int, default=2048); record.add_argument("--device-batch-size", type=int, default=16); record.add_argument("--eval-every", type=int, default=250); record.add_argument("--eval-tokens", type=int, default=41_943_040); record.add_argument("--core-metric-every", type=int, default=1000); record.add_argument("--core-metric-max-per-task", type=int, default=500); record.add_argument("--core-eval-seed", type=int, default=1337); record.add_argument("--window-pattern", default="L"); record.add_argument("--fp8", action="store_true")
    summarize = commands.add_parser("summarize-run"); summarize.add_argument("--run-dir", type=Path, required=True)
    report = commands.add_parser("report"); report.add_argument("--run-root", type=Path, required=True); report.add_argument("--output-dir", type=Path, required=True); report.add_argument("--fuzzy-manifest", type=Path, required=True); report.add_argument("--raw-manifest", type=Path, required=True); report.add_argument("--dedup-manifest", type=Path, required=True); report.add_argument("--component-audit", type=Path, required=True); report.add_argument("--pair-audit", type=Path, required=True); report.add_argument("--allow-incomplete", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "record-config": record_config(args)
    elif args.command == "summarize-run": summarize_run(args.run_dir.expanduser().resolve())
    else: generate_report(args)


if __name__ == "__main__": main()
