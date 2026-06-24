"""
Generate a cosine-similarity ECDF plot from NeMo Curator SemDeDup pairwise output.

The pairwise parquet files contain one max-similarity row per input document. For
cosine distance SemDeDup, eps maps to a similarity cutoff of 1 - eps. Documents
with cosine_sim_score >= 1 - eps are the documents removed by the dedup pass.
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_SCORE_COLUMN = "cosine_sim_score"


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


def _load_scores(pairwise_dir: Path, score_column: str) -> np.ndarray:
    paths = sorted(pairwise_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {pairwise_dir}")

    chunks = []
    for path in paths:
        table = pq.read_table(path, columns=[score_column])
        arr = table.column(score_column).combine_chunks().to_numpy(zero_copy_only=False)
        chunks.append(np.asarray(arr, dtype=np.float64))

    scores = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError(f"No finite {score_column!r} values found under {pairwise_dir}")
    scores.sort()
    return scores


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _polyline(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    first, *rest = points
    parts = [f"M {first[0]:.2f} {first[1]:.2f}"]
    parts.extend(f"L {x:.2f} {y:.2f}" for x, y in rest)
    return " ".join(parts)


def _svg(
    scores: np.ndarray,
    threshold: float,
    eps: float | None,
    dataset_label: str,
    output_stats: dict[str, Any],
    max_points: int,
) -> str:
    width, height = 1200, 760
    left, right, top, bottom = 110, 45, 105, 95
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return left + max(0.0, min(1.0, value)) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - max(0.0, min(1.0, value))) * plot_h

    n = scores.size
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(np.int64))
    ecdf_y = (idx + 1) / n
    points = [(sx(float(scores[i])), sy(float(ecdf_y[j]))) for j, i in enumerate(idx)]

    threshold_x = sx(threshold)
    below_ratio = float(output_stats["below_ratio"])
    threshold_y = sy(below_ratio)
    removed_text = (
        f"eps={eps:g} (sim>={threshold:.2f}) | removed "
        f"{_fmt_int(output_stats['removed_documents'])} ({_fmt_pct(output_stats['removed_ratio'])})"
        if eps is not None
        else f"sim>={threshold:.2f} | removed {_fmt_int(output_stats['removed_documents'])} ({_fmt_pct(output_stats['removed_ratio'])})"
    )
    title_eps = f", eps={eps:g}" if eps is not None else ""
    title = f"Semantic similarity ECDF - {dataset_label} (n170{title_eps})"
    subtitle = (
        f"Total documents: {_fmt_int(output_stats['total_documents'])} | "
        f"Removed @ sim>={threshold:.2f}: {_fmt_int(output_stats['removed_documents'])} "
        f"({_fmt_pct(output_stats['removed_ratio'])})"
    )

    grid = []
    tick_labels = []
    for i in range(11):
        v = i / 10
        x = sx(v)
        y = sy(v)
        grid.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" class="grid"/>')
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        tick_labels.append(f'<text x="{x:.2f}" y="{top + plot_h + 32}" class="tick" text-anchor="middle">{v:.1f}</text>')
        tick_labels.append(f'<text x="{left - 18}" y="{y + 5:.2f}" class="tick" text-anchor="end">{v:.1f}</text>')

    annotation_y = max(top + 120, min(top + plot_h - 70, threshold_y - 35))
    annotation_x = left + 145
    path = _polyline(points)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 28px sans-serif; fill: #222; }}
    .subtitle {{ font: 600 21px sans-serif; fill: #333; }}
    .axis {{ stroke: #222; stroke-width: 2; }}
    .grid {{ stroke: #d7d7d7; stroke-width: 1; }}
    .tick {{ font: 16px sans-serif; fill: #333; }}
    .label {{ font: 19px sans-serif; fill: #222; }}
    .curve {{ fill: none; stroke: #2b6f9f; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
    .cutoff {{ stroke: #c45f5f; stroke-width: 2; stroke-dasharray: 8 7; }}
    .annot {{ font: 18px sans-serif; fill: #b44f4f; }}
    .arrow {{ stroke: #c45f5f; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }}
  </style>
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#c45f5f" />
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="{width / 2:.2f}" y="40" class="title" text-anchor="middle">{html.escape(title)}</text>
  <text x="{width / 2:.2f}" y="72" class="subtitle" text-anchor="middle">{html.escape(subtitle)}</text>
  {''.join(grid)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>
  {''.join(tick_labels)}
  <path d="{path}" class="curve"/>
  <line x1="{threshold_x:.2f}" y1="{top}" x2="{threshold_x:.2f}" y2="{top + plot_h}" class="cutoff"/>
  <path d="M {annotation_x:.2f} {annotation_y:.2f} C {annotation_x + 240:.2f} {annotation_y - 25:.2f}, {threshold_x - 180:.2f} {threshold_y - 35:.2f}, {threshold_x - 10:.2f} {threshold_y:.2f}" class="arrow"/>
  <text x="{annotation_x:.2f}" y="{annotation_y - 8:.2f}" class="annot">{html.escape(removed_text)}</text>
  <text x="{left + plot_w / 2:.2f}" y="{height - 30}" class="label" text-anchor="middle">Cosine similarity</text>
  <text x="25" y="{top + plot_h / 2:.2f}" class="label" transform="rotate(-90 25 {top + plot_h / 2:.2f})" text-anchor="middle">Ratio of dataset below the similarity score (ECDF)</text>
</svg>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SemDeDup cosine similarity ECDF as SVG")
    parser.add_argument("--pairwise-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--dataset-label", default="fineweb-edu")
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=1400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.manifest)
    eps = args.eps
    if eps is None:
        eps = _get(manifest, "curator_config.eps")
    threshold = args.similarity_threshold
    if threshold is None:
        if eps is None:
            raise ValueError("Provide --eps or --similarity-threshold")
        threshold = 1.0 - float(eps)

    scores = _load_scores(args.pairwise_dir.expanduser().resolve(), args.score_column)
    below = int(np.searchsorted(scores, threshold, side="left"))
    total = int(scores.size)
    removed = total - below
    stats = {
        "score_column": args.score_column,
        "total_documents": total,
        "eps": eps,
        "similarity_threshold": threshold,
        "below_documents": below,
        "below_ratio": below / total,
        "removed_documents": removed,
        "removed_ratio": removed / total,
        "min": float(scores[0]),
        "max": float(scores[-1]),
        "quantiles": {f"p{int(q * 1000):03d}": float(np.quantile(scores, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999)},
        "manifest_removed_docs": _get(manifest, "removed_docs"),
        "manifest_keep_ratio_docs": _get(manifest, "keep_ratio_docs"),
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_svg(scores, threshold, eps, args.dataset_label, stats, args.max_points))
    stats_output = args.stats_output.expanduser().resolve() if args.stats_output else output.with_suffix(".json")
    stats_output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(f"Wrote ECDF SVG: {output}")
    print(f"Wrote ECDF stats: {stats_output}")


if __name__ == "__main__":
    main()
