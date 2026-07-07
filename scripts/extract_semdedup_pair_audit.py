"""
Extract a SemDeDup removed-to-root pair audit sheet when Curator artifacts allow it.

Curator artifact schemas vary across versions. This script searches the
semdedup_manifest.json backend metadata for semantic_dedup_path, then looks for
pairwise parquet columns that identify the removed document and its nearest/root
document. If those columns are unavailable, it degrades to a removed-only audit
sheet and records the reason in the summary.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


LABELS = [
    "exact_duplicate",
    "near_duplicate",
    "semantic_duplicate",
    "same_template_different_content",
    "related_but_both_valuable",
    "false_positive",
    "unclear",
]
ID_COLUMNS = ["doc_id", "id", "_id", "document_id", "docid", "text_id"]
ROOT_COLUMNS = [
    "root_doc_id",
    "root_id",
    "nearest_doc_id",
    "nearest_id",
    "duplicate_doc_id",
    "duplicate_id",
    "match_doc_id",
    "match_id",
    "parent_doc_id",
    "parent_id",
    "max_doc_id",
    "max_id",
]
SCORE_COLUMNS = ["cosine_sim_score", "similarity", "score", "max_similarity", "cosine_similarity"]


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _get(obj: dict[str, Any], dotted: str, default=None):
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _column_names(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def _first_present(names: list[str], candidates: list[str]) -> str | None:
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _candidate_pairwise_dirs(run_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    semantic_path = _get(manifest, "backend_result.metadata.semantic_dedup_path")
    if semantic_path:
        base = Path(str(semantic_path)).expanduser()
        candidates.extend([base / "pairwise_results", base])
    cache_dir = manifest.get("cache_dir")
    if cache_dir:
        base = Path(str(cache_dir)).expanduser()
        candidates.extend([base / "curator_cache/semantic_dedup/pairwise_results", base / "curator_cache/semantic_dedup"])
    candidates.extend([
        run_dir / "semdedup_cache/curator_cache/semantic_dedup/pairwise_results",
        run_dir / "semdedup_cache/curator_cache/semantic_dedup",
    ])
    out = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and list(resolved.rglob("*.parquet")):
            out.append(resolved)
    return out


def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _artifact_path(run_dir: Path, manifest: dict[str, Any], key: str, fallback: str) -> Path:
    value = (manifest.get("analysis_artifacts") or {}).get(key)
    if value:
        path = Path(value).expanduser()
        if path.exists():
            return path
    return run_dir / fallback


def _sample_removed_only(run_dir: Path, manifest: dict[str, Any], sample_size: int, seed: int) -> list[dict[str, Any]]:
    removed_path = _artifact_path(run_dir, manifest, "removed_ids_path", "removed_ids.txt")
    removed_ids = _read_ids(removed_path)
    rng = random.Random(seed)
    sample = rng.sample(removed_ids, min(sample_size, len(removed_ids))) if removed_ids else []
    return [{"removed_doc_id": doc_id, "root_doc_id": None, "similarity_score": None} for doc_id in sample]


def _extract_pairs_from_pairwise(
    pairwise_dir: Path,
    removed_ids: set[str],
    eps: float | None,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parquet_paths = sorted(pairwise_dir.rglob("*.parquet"))
    diagnostics: dict[str, Any] = {"pairwise_dir": str(pairwise_dir), "files": len(parquet_paths)}
    if not parquet_paths:
        diagnostics["reason"] = "no parquet files under pairwise dir"
        return [], diagnostics

    first_columns = _column_names(parquet_paths[0])
    id_col = _first_present(first_columns, ID_COLUMNS)
    root_col = _first_present(first_columns, ROOT_COLUMNS)
    score_col = _first_present(first_columns, SCORE_COLUMNS)
    diagnostics.update({"columns": first_columns, "id_column": id_col, "root_column": root_col, "score_column": score_col})
    if id_col is None:
        diagnostics["reason"] = "no document id column recognized in pairwise output"
        return [], diagnostics
    if root_col is None:
        diagnostics["reason"] = "no root/nearest document column recognized in pairwise output"
        return [], diagnostics

    threshold = 1.0 - eps if eps is not None else None
    rows: list[dict[str, Any]] = []
    for path in parquet_paths:
        names = _column_names(path)
        if id_col not in names or root_col not in names:
            continue
        columns = [id_col, root_col]
        if score_col and score_col in names:
            columns.append(score_col)
        table = pq.read_table(path, columns=columns)
        ids = [str(value) for value in table.column(id_col).to_pylist()]
        roots = [str(value) if value is not None else None for value in table.column(root_col).to_pylist()]
        scores = table.column(score_col).to_pylist() if score_col and score_col in table.column_names else [None] * len(ids)
        for doc_id, root_id, score in zip(ids, roots, scores):
            if removed_ids and doc_id not in removed_ids:
                continue
            if threshold is not None and score is not None:
                try:
                    if float(score) < threshold:
                        continue
                except (TypeError, ValueError):
                    pass
            if root_id in {None, "", "None", doc_id}:
                continue
            rows.append({"removed_doc_id": doc_id, "root_doc_id": root_id, "similarity_score": score})
    rng = random.Random(seed)
    if len(rows) > sample_size:
        rows = rng.sample(rows, sample_size)
    diagnostics["candidate_pairs"] = len(rows)
    return rows, diagnostics


def _parquet_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix == ".parquet" and not p.name.endswith(".tmp"))


def _selected_train_files(data_dir: Path, train_file_names: list[str] | None, num_train_shards: int) -> list[Path]:
    files = _parquet_files(data_dir)
    if train_file_names:
        wanted = set(train_file_names)
        files = [p for p in files if p.name in wanted]
    else:
        files = files[:-1]
        if num_train_shards > 0:
            files = files[:num_train_shards]
    return files


def _load_text_previews(input_data_dir: Path | None, ids: set[str], train_file_names: list[str] | None, num_train_shards: int) -> dict[str, dict[str, Any]]:
    if input_data_dir is None or not ids:
        return {}
    by_file: dict[str, set[int]] = {}
    for doc_id in ids:
        try:
            source_file, row = doc_id.rsplit(":", 1)
            by_file.setdefault(source_file, set()).add(int(row))
        except ValueError:
            continue
    out: dict[str, dict[str, Any]] = {}
    for path in _selected_train_files(input_data_dir, train_file_names, num_train_shards):
        wanted = by_file.get(path.name)
        if not wanted:
            continue
        pf = pq.ParquetFile(path)
        row_offset = 0
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["text"])
            texts = [(text or "") for text in table.column("text").to_pylist()]
            for idx, text in enumerate(texts):
                row = row_offset + idx
                if row not in wanted:
                    continue
                doc_id = f"{path.name}:{row}"
                out[doc_id] = {
                    "source_file": path.name,
                    "char_len": len(text),
                    "text_preview": " ".join(text[:1200].split()),
                }
            row_offset += len(texts)
            if len(out) >= len(ids):
                return out
    return out


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], mode: str, diagnostics: dict[str, Any], labels: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "pair_audit_records.jsonl"
    csv_path = output_dir / "pair_audit_label_sheet.csv"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fieldnames = [
        "label",
        "removed_doc_id",
        "root_doc_id",
        "similarity_score",
        "removed_char_len",
        "root_char_len",
        "removed_preview",
        "root_preview",
        "allowed_labels",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames[:-1]} | {"allowed_labels": "|".join(labels)})
    summary = {
        "mode": mode,
        "manual_precision_status": "not_completed",
        "allowed_labels": labels,
        "record_count": len(rows),
        "degradation_reason": diagnostics.get("reason") if mode == "removed_only" else None,
        "diagnostics": diagnostics,
        "artifact_paths": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "summary": str(output_dir / "pair_audit_summary.json"),
        },
    }
    (output_dir / "pair_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    md = [
        "# SemDeDup Pair Audit Sheet",
        "",
        f"Mode: `{mode}`",
        f"Records: `{len(rows)}`",
        "Manual precision status: not completed.",
        "",
        "Allowed labels: " + ", ".join(labels),
    ]
    if mode == "removed_only":
        md.extend(["", f"Degraded to removed-only audit: {diagnostics.get('reason', 'pair information unavailable')}."])
    (output_dir / "pair_audit.md").write_text("\n".join(md) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SemDeDup removed-to-root pair audit artifacts")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--input-data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-train-shards", type=int, default=-1)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--labels", default=",".join(LABELS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else run_dir / "semdedup_manifest.json"
    manifest = _load_json(manifest_path)
    output_dir = (args.output_dir or run_dir / "pair_audit").expanduser().resolve()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    eps = args.eps if args.eps is not None else _get(manifest, "curator_config.eps")
    eps = float(eps) if eps is not None else None

    removed_ids = set(_read_ids(_artifact_path(run_dir, manifest, "removed_ids_path", "removed_ids.txt")))
    pairwise_dirs = _candidate_pairwise_dirs(run_dir, manifest)
    pairs: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"reason": "no pairwise directory found", "searched_pairwise_dirs": [str(p) for p in pairwise_dirs]}
    for pairwise_dir in pairwise_dirs:
        pairs, diagnostics = _extract_pairs_from_pairwise(pairwise_dir, removed_ids, eps, args.sample_size, args.seed)
        if pairs:
            break
    mode = "pair" if pairs else "removed_only"
    if not pairs:
        pairs = _sample_removed_only(run_dir, manifest, args.sample_size, args.seed)
        diagnostics.setdefault("reason", "pair information unavailable; using removed_ids only")

    ids_to_load = {row["removed_doc_id"] for row in pairs if row.get("removed_doc_id")}
    ids_to_load.update(row["root_doc_id"] for row in pairs if row.get("root_doc_id"))
    previews = _load_text_previews(
        args.input_data_dir.expanduser().resolve() if args.input_data_dir else None,
        ids_to_load,
        manifest.get("train_files") if manifest else None,
        args.num_train_shards,
    )
    enriched = []
    for row in pairs:
        removed = previews.get(row.get("removed_doc_id"), {})
        root = previews.get(row.get("root_doc_id"), {}) if row.get("root_doc_id") else {}
        enriched.append({
            "label": "",
            "removed_doc_id": row.get("removed_doc_id"),
            "root_doc_id": row.get("root_doc_id"),
            "similarity_score": row.get("similarity_score"),
            "removed_char_len": removed.get("char_len"),
            "root_char_len": root.get("char_len"),
            "removed_preview": removed.get("text_preview"),
            "root_preview": root.get("text_preview"),
        })
    _write_outputs(output_dir, enriched, mode, diagnostics, labels)
    print(f"Wrote pair audit artifacts under: {output_dir}")


if __name__ == "__main__":
    main()
