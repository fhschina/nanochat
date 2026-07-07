"""
Audit kept vs removed documents for SemDeDup/random-drop experiments.

The script is dataset-neutral but tuned for FineWeb-EDU follow-up reporting. It
collects deterministic removed/kept samples and reports lightweight text-pattern
heuristics for triage. These heuristics are not manual labels and must not be
reported as manual precision.
"""

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
QA_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\?",
        r"\b(question|answer|answers|quiz|exam|worksheet|multiple choice|which of|what is|why does|how many|because)\b",
        r"\b(a\.|b\.|c\.|d\.)\s+\w+",
    ]
]
BOOLQ_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(true or false|yes or no|is it true|is this true|is the statement|does this mean)\b",
        r"\b(can|could|do|does|did|is|are|was|were|has|have|had|will|would|should)\b[^?]{0,160}\?",
        r"\b(passages?|paragraph|according to the text|based on the passage)\b",
    ]
]
EDU_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(lesson|teacher|student|classroom|curriculum|learning objective|grade level|homework|assignment)\b",
        r"\b(explain|definition|example|practice|exercise|chapter|section|study guide)\b",
    ]
]
BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(cookie|privacy policy|terms of use|subscribe|newsletter|advertisement|all rights reserved)\b",
        r"\b(click here|read more|sign up|log in|related articles|share this)\b",
        r"https?://|www\.",
    ]
]
TEMPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(table of contents|related links|posted by|last updated|printable version)\b",
        r"\b(page \d+ of \d+|copyright \d{4}|navigation menu)\b",
    ]
]


def _json_load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _artifact_path(run_dir: Path, manifest: dict, key: str, fallback_name: str) -> Path:
    artifacts = manifest.get("analysis_artifacts", {}) if isinstance(manifest, dict) else {}
    value = artifacts.get(key)
    if value:
        path = Path(value).expanduser()
        if path.exists():
            return path
    return run_dir / fallback_name


def _parquet_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix == ".parquet" and not p.name.endswith(".tmp"))


def _selected_train_files(data_dir: Path, num_train_shards: int, train_file_names: list[str] | None) -> list[Path]:
    files = _parquet_files(data_dir)
    if train_file_names:
        wanted = set(train_file_names)
        files = [p for p in files if p.name in wanted]
    else:
        files = files[:-1]
        if num_train_shards > 0:
            files = files[:num_train_shards]
    if not files:
        raise FileNotFoundError(f"No train parquet files selected from {data_dir}")
    return files


def _stream_original_records(data_dir: Path, train_file_names: list[str] | None, num_train_shards: int):
    for path in _selected_train_files(data_dir, num_train_shards, train_file_names):
        pf = pq.ParquetFile(path)
        row_offset = 0
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["text"])
            texts = [(text or "") for text in table.column("text").to_pylist()]
            for idx, text in enumerate(texts):
                doc_id = f"{path.name}:{row_offset + idx}"
                yield {
                    "doc_id": doc_id,
                    "source_file": path.name,
                    "char_len": len(text),
                    "token_len": None,
                    "text_preview": " ".join(text[:1200].split()),
                }
            row_offset += len(texts)


def _collect_samples(
    input_data_dir: Path,
    removed_ids: set[str],
    train_file_names: list[str] | None,
    num_train_shards: int,
    sample_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    removed_target = set(rng.sample(sorted(removed_ids), min(sample_size, len(removed_ids))))
    removed_records: list[dict] = []
    kept_reservoir: list[dict] = []
    kept_seen = 0

    for rec in _stream_original_records(input_data_dir, train_file_names, num_train_shards):
        doc_id = rec["doc_id"]
        if doc_id in removed_target:
            removed_records.append(rec)
        elif doc_id not in removed_ids:
            kept_seen += 1
            if len(kept_reservoir) < sample_size:
                kept_reservoir.append(rec)
            else:
                j = rng.randrange(kept_seen)
                if j < sample_size:
                    kept_reservoir[j] = rec
        if len(removed_records) >= len(removed_target) and len(kept_reservoir) >= sample_size and kept_seen > sample_size * 4:
            # Keep scanning only long enough for a stable kept reservoir on large corpora.
            continue
    return removed_records, kept_reservoir


def _quantiles(values: list[int | float | None]) -> dict[str, int | float | None]:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return {f"p{int(q * 100):02d}": None for q in QUANTILES} | {"mean": None}
    n = len(clean)
    out = {}
    for q in QUANTILES:
        idx = int(round(q * (n - 1)))
        out[f"p{int(q * 100):02d}"] = clean[idx]
    out["mean"] = sum(clean) / n
    return out


def _matches(patterns: list[re.Pattern], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _summarize_records(records: list[dict]) -> dict:
    docs = len(records)
    chars = [rec.get("char_len") for rec in records]
    tokens = [rec.get("token_len") for rec in records]
    source_counts = collections.Counter(rec.get("source_file") or str(rec.get("doc_id", "")).split(":", 1)[0] for rec in records)
    buckets = {
        "qa_like": 0,
        "boolq_like": 0,
        "education_like": 0,
        "boilerplate_like": 0,
        "template_like": 0,
        "question_mark": 0,
    }
    for rec in records:
        text = rec.get("text_preview") or ""
        if _matches(QA_PATTERNS, text):
            buckets["qa_like"] += 1
        if _matches(BOOLQ_PATTERNS, text):
            buckets["boolq_like"] += 1
        if _matches(EDU_PATTERNS, text):
            buckets["education_like"] += 1
        if _matches(BOILERPLATE_PATTERNS, text):
            buckets["boilerplate_like"] += 1
        if _matches(TEMPLATE_PATTERNS, text):
            buckets["template_like"] += 1
        if "?" in text:
            buckets["question_mark"] += 1
    return {
        "docs": docs,
        "char_length_quantiles": _quantiles(chars),
        "token_length_quantiles": _quantiles(tokens),
        "top_sources": source_counts.most_common(20),
        "pattern_counts": buckets,
        "pattern_rates": {key: (value / docs if docs else None) for key, value in buckets.items()},
    }


def _source_counts(ids: list[str]) -> dict:
    counts = collections.Counter(doc_id.split(":", 1)[0] for doc_id in ids)
    return {"unique_sources": len(counts), "top_sources": counts.most_common(30)}


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown(path: Path, payload: dict) -> None:
    removed = payload["removed_sample"]
    kept = payload["kept_sample"]
    lines = [
        f"# {payload['dataset_label']} Removed Sample Audit",
        "",
        f"Run dir: `{payload['run_dir']}`",
        f"Removed id count: `{payload['removed_id_count']}`",
        "",
        "Manual precision status: not completed. The rates below are heuristics only.",
        "",
        "## Sample Summary",
        "",
        "| Group | Docs | p50 chars | p95 chars | QA-like | BoolQ-like | Education-like | Boilerplate-like | Template-like |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in [("removed", removed), ("kept", kept)]:
        q = summary["char_length_quantiles"]
        rates = summary["pattern_rates"]
        lines.append(
            f"| {name} | {summary['docs']} | {_fmt(q.get('p50'))} | {_fmt(q.get('p95'))} | "
            f"{_fmt(rates.get('qa_like'))} | {_fmt(rates.get('boolq_like'))} | "
            f"{_fmt(rates.get('education_like'))} | {_fmt(rates.get('boilerplate_like'))} | {_fmt(rates.get('template_like'))} |"
        )
    lines.extend([
        "",
        "## Top Removed Sources",
        "",
        "| Source | Count |",
        "| --- | ---: |",
    ])
    for source, count in payload["removed_source_counts"].get("top_sources", [])[:20]:
        lines.append(f"| `{source}` | {count} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Heuristic audit is for triage and does not estimate manual duplicate precision.",
        "- BoolQ-like rates are lexical cues, not task contamination labels.",
    ])
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SemDeDup/random-drop removed samples")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run dir containing removed_ids.txt and sample artifacts")
    parser.add_argument("--input-data-dir", type=Path, default=None, help="Original data dir, required to resample beyond existing JSONL samples")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-label", default="FineWeb-EDU")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-train-shards", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or run_dir / "removed_audit").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    semd_manifest = _json_load(run_dir / "semdedup_manifest.json")
    random_manifest = _json_load(run_dir / "random_drop_manifest.json")
    manifest = semd_manifest or random_manifest
    train_file_names = manifest.get("train_files") if manifest else None

    removed_ids_path = _artifact_path(run_dir, manifest, "removed_ids_path", "removed_ids.txt")
    kept_ids_path = _artifact_path(run_dir, manifest, "kept_ids_path", "kept_ids.txt")
    removed_samples_path = _artifact_path(run_dir, manifest, "removed_samples_path", "removed_samples.jsonl")
    kept_samples_path = _artifact_path(run_dir, manifest, "kept_samples_path", "kept_samples.jsonl")

    removed_ids = _read_ids(removed_ids_path)
    kept_ids = _read_ids(kept_ids_path) if kept_ids_path.exists() and kept_ids_path.stat().st_size < 250_000_000 else []
    removed_records = _read_jsonl(removed_samples_path)
    kept_records = _read_jsonl(kept_samples_path)
    if (len(removed_records) < args.sample_size or len(kept_records) < args.sample_size) and args.input_data_dir is not None and removed_ids:
        removed_records, kept_records = _collect_samples(
            args.input_data_dir.expanduser().resolve(),
            set(removed_ids),
            train_file_names,
            args.num_train_shards,
            args.sample_size,
            args.seed,
        )

    removed_records = removed_records[: args.sample_size]
    kept_records = kept_records[: args.sample_size]
    _write_jsonl(output_dir / "removed_audit_samples.jsonl", removed_records)
    _write_jsonl(output_dir / "kept_audit_samples.jsonl", kept_records)

    payload = {
        "dataset_label": args.dataset_label,
        "run_dir": str(run_dir),
        "input_data_dir": str(args.input_data_dir.expanduser().resolve()) if args.input_data_dir else None,
        "sample_size_requested": args.sample_size,
        "seed": args.seed,
        "manual_precision_status": "not_completed",
        "heuristic_warning": "Pattern rates are heuristic triage signals and are not manual duplicate precision.",
        "removed_id_count": len(removed_ids),
        "kept_id_count_loaded": len(kept_ids),
        "removed_source_counts": _source_counts(removed_ids),
        "kept_source_counts_loaded": _source_counts(kept_ids) if kept_ids else {},
        "removed_sample": _summarize_records(removed_records),
        "kept_sample": _summarize_records(kept_records),
        "artifact_paths": {
            "json": str(output_dir / "removed_audit.json"),
            "markdown": str(output_dir / "removed_audit.md"),
            "removed_samples": str(output_dir / "removed_audit_samples.jsonl"),
            "kept_samples": str(output_dir / "kept_audit_samples.jsonl"),
        },
    }
    (output_dir / "removed_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_markdown(output_dir / "removed_audit.md", payload)
    print(f"Wrote audit JSON: {output_dir / 'removed_audit.json'}")
    print(f"Wrote audit report: {output_dir / 'removed_audit.md'}")


if __name__ == "__main__":
    main()
