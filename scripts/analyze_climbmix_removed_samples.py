"""
Audit kept vs removed ClimbMix documents for SemDeDup quality investigation.

The script reads SemDeDup/random-drop artifacts and optionally streams the
original ClimbMix parquet to collect larger deterministic samples. It reports
source distribution, length shifts, and simple QA/coreference/boilerplate text
pattern rates. These heuristics are meant for triage, not automatic labeling.
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
        r"\b(question|answer|answers|quiz|exam|multiple choice|which of|what is|why does|how many|because)\b",
        r"\b(a\.|b\.|c\.|d\.)\s+\w+",
    ]
]
COREF_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(he|she|him|her|his|hers|they|them|their|theirs|it|its)\b",
        r"\b(because|although|however|therefore|meanwhile|before|after|while|when)\b",
        r"\b(the former|the latter|this person|that person|someone|somebody)\b",
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


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


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
                    "text_preview": " ".join(text[:1000].split()),
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


def _count_matches(patterns: list[re.Pattern], text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in patterns)


def _summarize_records(records: list[dict]) -> dict:
    docs = len(records)
    chars = [rec.get("char_len") for rec in records]
    tokens = [rec.get("token_len") for rec in records]
    source_counts = collections.Counter(rec.get("source_file") or str(rec.get("doc_id", "")).split(":", 1)[0] for rec in records)
    buckets = {
        "qa_like": 0,
        "coref_like": 0,
        "boilerplate_like": 0,
        "question_mark": 0,
    }
    coref_counts = []
    for rec in records:
        text = rec.get("text_preview") or ""
        if _matches(QA_PATTERNS, text):
            buckets["qa_like"] += 1
        if _matches(COREF_PATTERNS, text):
            buckets["coref_like"] += 1
        if _matches(BOILERPLATE_PATTERNS, text):
            buckets["boilerplate_like"] += 1
        if "?" in text:
            buckets["question_mark"] += 1
        coref_counts.append(_count_matches(COREF_PATTERNS, text))
    return {
        "docs": docs,
        "char_length_quantiles": _quantiles(chars),
        "token_length_quantiles": _quantiles(tokens),
        "top_sources": source_counts.most_common(20),
        "pattern_counts": buckets,
        "pattern_rates": {key: (value / docs if docs else None) for key, value in buckets.items()},
        "coref_pattern_count_quantiles": _quantiles(coref_counts),
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
        "# ClimbMix Removed Sample Audit",
        "",
        f"Run dir: `{payload['run_dir']}`",
        f"Removed id count: `{payload['removed_id_count']}`",
        "",
        "## Sample Summary",
        "",
        "| Group | Docs | p50 chars | p95 chars | QA-like | Coref-like | Boilerplate-like |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in [("removed", removed), ("kept", kept)]:
        q = summary["char_length_quantiles"]
        rates = summary["pattern_rates"]
        lines.append(
            f"| {name} | {summary['docs']} | {_fmt(q.get('p50'))} | {_fmt(q.get('p95'))} | "
            f"{_fmt(rates.get('qa_like'))} | {_fmt(rates.get('coref_like'))} | {_fmt(rates.get('boilerplate_like'))} |"
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
        "## Interpretation Hints",
        "",
        "- If removed QA-like/coreference rates are much higher than kept, inspect those samples manually before trusting task deltas.",
        "- If removed source counts are concentrated in a few shards, check whether those shards map to domains that resemble CORE tasks.",
        "- If boilerplate is much higher in removed docs, SemDeDup may be removing mostly low-value duplication.",
    ])
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SemDeDup/random-drop removed samples")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run dir containing removed_ids.txt and sample artifacts")
    parser.add_argument("--input-data-dir", type=Path, default=None, help="Original ClimbMix data dir, required to resample beyond existing JSONL samples")
    parser.add_argument("--output-dir", type=Path, default=None)
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

    removed_ids_path = run_dir / "removed_ids.txt"
    kept_ids_path = run_dir / "kept_ids.txt"
    removed_ids = _read_ids(removed_ids_path)
    kept_ids = _read_ids(kept_ids_path) if kept_ids_path.exists() and kept_ids_path.stat().st_size < 250_000_000 else []

    removed_records = _read_jsonl(run_dir / "removed_samples.jsonl")
    kept_records = _read_jsonl(run_dir / "kept_samples.jsonl")
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
        "run_dir": str(run_dir),
        "input_data_dir": str(args.input_data_dir.expanduser().resolve()) if args.input_data_dir else None,
        "sample_size_requested": args.sample_size,
        "seed": args.seed,
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
