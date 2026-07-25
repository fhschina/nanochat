"""Build deterministic stable-hash parquet views for the Fortified fuzzy A/B."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import xxhash

FORMAT_VERSION = 1
HASH_SPACE = 1 << 128
OUTPUT_COLUMNS = ("text", "subset", "doc_id", "nanochat_token_count", "shuffle_hash")
_SCAN_CONTEXT: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def source_files(root: Path) -> list[Path]:
    manifest = root / "materialize_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [root / row["file"] for row in payload.get("shards", [])]
        if paths and all(path.is_file() for path in paths):
            return sorted(paths)
    paths = sorted(root.rglob("*.parquet"))
    paths = [path for path in paths if not path.name.endswith(".tmp")]
    if not paths:
        raise FileNotFoundError(f"No parquet files below {root}")
    return paths


def stable_hash(subset: str, doc_id: str, seed: int) -> int:
    key = subset.encode("utf-8") + b"\0" + doc_id.encode("utf-8")
    return xxhash.xxh3_128_intdigest(key, seed=seed)


def text_hash(text: str) -> bytes:
    return xxhash.xxh3_128_digest(text.encode("utf-8"))


def hash_bounds(start: float, end: float) -> tuple[int, int]:
    if not 0 <= start < end <= 1:
        raise ValueError("hash interval must satisfy 0 <= start < end <= 1")
    return int(start * HASH_SPACE), int(end * HASH_SPACE)


def bucket_for(value: int, lower: int, upper: int, buckets: int) -> int:
    return min(buckets - 1, ((value - lower) * buckets) // (upper - lower))


def length_bin(token_count: int) -> str:
    if token_count < 128:
        return "lt128"
    if token_count < 512:
        return "128_511"
    if token_count < 2048:
        return "512_2047"
    return "ge2048"


def validation_hashes(path: Path) -> tuple[set[bytes], int]:
    hashes: set[bytes] = set()
    docs = 0
    pf = pq.ParquetFile(path)
    if "text" not in pf.schema_arrow.names:
        raise ValueError(f"Validation parquet has no text column: {path}")
    for row_group in range(pf.num_row_groups):
        texts = pf.read_row_group(row_group, columns=["text"])["text"].to_pylist()
        docs += len(texts)
        hashes.update(text_hash(text) for text in texts)
    return hashes, docs


def empty_buffers(buckets: int) -> list[dict[str, list]]:
    return [{name: [] for name in OUTPUT_COLUMNS} for _ in range(buckets)]


def output_schema(raw: bool) -> pa.Schema:
    fields = [
        pa.field("text", pa.string(), nullable=False),
        pa.field("subset", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("nanochat_token_count", pa.int64(), nullable=False),
        pa.field("shuffle_hash", pa.binary(16), nullable=False),
    ]
    if raw:
        fields.append(pa.field("removed_by_fuzzy", pa.bool_(), nullable=False))
    return pa.schema(fields)


def flush_buffers(buffers, staging: Path, fragment_index: int, worker_index: int = 0) -> int:
    for bucket, columns in enumerate(buffers):
        if not columns["text"]:
            continue
        directory = staging / f"bucket_{bucket:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        table = pa.table(columns, schema=output_schema(raw=False))
        pq.write_table(
            table,
            directory / f"fragment_w{worker_index:04d}_{fragment_index:06d}.parquet",
            compression="zstd",
            row_group_size=8192,
        )
    return fragment_index + 1


def configure_scan_context(
    *,
    lower: int,
    upper: int,
    num_buckets: int,
    shuffle_seed: int,
    validation_hashes_set: set[bytes],
    staging: Path,
    buffer_rows: int,
) -> None:
    global _SCAN_CONTEXT
    _SCAN_CONTEXT = {
        "lower": lower,
        "upper": upper,
        "num_buckets": num_buckets,
        "shuffle_seed": shuffle_seed,
        "validation_hashes": validation_hashes_set,
        "staging": staging,
        "buffer_rows": buffer_rows,
    }


def scan_partition(task: tuple[int, list[tuple[int, str]]]) -> dict[str, int]:
    worker_index, indexed_paths = task
    if not _SCAN_CONTEXT:
        raise RuntimeError("scan worker context was not initialized")
    lower = int(_SCAN_CONTEXT["lower"])
    upper = int(_SCAN_CONTEXT["upper"])
    num_buckets = int(_SCAN_CONTEXT["num_buckets"])
    shuffle_seed = int(_SCAN_CONTEXT["shuffle_seed"])
    val_hashes = _SCAN_CONTEXT["validation_hashes"]
    staging = Path(_SCAN_CONTEXT["staging"])
    buffer_rows = int(_SCAN_CONTEXT["buffer_rows"])

    buffers = empty_buffers(num_buckets)
    buffered_rows = 0
    fragment_index = 0
    stats = {
        "files": 0,
        "scanned_docs": 0,
        "selected_docs": 0,
        "selected_tokens": 0,
        "excluded_val_docs": 0,
        "excluded_val_tokens": 0,
    }
    required = {"text", "subset", "doc_id", "nanochat_token_count"}
    for local_index, (_, path_string) in enumerate(indexed_paths, start=1):
        path = Path(path_string)
        pf = pq.ParquetFile(path)
        missing = required - set(pf.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for rg_index in range(pf.num_row_groups):
            rows = pf.read_row_group(rg_index, columns=sorted(required)).to_pydict()
            for text, subset, doc_id, tokens in zip(
                rows["text"], rows["subset"], rows["doc_id"], rows["nanochat_token_count"], strict=True
            ):
                stats["scanned_docs"] += 1
                digest_int = stable_hash(subset, doc_id, shuffle_seed)
                if digest_int < lower or digest_int >= upper:
                    continue
                tokens = int(tokens)
                if text_hash(text) in val_hashes:
                    stats["excluded_val_docs"] += 1
                    stats["excluded_val_tokens"] += tokens
                    continue
                bucket = bucket_for(digest_int, lower, upper, num_buckets)
                target = buffers[bucket]
                target["text"].append(text)
                target["subset"].append(subset)
                target["doc_id"].append(doc_id)
                target["nanochat_token_count"].append(tokens)
                target["shuffle_hash"].append(digest_int.to_bytes(16, "big"))
                buffered_rows += 1
                stats["selected_docs"] += 1
                stats["selected_tokens"] += tokens
            if buffered_rows >= buffer_rows:
                fragment_index = flush_buffers(buffers, staging, fragment_index, worker_index)
                buffers = empty_buffers(num_buckets)
                buffered_rows = 0
        stats["files"] += 1
        if local_index % 25 == 0:
            print(
                f"scan worker={worker_index:02d} files={local_index}/{len(indexed_paths)} "
                f"docs={stats['scanned_docs']:,} selected={stats['selected_docs']:,}",
                flush=True,
            )
    if buffered_rows:
        flush_buffers(buffers, staging, fragment_index, worker_index)
    print(
        f"scan worker={worker_index:02d} complete files={stats['files']} "
        f"docs={stats['scanned_docs']:,} selected={stats['selected_docs']:,}",
        flush=True,
    )
    return stats


def _table_keys(table: pa.Table) -> list[tuple[str, str]]:
    subsets = table["subset"].to_pylist()
    doc_ids = table["doc_id"].to_pylist()
    return list(zip(subsets, doc_ids, strict=True))


def _validate_unique_sorted(table: pa.Table, bucket: int) -> None:
    hashes = table["shuffle_hash"].to_pylist()
    keys = _table_keys(table)
    previous = None
    seen_same_hash: set[tuple[str, str]] = set()
    for digest, key in zip(hashes, keys, strict=True):
        current = (digest, key[0], key[1])
        if previous is not None and current < previous:
            raise RuntimeError(f"bucket {bucket} is not sorted")
        if key in seen_same_hash:
            raise RuntimeError(f"duplicate document key in bucket {bucket}: {key}")
        if previous is None or digest != previous[0]:
            seen_same_hash.clear()
        seen_same_hash.add(key)
        previous = current


def compact_bucket(
    staging: Path,
    output: Path,
    bucket: int,
    arm: str,
    fuzzy_view: Path | None,
) -> dict[str, Any]:
    fragments = sorted((staging / f"bucket_{bucket:05d}").glob("*.parquet"))
    if not fragments:
        raise RuntimeError(f"Selected view has empty bucket {bucket}; use fewer buckets")
    table = pa.concat_tables([pq.read_table(path) for path in fragments], promote_options="default")
    order = pc.sort_indices(table, sort_keys=[
        ("shuffle_hash", "ascending"), ("subset", "ascending"), ("doc_id", "ascending")
    ])
    table = pc.take(table, order)
    _validate_unique_sorted(table, bucket)

    removed_docs = 0
    removed_tokens = 0
    if arm == "raw":
        if fuzzy_view is None:
            raise ValueError("raw compaction requires --fuzzy-view-dir")
        fuzzy_path = fuzzy_view / f"train_{bucket:05d}.parquet"
        if not fuzzy_path.is_file():
            raise FileNotFoundError(fuzzy_path)
        fuzzy_table = pq.read_table(fuzzy_path, columns=[
            "subset", "doc_id", "text", "nanochat_token_count", "shuffle_hash"
        ])
        fuzzy_rows = {
            key: (text, int(tokens), digest)
            for key, text, tokens, digest in zip(
                _table_keys(fuzzy_table),
                fuzzy_table["text"].to_pylist(),
                fuzzy_table["nanochat_token_count"].to_pylist(),
                fuzzy_table["shuffle_hash"].to_pylist(),
                strict=True,
            )
        }
        raw_keys = _table_keys(table)
        raw_texts = table["text"].to_pylist()
        raw_tokens = table["nanochat_token_count"].to_pylist()
        raw_hashes = table["shuffle_hash"].to_pylist()
        flags = []
        matched = 0
        for key, text, tokens, digest in zip(raw_keys, raw_texts, raw_tokens, raw_hashes, strict=True):
            fuzzy = fuzzy_rows.get(key)
            removed = fuzzy is None
            flags.append(removed)
            if removed:
                removed_docs += 1
                removed_tokens += int(tokens)
            else:
                matched += 1
                if fuzzy != (text, int(tokens), digest):
                    raise RuntimeError(f"raw/fuzzy retained row mismatch in bucket {bucket}: {key}")
        if matched != len(fuzzy_rows):
            raise RuntimeError(
                f"bucket {bucket}: matched {matched:,} fuzzy rows, expected {len(fuzzy_rows):,}"
            )
        table = table.append_column("removed_by_fuzzy", pa.array(flags, type=pa.bool_()))

    output_path = output / f"train_{bucket:05d}.parquet"
    pq.write_table(table, output_path, compression="zstd", row_group_size=1024)
    subset_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"docs": 0, "tokens": 0})
    length_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"docs": 0, "tokens": 0})
    for subset, tokens in zip(table["subset"].to_pylist(), table["nanochat_token_count"].to_pylist(), strict=True):
        subset_stats[subset]["docs"] += 1
        subset_stats[subset]["tokens"] += int(tokens)
        label = length_bin(int(tokens))
        length_stats[label]["docs"] += 1
        length_stats[label]["tokens"] += int(tokens)
    return {
        "file": output_path.name,
        "sha256": sha256_file(output_path),
        "docs": table.num_rows,
        "tokens": int(pc.sum(table["nanochat_token_count"]).as_py()),
        "removed_docs": removed_docs,
        "removed_tokens": removed_tokens,
        "min_hash": table["shuffle_hash"][0].as_py().hex(),
        "max_hash": table["shuffle_hash"][-1].as_py().hex(),
        "subset_stats": dict(subset_stats),
        "length_stats": dict(length_stats),
    }


def merge_nested(rows: Iterable[dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = defaultdict(lambda: {"docs": 0, "tokens": 0})
    for row in rows:
        for key, values in row.items():
            merged[key]["docs"] += int(values["docs"])
            merged[key]["tokens"] += int(values["tokens"])
    return dict(sorted(merged.items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    validation = args.validation_file.expanduser().resolve()
    fuzzy_view = args.fuzzy_view_dir.expanduser().resolve() if args.fuzzy_view_dir else None
    manifest_path = output / "view_manifest.json"
    config = {
        "arm": args.arm,
        "source_root": str(source_root),
        "validation_file": str(validation),
        "shuffle_seed": args.shuffle_seed,
        "hash_start": args.hash_start,
        "hash_end": args.hash_end,
        "num_buckets": args.num_buckets,
    }
    if manifest_path.is_file() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("config") == config:
            print(f"View already complete: {manifest_path}")
            return existing
        raise FileExistsError(f"Output exists with a different/incomplete config: {output}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        if output == source_root or output == Path("/"):
            raise ValueError(f"Refusing unsafe overwrite target: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    staging = output / "_staging"
    staging.mkdir()
    started = utc_now()
    atomic_json(manifest_path, {"format_version": FORMAT_VERSION, "status": "running", "config": config, "started_at": started})

    source_manifest_candidates = [
        source_root / "materialize_manifest.json",
        source_root.parent / "work" / "exact_fuzzy_manifest.json",
    ]
    source_manifest = next((path for path in source_manifest_candidates if path.is_file()), None)
    lower, upper = hash_bounds(args.hash_start, args.hash_end)
    val_hashes, val_docs = validation_hashes(validation)
    files = source_files(source_root)
    workers = min(max(1, int(getattr(args, "workers", 1))), len(files))
    configure_scan_context(
        lower=lower,
        upper=upper,
        num_buckets=args.num_buckets,
        shuffle_seed=args.shuffle_seed,
        validation_hashes_set=val_hashes,
        staging=staging,
        buffer_rows=args.buffer_rows,
    )
    indexed_files = [(index, str(path)) for index, path in enumerate(files, start=1)]
    partitions = [indexed_files[worker::workers] for worker in range(workers)]
    tasks = [(worker, partition) for worker, partition in enumerate(partitions) if partition]
    if workers == 1:
        scan_rows = [scan_partition(tasks[0])]
    else:
        # fork inherits the validation hash set without serializing it once per worker.
        with mp.get_context("fork").Pool(processes=workers) as pool:
            scan_rows = pool.map(scan_partition, tasks)

    scanned_docs = sum(row["scanned_docs"] for row in scan_rows)
    selected_docs = sum(row["selected_docs"] for row in scan_rows)
    selected_tokens = sum(row["selected_tokens"] for row in scan_rows)
    excluded_val_docs = sum(row["excluded_val_docs"] for row in scan_rows)
    excluded_val_tokens = sum(row["excluded_val_tokens"] for row in scan_rows)
    print(
        f"scan complete workers={workers} files={len(files)} docs={scanned_docs:,} "
        f"selected={selected_docs:,} tokens={selected_tokens:,}",
        flush=True,
    )

    bucket_rows = [compact_bucket(staging, output, bucket, args.arm, fuzzy_view) for bucket in range(args.num_buckets)]
    total_docs = sum(row["docs"] for row in bucket_rows)
    total_tokens = sum(row["tokens"] for row in bucket_rows)
    removed_docs = sum(row["removed_docs"] for row in bucket_rows)
    removed_tokens = sum(row["removed_tokens"] for row in bucket_rows)
    if total_docs != selected_docs or total_tokens != selected_tokens:
        raise RuntimeError("compacted totals do not match staged totals")
    if args.min_source_tokens and total_tokens < args.min_source_tokens:
        raise RuntimeError(f"view has {total_tokens:,} tokens, below required {args.min_source_tokens:,}")
    removed_fraction = removed_tokens / total_tokens if total_tokens else 0.0
    if args.arm == "raw" and not args.skip_removal_fraction_gate:
        expected = args.expected_removed_token_fraction
        if abs(removed_fraction - expected) > args.removed_token_fraction_tolerance:
            raise RuntimeError(
                f"raw removed-token fraction {removed_fraction:.6f} is outside "
                f"{expected:.6f} +/- {args.removed_token_fraction_tolerance:.6f}"
            )

    payload = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "config": config,
        "started_at": started,
        "completed_at": utc_now(),
        "source_manifest": str(source_manifest) if source_manifest else None,
        "source_manifest_sha256": sha256_file(source_manifest) if source_manifest else None,
        "validation": {"file": str(validation), "sha256": sha256_file(validation), "docs": val_docs},
        "scan": {"files": len(files), "docs": scanned_docs, "workers": workers},
        "selection": {
            "docs": total_docs,
            "tokens": total_tokens,
            "excluded_validation_overlap_docs": excluded_val_docs,
            "excluded_validation_overlap_tokens": excluded_val_tokens,
            "removed_docs": removed_docs,
            "removed_tokens": removed_tokens,
            "removed_doc_fraction": removed_docs / total_docs if total_docs else 0.0,
            "removed_token_fraction": removed_fraction,
        },
        "by_subset": merge_nested(row["subset_stats"] for row in bucket_rows),
        "by_length": merge_nested(row["length_stats"] for row in bucket_rows),
        "files": [{key: value for key, value in row.items() if key not in {"subset_stats", "length_stats"}} for row in bucket_rows],
    }
    atomic_json(manifest_path, payload)
    shutil.rmtree(staging)
    print(f"Completed {args.arm} view: docs={total_docs:,} tokens={total_tokens:,} path={output}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("fuzzy", "raw"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--fuzzy-view-dir", type=Path)
    parser.add_argument("--shuffle-seed", type=int, default=20260722)
    parser.add_argument("--hash-start", type=float, default=0.0)
    parser.add_argument("--hash-end", type=float, default=0.08)
    parser.add_argument("--num-buckets", type=int, default=512)
    parser.add_argument("--buffer-rows", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--min-source-tokens", type=int, default=0)
    parser.add_argument("--expected-removed-token-fraction", type=float, default=0.44475)
    parser.add_argument("--removed-token-fraction-tolerance", type=float, default=0.01)
    parser.add_argument("--skip-removal-fraction-gate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_buckets <= 0 or args.buffer_rows <= 0 or args.workers <= 0:
        parser.error("--num-buckets, --buffer-rows, and --workers must be positive")
    if args.arm == "raw" and args.fuzzy_view_dir is None:
        parser.error("--arm=raw requires --fuzzy-view-dir")
    return args


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
