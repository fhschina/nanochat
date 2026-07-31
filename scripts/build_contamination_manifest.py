"""Build component-level validation/CORE exclusion keys from Curator artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import unicodedata
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from scripts.audit_fuzzy_components import _source_file_ranges


CURATOR_ID = "_curator_dedup_id"
COMPONENT_ID = "_duplicate_group_id"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalized_digest(text: str) -> bytes:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).digest()


def parquet_files(path: Path) -> list[Path]:
    files = sorted(item for item in path.rglob("*.parquet") if not item.name.endswith(".tmp"))
    if not files:
        raise FileNotFoundError(f"No parquet files below {path}")
    return files


def ids_in_intervals(values: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(len(values), dtype=np.bool_)
    for start, end in intervals:
        mask |= (values >= start) & (values < end)
    return mask


def ids_in_sorted(values: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    if not len(wanted):
        return np.zeros(len(values), dtype=np.bool_)
    positions = np.searchsorted(wanted, values)
    mask = positions < len(wanted)
    mask[mask] &= wanted[positions[mask]] == values[mask]
    return mask


def groups_in_sorted(values: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    return ids_in_sorted(values, wanted)


def identify_tainted_training_ids(
    component_files: Iterable[Path],
    eval_intervals: list[tuple[int, int]],
    exact_training_ids: np.ndarray,
) -> tuple[set[int], dict[int, int], set[int]]:
    """Return all training vertices in any component touched by eval or exact."""
    tainted_groups: set[int] = set()
    files = list(component_files)
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1_000_000, columns=[CURATOR_ID, COMPONENT_ID]):
            ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            groups = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            seeds = ids_in_intervals(ids, eval_intervals) | ids_in_sorted(ids, exact_training_ids)
            tainted_groups.update(int(value) for value in groups[seeds])
    sorted_groups = np.asarray(sorted(tainted_groups), dtype=np.int64)
    selected = set(int(value) for value in exact_training_ids)
    group_by_id: dict[int, int] = {}
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1_000_000, columns=[CURATOR_ID, COMPONENT_ID]):
            ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            groups = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            mask = groups_in_sorted(groups, sorted_groups)
            mask &= ~ids_in_intervals(ids, eval_intervals)
            for curator_id, component_id in zip(ids[mask].tolist(), groups[mask].tolist(), strict=True):
                selected.add(int(curator_id))
                group_by_id[int(curator_id)] = int(component_id)
    return selected, group_by_id, tainted_groups


def _range_kind(record: dict[str, Any], union_root: Path) -> str:
    relative = record["source"].resolve().relative_to(union_root)
    return relative.parts[0]


def _evaluation_hashes(ranges: list[dict[str, Any]], union_root: Path) -> tuple[set[bytes], list[tuple[int, int]], int]:
    hashes: set[bytes] = set()
    intervals = []
    docs = 0
    for record in ranges:
        if _range_kind(record, union_root) != "eval":
            continue
        intervals.append((int(record["start"]), int(record["end"])))
        parquet = pq.ParquetFile(record["source"])
        for batch in parquet.iter_batches(batch_size=8192, columns=["text"]):
            texts = batch.column(0).to_pylist()
            hashes.update(normalized_digest(str(text)) for text in texts)
            docs += len(texts)
    if not intervals:
        raise RuntimeError("Contamination union contains no eval-prefixed files")
    return hashes, intervals, docs


def _exact_training_ids(
    ranges: list[dict[str, Any]], union_root: Path, eval_hashes: set[bytes]
) -> np.ndarray:
    ids = []
    for record in ranges:
        if _range_kind(record, union_root) != "training":
            continue
        parquet = pq.ParquetFile(record["source"])
        offset = 0
        for batch in parquet.iter_batches(batch_size=8192, columns=["text"]):
            for index, text in enumerate(batch.column(0).to_pylist()):
                if normalized_digest(str(text)) in eval_hashes:
                    ids.append(int(record["start"]) + offset + index)
            offset += batch.num_rows
        if offset != int(record["end"]) - int(record["start"]):
            raise RuntimeError(f"Curator range mismatch for {record['source']}")
    return np.asarray(sorted(set(ids)), dtype=np.int64)


def _extract_training_keys(
    selected_ids: set[int],
    group_by_id: dict[int, int],
    exact_ids: set[int],
    ranges: list[dict[str, Any]],
    union_root: Path,
) -> list[dict[str, Any]]:
    wanted = np.asarray(sorted(selected_ids), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for record in ranges:
        if _range_kind(record, union_root) != "training":
            continue
        start, end = int(record["start"]), int(record["end"])
        left = int(np.searchsorted(wanted, start, side="left"))
        right = int(np.searchsorted(wanted, end, side="left"))
        if left == right:
            continue
        local_wanted = wanted[left:right] - start
        parquet = pq.ParquetFile(record["source"])
        row_offset = 0
        for batch in parquet.iter_batches(
            batch_size=8192,
            columns=["subset", "doc_id", "nanochat_token_count"],
        ):
            batch_end = row_offset + batch.num_rows
            first = int(np.searchsorted(local_wanted, row_offset, side="left"))
            last = int(np.searchsorted(local_wanted, batch_end, side="left"))
            if first != last:
                values = batch.to_pydict()
                for local_row in local_wanted[first:last].tolist():
                    index = int(local_row) - row_offset
                    curator_id = start + int(local_row)
                    reasons = []
                    if curator_id in exact_ids:
                        reasons.append("normalized_exact")
                    if curator_id in group_by_id:
                        reasons.append("fuzzy_component")
                    rows.append({
                        "subset": str(values["subset"][index]),
                        "doc_id": str(values["doc_id"][index]),
                        "nanochat_token_count": int(values["nanochat_token_count"][index]),
                        "curator_id": curator_id,
                        "component_id": group_by_id.get(curator_id),
                        "reasons": reasons,
                    })
            row_offset = batch_end
    if len(rows) != len(selected_ids):
        raise RuntimeError(f"Resolved {len(rows):,}/{len(selected_ids):,} selected training IDs")
    rows.sort(key=lambda row: (row["subset"], row["doc_id"]))
    keys = [(row["subset"], row["doc_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Contamination output contains duplicate training document keys")
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    union_root = args.union_data_dir.expanduser().resolve()
    components = args.component_dir.expanduser().resolve()
    id_generator = args.id_generator_path.expanduser().resolve()
    artifacts_manifest = args.fuzzy_artifacts_manifest.expanduser().resolve()
    eval_manifest = args.eval_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "contamination_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
        raise FileExistsError(manifest_path)

    artifacts = json.loads(artifacts_manifest.read_text(encoding="utf-8"))
    fuzzy_config = artifacts.get("config", {})
    expected = {"seed": 42, "char_ngrams": 24, "num_bands": 20, "minhashes_per_band": 13}
    for key, value in expected.items():
        if int(fuzzy_config.get(key, -1)) != value:
            raise RuntimeError(f"Fuzzy artifact {key}={fuzzy_config.get(key)!r}, expected {value}")
    ranges = _source_file_ranges(union_root, id_generator, args.input_blocksize)
    eval_hashes, eval_intervals, eval_docs = _evaluation_hashes(ranges, union_root)
    exact_array = _exact_training_ids(ranges, union_root, eval_hashes)
    component_files = parquet_files(components)
    selected_ids, group_by_id, tainted_groups = identify_tainted_training_ids(
        component_files, eval_intervals, exact_array
    )
    rows = _extract_training_keys(
        selected_ids, group_by_id, set(int(value) for value in exact_array), ranges, union_root
    )
    if args.expected_subset:
        unexpected = sorted({row["subset"] for row in rows if row["subset"] != args.expected_subset})
        if unexpected:
            raise RuntimeError(f"Contamination keys escaped subset {args.expected_subset}: {unexpected}")

    keys_path = output / "excluded_keys.jsonl"
    temporary = keys_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, keys_path)
    eval_payload = json.loads(eval_manifest.read_text(encoding="utf-8"))
    payload = {
        "format_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "config": {
            **expected,
            "minhashes": 260,
            "input_blocksize": args.input_blocksize,
            "normalization": "Unicode NFKC + casefold + whitespace collapse",
            "keeper_policy": "exclude every training vertex in an eval-tainted connected component",
            "expected_subset": args.expected_subset,
        },
        "union_data_dir": str(union_root),
        "fuzzy_artifacts_manifest": str(artifacts_manifest),
        "fuzzy_artifacts_manifest_sha256": sha256_file(artifacts_manifest),
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": sha256_file(eval_manifest),
        "validation_sha256": eval_payload.get("validation", {}).get("sha256"),
        "core_config_sha256": eval_payload.get("core", {}).get("config_sha256"),
        "eval_docs": eval_docs,
        "normalized_exact_training_docs": len(exact_array),
        "tainted_components": len(tainted_groups),
        "fuzzy_component_training_docs": len(group_by_id),
        "excluded_training_docs": len(rows),
        "excluded_training_tokens": sum(int(row["nanochat_token_count"]) for row in rows),
        "excluded_keys_file": keys_path.name,
        "excluded_keys_sha256": sha256_file(keys_path),
        "post_exclusion_expected_exact_overlap": 0,
        "post_exclusion_expected_fuzzy_overlap": 0,
    }
    atomic_json(manifest_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-data-dir", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--id-generator-path", type=Path, required=True)
    parser.add_argument("--fuzzy-artifacts-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-blocksize", default="512MiB")
    parser.add_argument("--expected-subset")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = build(parse_args())
    print(json.dumps({
        "excluded_training_docs": result["excluded_training_docs"],
        "tainted_components": result["tainted_components"],
    }, indent=2))


if __name__ == "__main__":
    main()
