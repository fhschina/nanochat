import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_fortified_hash_view import build, source_files, stable_hash


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)


def args_for(arm: str, source: Path, output: Path, validation: Path, fuzzy_view: Path | None = None):
    return argparse.Namespace(
        arm=arm,
        source_root=source,
        output_dir=output,
        validation_file=validation,
        fuzzy_view_dir=fuzzy_view,
        shuffle_seed=20260722,
        hash_start=0.0,
        hash_end=1.0,
        num_buckets=1,
        buffer_rows=2,
        min_source_tokens=0,
        expected_removed_token_fraction=0.44475,
        removed_token_fraction_tolerance=0.01,
        skip_removal_fraction_gate=True,
        overwrite=False,
    )


def test_stable_hash_and_source_order_are_deterministic(tmp_path: Path) -> None:
    assert stable_hash("subset", "doc", 7) == stable_hash("subset", "doc", 7)
    assert stable_hash("subset", "doc", 7) != stable_hash("subset", "doc", 8)
    root = tmp_path / "source"
    write_rows(root / "z.parquet", [{"text": "z", "subset": "s", "doc_id": "z", "nanochat_token_count": 2}])
    write_rows(root / "a.parquet", [{"text": "a", "subset": "s", "doc_id": "a", "nanochat_token_count": 2}])
    assert [path.name for path in source_files(root)] == ["a.parquet", "z.parquet"]


def test_fuzzy_is_raw_order_with_removed_rows_deleted_and_val_excluded(tmp_path: Path) -> None:
    raw_root, fuzzy_root = tmp_path / "raw", tmp_path / "fuzzy"
    rows = [
        {"text": "keep alpha", "subset": "s1", "doc_id": "a", "nanochat_token_count": 10},
        {"text": "remove beta", "subset": "s1", "doc_id": "b", "nanochat_token_count": 20},
        {"text": "validation text", "subset": "s2", "doc_id": "c", "nanochat_token_count": 30},
        {"text": "keep delta", "subset": "s2", "doc_id": "d", "nanochat_token_count": 40},
    ]
    write_rows(raw_root / "part.parquet", rows)
    write_rows(fuzzy_root / "part.parquet", [rows[0], rows[2], rows[3]])
    validation = tmp_path / "val.parquet"
    pq.write_table(pa.table({"text": ["validation text"]}), validation)

    fuzzy_view, raw_view = tmp_path / "fuzzy_view", tmp_path / "raw_view"
    fuzzy_manifest = build(args_for("fuzzy", fuzzy_root, fuzzy_view, validation))
    raw_manifest = build(args_for("raw", raw_root, raw_view, validation, fuzzy_view))
    fuzzy = pq.read_table(fuzzy_view / "train_00000.parquet").to_pydict()
    raw = pq.read_table(raw_view / "train_00000.parquet").to_pydict()

    assert fuzzy_manifest["selection"]["excluded_validation_overlap_docs"] == 1
    assert raw_manifest["selection"]["excluded_validation_overlap_docs"] == 1
    retained_raw = [
        (subset, doc_id, text, tokens, digest)
        for subset, doc_id, text, tokens, digest, removed in zip(
            raw["subset"], raw["doc_id"], raw["text"], raw["nanochat_token_count"],
            raw["shuffle_hash"], raw["removed_by_fuzzy"], strict=True,
        )
        if not removed
    ]
    fuzzy_rows = list(zip(
        fuzzy["subset"], fuzzy["doc_id"], fuzzy["text"], fuzzy["nanochat_token_count"],
        fuzzy["shuffle_hash"], strict=True,
    ))
    assert retained_raw == fuzzy_rows
    assert raw_manifest["selection"]["removed_docs"] == 1
    assert raw_manifest["selection"]["removed_tokens"] == 20
    assert "validation text" not in raw["text"]
    assert raw["shuffle_hash"] == sorted(raw["shuffle_hash"])



def test_parallel_scan_is_worker_count_insensitive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rows = [
        {"text": f"text {index}", "subset": f"s{index % 3}", "doc_id": str(index), "nanochat_token_count": index + 1}
        for index in range(24)
    ]
    for file_index in range(4):
        write_rows(source / f"part_{file_index}.parquet", rows[file_index::4])
    validation = tmp_path / "val.parquet"
    pq.write_table(pa.table({"text": ["not present"]}), validation)

    single_args = args_for("fuzzy", source, tmp_path / "single", validation)
    single_args.workers = 1
    parallel_args = args_for("fuzzy", source, tmp_path / "parallel", validation)
    parallel_args.workers = 2
    single = build(single_args)
    parallel = build(parallel_args)

    single_table = pq.read_table(tmp_path / "single" / "train_00000.parquet")
    parallel_table = pq.read_table(tmp_path / "parallel" / "train_00000.parquet")
    assert single_table.equals(parallel_table)
    assert single["files"][0]["sha256"] == parallel["files"][0]["sha256"]
    assert single["selection"] == parallel["selection"]
