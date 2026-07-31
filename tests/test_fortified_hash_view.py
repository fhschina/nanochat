import argparse
import json
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
        selection_seed=20260722,
        order_seed=20260723,
        subset=None,
        contamination_manifest=None,
        hash_start=0.0,
        hash_end=1.0,
        num_buckets=1,
        buffer_rows=2,
        min_source_tokens=0,
        target_token_capacity=0,
        capacity_world_size=1,
        capacity_seq_len=4,
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


def test_subset_contamination_and_independent_order_seed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rows = [
        {"text": "alpha", "subset": "CC-MAIN-2023-14", "doc_id": "a", "nanochat_token_count": 10},
        {"text": "beta", "subset": "CC-MAIN-2023-14", "doc_id": "b", "nanochat_token_count": 10},
        {"text": "gamma", "subset": "CC-MAIN-2023-14", "doc_id": "c", "nanochat_token_count": 10},
        {"text": "other", "subset": "CC-MAIN-2023-50", "doc_id": "d", "nanochat_token_count": 10},
    ]
    write_rows(source / "part.parquet", rows)
    validation = tmp_path / "val.parquet"
    pq.write_table(pa.table({"text": ["not present"]}), validation)
    keys = tmp_path / "excluded_keys.jsonl"
    keys.write_text(json.dumps({"subset": "CC-MAIN-2023-14", "doc_id": "b"}) + "\n")
    contamination = tmp_path / "contamination_manifest.json"
    contamination.write_text(json.dumps({
        "status": "complete",
        "excluded_keys_file": keys.name,
        "excluded_training_docs": 1,
        "config": {"char_ngrams": 24, "num_bands": 20, "minhashes_per_band": 13},
    }))

    first_args = args_for("fuzzy", source, tmp_path / "first", validation)
    first_args.subset = ["CC-MAIN-2023-14"]
    first_args.contamination_manifest = contamination
    first_args.selection_seed = 7
    first_args.order_seed = 8
    second_args = args_for("fuzzy", source, tmp_path / "second", validation)
    second_args.subset = ["CC-MAIN-2023-14"]
    second_args.contamination_manifest = contamination
    second_args.selection_seed = 7
    second_args.order_seed = 9

    first = build(first_args)
    second = build(second_args)
    first_table = pq.read_table(tmp_path / "first" / "train_00000.parquet").to_pydict()
    second_table = pq.read_table(tmp_path / "second" / "train_00000.parquet").to_pydict()
    assert set(first_table["doc_id"]) == set(second_table["doc_id"]) == {"a", "c"}
    assert first_table["shuffle_hash"] != second_table["shuffle_hash"]
    assert first["selection"]["excluded_subset_docs"] == 1
    assert first["selection"]["excluded_contamination_docs"] == 1
    assert first["contamination"]["excluded_docs"] == 1
    assert set(first["by_subset"]) == {"CC-MAIN-2023-14"}


def test_target_capacity_gate_is_per_rank(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rows = [
        {"text": f"text {index}", "subset": "s", "doc_id": str(index), "nanochat_token_count": 5}
        for index in range(2048)
    ]
    write_rows(source / "part.parquet", rows)
    validation = tmp_path / "val.parquet"
    pq.write_table(pa.table({"text": ["not present"]}), validation)
    args = args_for("fuzzy", source, tmp_path / "view", validation)
    args.capacity_world_size = 2
    args.target_token_capacity = 8
    manifest = build(args)
    assert manifest["capacity"]["world_size"] == 2
    assert all(row["consumable_target_tokens"] >= 4 for row in manifest["capacity"]["per_rank"])

    failing = args_for("fuzzy", source, tmp_path / "too_large", validation)
    failing.capacity_world_size = 2
    failing.target_token_capacity = 12_000
    try:
        build(failing)
    except RuntimeError as error:
        assert "without rollover" in str(error)
    else:
        raise AssertionError("capacity gate should reject an undersized rank")
