import argparse
import hashlib
import json
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.preflight_cc_snapshot_experiment import TARGET_CAPACITY, preflight


SNAPSHOTS = {
    "cc_2013_20": "CC-MAIN-2013-20",
    "cc_2023_14": "CC-MAIN-2023-14",
    "cc_2023_50": "CC-MAIN-2023-50",
    "mixed": None,
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_args(tmp_path: Path) -> argparse.Namespace:
    validation = tmp_path / "validation.parquet"
    tokenizer = tmp_path / "tokenizer.pkl"
    validation.write_bytes(b"validation")
    tokenizer.write_bytes(b"tokenizer")
    conditions = []
    for name, subset in SNAPSHOTS.items():
        root = tmp_path / name
        raw_source = root / "source_raw"
        fuzzy_source = root / "source_fuzzy"
        raw_source.mkdir(parents=True)
        fuzzy_source.mkdir(parents=True)
        if subset:
            write_json(
                raw_source / "materialize_manifest.json",
                {"status": "complete", "subsets": [subset]},
            )
        dedup = root / "dedup.json"
        write_json(
            dedup,
            {
                "status": "complete",
                "input_data_dir": str(raw_source),
                "output_data_dir": str(fuzzy_source),
                "config": {
                    "seed": 42,
                    "char_ngrams": 24,
                    "num_bands": 20,
                    "minhashes_per_band": 13,
                },
                "stages": {
                    stage: {"status": "complete"}
                    for stage in ("exact", "fuzzy", "stats", "report")
                },
            },
        )
        contamination_dir = root / "contamination"
        keys = contamination_dir / "keys.jsonl"
        keys.parent.mkdir(parents=True)
        keys.write_text("", encoding="utf-8")
        contamination = contamination_dir / "manifest.json"
        write_json(
            contamination,
            {
                "status": "complete",
                "config": {
                    "seed": 42,
                    "char_ngrams": 24,
                    "num_bands": 20,
                    "minhashes_per_band": 13,
                    "minhashes": 260,
                },
                "eval_manifest_sha256": "eval",
                "core_config_sha256": "core",
                "validation_sha256": sha(validation),
                "excluded_keys_file": keys.name,
                "excluded_training_docs": 0,
                "post_exclusion_expected_exact_overlap": 0,
                "post_exclusion_expected_fuzzy_overlap": 0,
            },
        )
        views = {}
        for arm, source in (("raw", raw_source), ("fuzzy", fuzzy_source)):
            view = root / arm
            views[arm] = view
            shard = view / "train_00000.parquet"
            columns = {
                "text": ["text"],
                "subset": [subset or "mixed"],
                "doc_id": ["doc"],
                "nanochat_token_count": [TARGET_CAPACITY],
                "shuffle_hash": [b"0" * 16],
            }
            if arm == "raw":
                columns["removed_by_fuzzy"] = [False]
            shard.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(columns), shard)
            write_json(
                view / "view_manifest.json",
                {
                    "format_version": 2,
                    "status": "complete",
                    "config": {
                        "arm": arm,
                        "source_root": str(source),
                        "subsets": [] if subset is None else [subset],
                        "selection_seed": 20260722,
                        "order_seed": 20260723,
                        "target_token_capacity": TARGET_CAPACITY,
                    },
                    "source_manifest_sha256": sha(dedup) if arm == "fuzzy" else None,
                    "validation": {"sha256": sha(validation)},
                    "contamination": {"manifest_sha256": sha(contamination)},
                    "capacity": {
                        "world_size": 8,
                        "sequence_length": 2048,
                        "min_rank_consumable_target_tokens": TARGET_CAPACITY // 8,
                        "per_rank": [
                            {
                                "rank": rank,
                                "consumable_target_tokens": TARGET_CAPACITY // 8,
                            }
                            for rank in range(8)
                        ],
                    },
                    "selection": {
                        "docs": 1,
                        "tokens": TARGET_CAPACITY,
                        "excluded_contamination_docs": 0,
                        "excluded_validation_overlap_docs": 0,
                    },
                    "by_subset": {} if subset is None else {subset: {"docs": 1}},
                    "files": [{"file": shard.name, "sha256": sha(shard)}],
                },
            )
        conditions.append(
            {
                "name": name,
                "subset": subset,
                "raw_view": views["raw"],
                "fuzzy_view": views["fuzzy"],
                "dedup_manifest": dedup,
                "contamination_manifest": contamination,
            }
        )
    return argparse.Namespace(
        condition=conditions,
        validation_file=validation,
        validation_sha256=sha(validation),
        tokenizer_file=tokenizer,
        output=tmp_path / "preflight.json",
        skip_key_scan=True,
    )


def test_preflight_certifies_all_eight_views(tmp_path: Path) -> None:
    args = fixture_args(tmp_path)
    result = preflight(args)
    assert result["status"] == "passed"
    assert len(result["views"]) == 8
    assert result["code_sha256"]


def test_preflight_rejects_cross_snapshot_dedup_input(tmp_path: Path) -> None:
    args = fixture_args(tmp_path)
    condition = next(row for row in args.condition if row["name"] == "cc_2023_14")
    dedup = json.loads(condition["dedup_manifest"].read_text(encoding="utf-8"))
    input_manifest = Path(dedup["input_data_dir"]) / "materialize_manifest.json"
    write_json(
        input_manifest,
        {"status": "complete", "subsets": ["CC-MAIN-2023-14", "CC-MAIN-2023-50"]},
    )
    with pytest.raises(RuntimeError, match="dedup input is not isolated"):
        preflight(args)


def test_preflight_supports_parameterized_single_4x_condition(tmp_path: Path) -> None:
    args = fixture_args(tmp_path)
    condition = next(row for row in args.condition if row["name"] == "mixed")
    condition["name"] = "cw_4x"
    args.condition = [condition]
    args.expected_condition = ["cw_4x"]
    args.num_iterations = 26_448
    args.total_batch_size = 1_048_576
    args.world_size = 8
    args.sequence_length = 2_048
    args.target_token_capacity = args.num_iterations * args.total_batch_size + args.total_batch_size
    for arm in ("raw", "fuzzy"):
        manifest_path = condition[f"{arm}_view"] / "view_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["target_token_capacity"] = args.target_token_capacity
        manifest["capacity"]["min_rank_consumable_target_tokens"] = args.target_token_capacity // 8
        for rank in manifest["capacity"]["per_rank"]:
            rank["consumable_target_tokens"] = args.target_token_capacity // 8
        shard = condition[f"{arm}_view"] / "train_00000.parquet"
        table = pq.read_table(shard)
        columns = table.to_pydict()
        columns["nanochat_token_count"] = [args.target_token_capacity]
        pq.write_table(pa.table(columns), shard)
        manifest["files"][0]["sha256"] = sha(shard)
        manifest_path.write_text(json.dumps(manifest))
    result = preflight(args)
    assert result["steps"] == 26_448
    assert result["training_tokens"] == 27_732_738_048
    assert result["target_capacity"] == 27_733_786_624
    assert result["expected_conditions"] == ["cw_4x"]


def test_preflight_rejects_capacity_not_equal_to_horizon_plus_batch(tmp_path: Path) -> None:
    args = fixture_args(tmp_path)
    args.target_token_capacity = TARGET_CAPACITY + 1
    with pytest.raises(ValueError, match="one global batch"):
        preflight(args)
