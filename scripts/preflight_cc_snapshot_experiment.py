"""Preflight one or more paired Fortified fuzzy-dedup training views."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pyarrow.parquet as pq

from scripts.experiment_fingerprints import training_code_sha256


EXPECTED_CONDITIONS = ("cc_2013_20", "cc_2023_14", "cc_2023_50", "mixed")
EXPECTED_STEPS = 6_612
TOTAL_BATCH_SIZE = 1_048_576
TRAINING_TOKENS = EXPECTED_STEPS * TOTAL_BATCH_SIZE
TARGET_CAPACITY = TRAINING_TOKENS + TOTAL_BATCH_SIZE


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


def contamination_keys(manifest_path: Path) -> set[tuple[str, str]]:
    manifest = load_json(manifest_path)
    keys_path = Path(manifest["excluded_keys_file"])
    if not keys_path.is_absolute():
        keys_path = manifest_path.parent / keys_path
    keys = set()
    with keys_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                keys.add((str(row["subset"]), str(row["doc_id"])))
    if len(keys) != int(manifest["excluded_training_docs"]):
        raise RuntimeError(f"Contamination key count mismatch: {manifest_path}")
    return keys


def assert_view_has_no_keys(view: Path, excluded: set[tuple[str, str]]) -> int:
    docs = 0
    for path in sorted(view.glob("train_*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=100_000, columns=["subset", "doc_id"]):
            values = batch.to_pydict()
            for key in zip(values["subset"], values["doc_id"], strict=True):
                if key in excluded:
                    raise RuntimeError(f"Contaminated key remains in {view}: {key}")
            docs += batch.num_rows
    return docs


def parse_condition(value: str) -> dict[str, Any]:
    parts = value.split("|", 5)
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "condition must be NAME|SUBSET_OR_-|RAW_VIEW|FUZZY_VIEW|DEDUP_MANIFEST|CONTAMINATION_MANIFEST"
        )
    name, subset, raw, fuzzy, dedup, contamination = parts
    return {
        "name": name,
        "subset": None if subset == "-" else subset,
        "raw_view": Path(raw).expanduser().resolve(),
        "fuzzy_view": Path(fuzzy).expanduser().resolve(),
        "dedup_manifest": Path(dedup).expanduser().resolve(),
        "contamination_manifest": Path(contamination).expanduser().resolve(),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    steps = int(getattr(args, "num_iterations", EXPECTED_STEPS))
    total_batch_size = int(getattr(args, "total_batch_size", TOTAL_BATCH_SIZE))
    world_size = int(getattr(args, "world_size", 8))
    sequence_length = int(getattr(args, "sequence_length", 2048))
    if min(steps, total_batch_size, world_size, sequence_length) <= 0:
        raise ValueError("training horizon, batch size, world size, and sequence length must be positive")
    training_tokens = steps * total_batch_size
    calculated_capacity = training_tokens + total_batch_size
    requested_capacity = int(getattr(args, "target_token_capacity", -1))
    target_capacity = calculated_capacity if requested_capacity < 0 else requested_capacity
    if target_capacity != calculated_capacity:
        raise ValueError(
            f"target token capacity must equal training tokens plus one global batch: "
            f"expected {calculated_capacity:,}, got {target_capacity:,}"
        )
    selection_seed = int(getattr(args, "selection_seed", 20260722))
    order_seed = int(getattr(args, "order_seed", 20260723))
    expected_hash_start = getattr(args, "expected_hash_start", None)
    expected_hash_end = getattr(args, "expected_hash_end", None)
    validation = args.validation_file.expanduser().resolve()
    tokenizer = args.tokenizer_file.expanduser().resolve()
    expected_validation_sha = args.validation_sha256 or sha256_file(validation)
    if sha256_file(validation) != expected_validation_sha:
        raise RuntimeError("Frozen validation checksum changed")
    if not tokenizer.is_file():
        raise FileNotFoundError(tokenizer)
    by_name = {condition["name"]: condition for condition in args.condition}
    configured_conditions = tuple(getattr(args, "expected_condition", None) or EXPECTED_CONDITIONS)
    if set(by_name) != set(configured_conditions):
        raise RuntimeError(f"Expected conditions {configured_conditions}, got {sorted(by_name)}")

    rows = []
    manifest_checksums = {}
    eval_manifest_checksums = set()
    core_config_checksums = set()
    for name in configured_conditions:
        condition = by_name[name]
        dedup = load_json(condition["dedup_manifest"])
        if dedup.get("status") != "complete" or any(
            dedup.get("stages", {}).get(stage, {}).get("status") != "complete"
            for stage in ("exact", "fuzzy", "stats", "report")
        ):
            raise RuntimeError(f"{name}: dedup manifest is incomplete")
        dedup_config = dedup.get("config", {})
        for key, expected in (("seed", 42), ("char_ngrams", 24), ("num_bands", 20), ("minhashes_per_band", 13)):
            if int(dedup_config.get(key, -1)) != expected:
                raise RuntimeError(f"{name}: dedup {key} mismatch")
        if condition["subset"]:
            dedup_input = Path(dedup["input_data_dir"])
            source_manifest = load_json(dedup_input / "materialize_manifest.json")
            if source_manifest.get("subsets") != [condition["subset"]]:
                raise RuntimeError(
                    f"{name}: dedup input is not isolated to {condition['subset']}"
                )
        contamination = load_json(condition["contamination_manifest"])
        if contamination.get("status") != "complete":
            raise RuntimeError(f"{name}: contamination manifest incomplete")
        contamination_config = contamination.get("config", {})
        for key, expected in (("seed", 42), ("char_ngrams", 24), ("num_bands", 20), ("minhashes_per_band", 13), ("minhashes", 260)):
            if int(contamination_config.get(key, -1)) != expected:
                raise RuntimeError(f"{name}: contamination {key} mismatch")
        eval_manifest_checksums.add(contamination.get("eval_manifest_sha256"))
        core_config_checksums.add(contamination.get("core_config_sha256"))
        if contamination.get("validation_sha256") != expected_validation_sha:
            raise RuntimeError(f"{name}: contamination validation checksum mismatch")
        if contamination.get("post_exclusion_expected_exact_overlap") != 0 or contamination.get("post_exclusion_expected_fuzzy_overlap") != 0:
            raise RuntimeError(f"{name}: contamination manifest does not certify zero expected overlap")
        excluded = contamination_keys(condition["contamination_manifest"])

        arm_manifests = {}
        for arm in ("raw", "fuzzy"):
            view = condition[f"{arm}_view"]
            manifest_path = view / "view_manifest.json"
            manifest = load_json(manifest_path)
            arm_manifests[arm] = manifest
            manifest_checksums[f"{name}/{arm}"] = sha256_file(manifest_path)
            if (
                manifest.get("format_version") != 2
                or manifest.get("status") != "complete"
                or manifest.get("config", {}).get("arm") != arm
            ):
                raise RuntimeError(f"{name}/{arm}: view is incomplete, outdated, or mislabeled")
            config = manifest["config"]
            expected_source = dedup["input_data_dir"] if arm == "raw" else dedup["output_data_dir"]
            if Path(config["source_root"]).resolve() != Path(expected_source).resolve():
                raise RuntimeError(f"{name}/{arm}: view source does not match dedup manifest")
            if arm == "fuzzy" and manifest.get("source_manifest_sha256") != sha256_file(
                condition["dedup_manifest"]
            ):
                raise RuntimeError(f"{name}/fuzzy: source dedup checksum mismatch")
            if int(config.get("selection_seed", -1)) != selection_seed or int(config.get("order_seed", -1)) != order_seed:
                raise RuntimeError(f"{name}/{arm}: selection/order seed mismatch")
            if int(config.get("target_token_capacity", -1)) != target_capacity:
                raise RuntimeError(f"{name}/{arm}: target capacity config mismatch")
            if expected_hash_start is not None and float(config.get("hash_start", -1)) != float(expected_hash_start):
                raise RuntimeError(f"{name}/{arm}: hash start mismatch")
            if expected_hash_end is not None and float(config.get("hash_end", -1)) != float(expected_hash_end):
                raise RuntimeError(f"{name}/{arm}: hash end mismatch")
            if manifest.get("validation", {}).get("sha256") != expected_validation_sha:
                raise RuntimeError(f"{name}/{arm}: validation checksum mismatch")
            if manifest.get("contamination", {}).get("manifest_sha256") != sha256_file(condition["contamination_manifest"]):
                raise RuntimeError(f"{name}/{arm}: contamination checksum mismatch")
            expected_subsets = [] if condition["subset"] is None else [condition["subset"]]
            if config.get("subsets") != expected_subsets:
                raise RuntimeError(f"{name}/{arm}: subset filter mismatch")
            if condition["subset"] and set(manifest.get("by_subset", {})) != {condition["subset"]}:
                raise RuntimeError(f"{name}/{arm}: view contains another CC snapshot")
            capacity = manifest.get("capacity", {})
            if int(capacity.get("world_size", -1)) != world_size or int(capacity.get("sequence_length", -1)) != sequence_length:
                raise RuntimeError(f"{name}/{arm}: capacity topology mismatch")
            required_per_rank = (target_capacity + world_size - 1) // world_size
            per_rank = capacity.get("per_rank", [])
            if len(per_rank) != world_size or {int(row["rank"]) for row in per_rank} != set(range(world_size)):
                raise RuntimeError(f"{name}/{arm}: capacity does not enumerate all ranks")
            for rank in per_rank:
                if int(rank["consumable_target_tokens"]) < required_per_rank:
                    raise RuntimeError(f"{name}/{arm}: rank {rank['rank']} would roll over")
            scanned_docs = None if args.skip_key_scan else assert_view_has_no_keys(view, excluded)
            if scanned_docs is not None and scanned_docs != int(manifest["selection"]["docs"]):
                raise RuntimeError(f"{name}/{arm}: parquet doc count differs from manifest")
            required_columns = {"text", "subset", "doc_id", "nanochat_token_count", "shuffle_hash"}
            if arm == "raw":
                required_columns.add("removed_by_fuzzy")
            file_rows = manifest.get("files", [])
            if not file_rows:
                raise RuntimeError(f"{name}/{arm}: view contains no parquet shards")
            for file_row in file_rows:
                file_path = view / file_row["file"]
                if sha256_file(file_path) != file_row["sha256"]:
                    raise RuntimeError(f"{name}/{arm}: shard checksum mismatch: {file_path}")
                missing_columns = required_columns - set(pq.ParquetFile(file_path).schema_arrow.names)
                if missing_columns:
                    raise RuntimeError(f"{name}/{arm}: shard is missing columns {sorted(missing_columns)}")
            rows.append({
                "condition": name,
                "subset": condition["subset"],
                "arm": arm,
                "docs": int(manifest["selection"]["docs"]),
                "source_tokens": int(manifest["selection"]["tokens"]),
                "min_rank_target_capacity": int(capacity["min_rank_consumable_target_tokens"]),
                "contamination_excluded_docs": int(manifest["selection"]["excluded_contamination_docs"]),
                "validation_exact_excluded_docs": int(manifest["selection"]["excluded_validation_overlap_docs"]),
                "expected_epoch": 1,
                "expected_discarded_source_tokens": 0,
            })
        if arm_manifests["raw"]["validation"]["sha256"] != arm_manifests["fuzzy"]["validation"]["sha256"]:
            raise RuntimeError(f"{name}: paired validation mismatch")

    if len(eval_manifest_checksums) != 1 or None in eval_manifest_checksums:
        raise RuntimeError("Conditions do not share one frozen validation/CORE contamination corpus")
    if len(core_config_checksums) != 1 or None in core_config_checksums:
        raise RuntimeError("Conditions do not share one frozen CORE configuration")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    payload = {
        "format_version": 1,
        "status": "passed",
        "checked_at": utc_now(),
        "single_seed_exploratory": True,
        "model_seed": 42,
        "steps": steps,
        "training_tokens": training_tokens,
        "target_capacity": target_capacity,
        "world_size": world_size,
        "sequence_length": sequence_length,
        "selection_seed": selection_seed,
        "order_seed": order_seed,
        "expected_conditions": list(configured_conditions),
        "expected_hash_start": expected_hash_start,
        "expected_hash_end": expected_hash_end,
        "validation": {"file": str(validation), "sha256": expected_validation_sha},
        "tokenizer": {"file": str(tokenizer), "sha256": sha256_file(tokenizer)},
        "git_commit": commit or "unknown",
        "code_sha256": training_code_sha256(),
        "eval_manifest_sha256": next(iter(eval_manifest_checksums)),
        "core_config_sha256": next(iter(core_config_checksums)),
        "manifest_checksums": manifest_checksums,
        "views": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", type=parse_condition, required=True)
    parser.add_argument("--expected-condition", action="append")
    parser.add_argument("--num-iterations", type=int, default=EXPECTED_STEPS)
    parser.add_argument("--total-batch-size", type=int, default=TOTAL_BATCH_SIZE)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--target-token-capacity", type=int, default=-1)
    parser.add_argument("--selection-seed", type=int, default=20260722)
    parser.add_argument("--order-seed", type=int, default=20260723)
    parser.add_argument("--expected-hash-start", type=float)
    parser.add_argument("--expected-hash-end", type=float)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--validation-sha256")
    parser.add_argument("--tokenizer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-key-scan", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = preflight(parse_args())
    print(json.dumps({"status": result["status"], "views": len(result["views"])}, indent=2))


if __name__ == "__main__":
    main()
