"""Run exact then MinHash fuzzy deduplication on a parquet corpus.

NeMo Curator identifies exact and fuzzy duplicate IDs. This driver replays the
saved Curator ID registry against source Parquet files with a bounded, resumable
streaming remover so corpus-scale filtering does not materialize every reader
batch in Ray's object store at once.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


MANIFEST_NAME = "exact_fuzzy_manifest.json"
CURATOR_ID_FIELD = "_curator_dedup_id"
REMOVAL_MANIFEST_NAME = "streaming_removal_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _parquet_files(path: Path) -> list[Path]:
    return sorted(file for file in path.rglob("*.parquet") if not file.name.endswith(".tmp"))


def _source_fields(path: Path) -> list[str]:
    files = _parquet_files(path)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return [
        name
        for name in pq.ParquetFile(files[0]).schema_arrow.names
        if name != CURATOR_ID_FIELD and not name.startswith("__index_level_")
    ]


def _load_tokenizer(tokenizer_dir: Path | None):
    if tokenizer_dir is None:
        return None
    from nanochat.tokenizer import RustBPETokenizer

    resolved = tokenizer_dir.expanduser().resolve()
    if not (resolved / "tokenizer.pkl").is_file():
        raise FileNotFoundError(f"NanoChat tokenizer.pkl not found in {resolved}")
    return RustBPETokenizer.from_directory(str(resolved))


def _token_count(tokenizer, texts: list[str], batch_size: int, num_threads: int) -> int:
    if tokenizer is None:
        return 0
    bos = tokenizer.get_bos_token_id()
    total = 0
    for offset in range(0, len(texts), batch_size):
        encoded = tokenizer.encode(
            texts[offset : offset + batch_size],
            prepend=bos,
            num_threads=num_threads,
        )
        total += sum(len(token_ids) for token_ids in encoded)
    return total


def _sum_column(batch, name: str) -> tuple[int, bool]:
    index = batch.schema.get_field_index(name)
    column = batch.column(index)
    value = pc.sum(column).as_py()
    return int(value or 0), column.null_count < len(column)


def _scan_stats(
    path: Path,
    tokenizer,
    tokenizer_batch_size: int,
    tokenizer_threads: int,
) -> dict:
    files = _parquet_files(path)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")

    docs = 0
    chars = 0
    hf_tokens = 0
    nanochat_tokens = 0
    original_occurrences = 0
    hf_seen = False
    nanochat_seen = False
    count_seen = False

    for file in files:
        parquet = pq.ParquetFile(file)
        names = set(parquet.schema_arrow.names)
        columns = ["text"]
        columns.extend(
            name for name in ("hf_token_count", "nanochat_token_count", "count") if name in names
        )
        for batch in parquet.iter_batches(batch_size=8192, columns=columns):
            batch_docs = batch.num_rows
            text_column = batch.column(batch.schema.get_field_index("text"))
            batch_chars = pc.sum(pc.utf8_length(text_column)).as_py()
            docs += batch_docs
            chars += int(batch_chars or 0)

            if "hf_token_count" in columns:
                value, seen = _sum_column(batch, "hf_token_count")
                hf_tokens += value
                hf_seen = hf_seen or seen
            if "nanochat_token_count" in columns:
                value, seen = _sum_column(batch, "nanochat_token_count")
                nanochat_tokens += value
                nanochat_seen = nanochat_seen or seen
            elif tokenizer is not None:
                texts = [text or "" for text in text_column.to_pylist()]
                nanochat_tokens += _token_count(tokenizer, texts, tokenizer_batch_size, tokenizer_threads)
                nanochat_seen = True
            if "count" in columns:
                value, seen = _sum_column(batch, "count")
                original_occurrences += value
                count_seen = count_seen or seen

    return {
        "docs": docs,
        "chars": chars,
        "hf_tokens": hf_tokens if hf_seen else None,
        "nanochat_tokens": nanochat_tokens if nanochat_seen else None,
        "original_occurrences": original_occurrences if count_seen else None,
        "bytes": sum(file.stat().st_size for file in files),
        "files": len(files),
        "path": str(path),
        "nanochat_tokens_include_bos": nanochat_seen,
        "computed_at": _utc_now(),
    }


def _input_stats_from_materialize_manifest(path: Path) -> dict | None:
    manifest_path = path / "materialize_manifest.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    totals = payload.get("totals")
    if not isinstance(totals, dict) or "docs" not in totals:
        return None
    if payload.get("mode") == "corpus" and payload.get("status") != "complete":
        raise RuntimeError(f"Input materialization is not complete: {manifest_path}")
    return {
        "docs": totals["docs"],
        "chars": totals.get("chars"),
        "hf_tokens": totals.get("hf_tokens"),
        "nanochat_tokens": totals.get("nanochat_tokens"),
        "original_occurrences": totals.get("original_occurrences"),
        "bytes": totals.get("bytes"),
        "files": totals.get("files"),
        "path": str(path),
        "nanochat_tokens_include_bos": bool(payload.get("include_bos")),
        "computed_at": _utc_now(),
        "source": str(manifest_path),
        "dataset": payload.get("dataset"),
        "subsets": payload.get("subsets"),
        "max_docs_per_subset": payload.get("max_docs_per_subset"),
    }


def _get_input_stats(args: argparse.Namespace, tokenizer) -> dict:
    cached = _input_stats_from_materialize_manifest(args.input_data_dir)
    if cached is not None:
        return cached
    return _scan_stats(
        args.input_data_dir,
        tokenizer,
        args.tokenizer_batch_size,
        args.tokenizer_threads,
    )


def _workflow_metadata(result) -> dict:
    return _jsonable(getattr(result, "metadata", {}) or {})


def _load_identification_checkpoint(root: Path) -> dict | None:
    path = root / "identification_checkpoint.json"
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    id_generator_path = Path(metadata["id_generator_path"])
    if not id_generator_path.is_file():
        raise FileNotFoundError(f"Checkpoint references missing ID generator: {id_generator_path}")
    return metadata


def _write_identification_checkpoint(root: Path, metadata: dict) -> None:
    _atomic_write_json(root / "identification_checkpoint.json", metadata)


def _reset_generated_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _clone_corpus_with_hardlinks(input_path: Path, output_path: Path) -> None:
    """Create a zero-copy clone when a dedup stage has nothing to remove."""
    _reset_generated_path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for source in _parquet_files(input_path):
        relative = source.relative_to(input_path)
        destination = output_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _make_executor():
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor

    return RayActorPoolExecutor(show_progress=True, progress_interval=30.0)


def _configure_local_ray(args: argparse.Namespace) -> None:
    """Bound local Ray startup and keep ID actors alive across Curator sub-pipelines."""
    import ray

    original_init = getattr(ray.init, "_nanochat_original_init", ray.init)
    original_shutdown = getattr(ray.shutdown, "_nanochat_original_shutdown", ray.shutdown)
    temp_dir = args.ray_temp_dir or Path("/tmp") / f"nanochat_curator_ray_{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    def controlled_init(*positional, **kwargs):
        if not positional and "address" not in kwargs:
            kwargs["address"] = "local"
        if not positional and kwargs.get("address") == "local":
            kwargs.setdefault("num_cpus", args.ray_num_cpus)
            kwargs.setdefault("include_dashboard", False)
            kwargs.setdefault("_temp_dir", str(temp_dir))
        return original_init(*positional, **kwargs)

    controlled_init._nanochat_original_init = original_init

    def keep_cluster_alive() -> None:
        return None

    keep_cluster_alive._nanochat_original_shutdown = original_shutdown
    ray.init = controlled_init
    ray.shutdown = keep_cluster_alive
    atexit.register(original_shutdown)


def _run_removal(
    input_path: Path,
    ids_path: Path,
    id_generator_path: Path,
    output_path: Path,
    input_blocksize: str,
    expected_duplicates: int,
    workers: int,
    batch_size: int,
    checkpoint_files: int,
    resume: bool,
) -> dict:
    from nemo_curator.stages.deduplication.id_generator import IdGeneratorBase
    from nemo_curator.utils.file_utils import (
        _split_files_as_per_blocksize,
        get_all_file_paths_and_size_under,
        parse_bytes_string_to_int,
    )

    started = time.time()
    checkpoint_path = output_path / REMOVAL_MANIFEST_NAME
    identity = {
        "format_version": 1,
        "input_path": str(input_path),
        "ids_path": str(ids_path),
        "id_generator_path": str(id_generator_path),
        "input_blocksize": input_blocksize,
        "expected_duplicates": expected_duplicates,
    }

    checkpoint = None
    if resume and checkpoint_path.is_file():
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if all(candidate.get(key) == value for key, value in identity.items()):
            checkpoint = candidate
            print(f"Reusing streaming removal checkpoint: {checkpoint_path}")
        else:
            raise ValueError(f"Streaming removal checkpoint does not match this run: {checkpoint_path}")
    if checkpoint is None:
        _reset_generated_path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            **identity,
            "status": "running",
            "created_at": _utc_now(),
            "completed_files": {},
        }
        _atomic_write_json(checkpoint_path, checkpoint)

    duplicate_files = _parquet_files(ids_path)
    if not duplicate_files:
        raise FileNotFoundError(f"No duplicate ID parquet files found under {ids_path}")
    duplicate_count = sum(pq.ParquetFile(path).metadata.num_rows for path in duplicate_files)
    if duplicate_count != expected_duplicates:
        raise RuntimeError(
            f"Duplicate ID count mismatch: expected {expected_duplicates:,}, found {duplicate_count:,}"
        )
    duplicate_ids = np.empty(duplicate_count, dtype=np.int64)
    offset = 0
    for path in duplicate_files:
        parquet = pq.ParquetFile(path)
        for record_batch in parquet.iter_batches(batch_size=1_000_000, columns=[CURATOR_ID_FIELD]):
            values = record_batch.column(0).to_numpy(zero_copy_only=False)
            duplicate_ids[offset : offset + len(values)] = values
            offset += len(values)
    duplicate_ids.sort()
    if duplicate_count > 1 and np.any(duplicate_ids[1:] == duplicate_ids[:-1]):
        raise RuntimeError("Duplicate ID output contains repeated IDs")
    print(f"Loaded and sorted {duplicate_count:,} duplicate IDs")

    registry_payload = json.loads(id_generator_path.read_text(encoding="utf-8"))
    id_generator = IdGeneratorBase(
        start_id=int(registry_payload["next_id"]),
        batch_registry=registry_payload["batch_registry"],
    )
    file_records = get_all_file_paths_and_size_under(
        str(input_path),
        recurse_subdirectories=True,
        keep_extensions=[".parquet"],
        storage_options=None,
        sort_by_size=True,
    )
    file_groups = _split_files_as_per_blocksize(
        sorted(file_records, key=lambda item: item[1]),
        parse_bytes_string_to_int(input_blocksize),
    )

    file_tasks = []
    registered_rows = 0
    for group in file_groups:
        group_hash = str(uuid.uuid5(uuid.NAMESPACE_URL, ";".join(group)))
        if group_hash not in registry_payload["batch_registry"]:
            raise KeyError(f"Source file group is absent from Curator ID registry: {group}")
        min_id, max_id = id_generator.get_batch_range(files=group, key=None)
        next_id = int(min_id)
        group_rows = 0
        for source_name in group:
            source = Path(source_name).resolve()
            source_rows = pq.ParquetFile(source).metadata.num_rows
            relative = source.relative_to(input_path)
            file_tasks.append(
                {
                    "source": source,
                    "relative": relative,
                    "source_size": source.stat().st_size,
                    "source_rows": source_rows,
                    "min_id": next_id,
                }
            )
            next_id += source_rows
            group_rows += source_rows
        if next_id - 1 != int(max_id):
            raise RuntimeError(
                f"Curator ID range mismatch for {group_hash}: registry={min_id}-{max_id}, rows={group_rows:,}"
            )
        registered_rows += group_rows
    if registered_rows != int(registry_payload["next_id"]):
        raise RuntimeError(
            f"Curator registry covers {registered_rows:,} rows, next_id is {registry_payload['next_id']:,}"
        )
    print(f"Validated Curator ID replay for {len(file_tasks):,} files and {registered_rows:,} rows")

    completed = checkpoint["completed_files"]
    pending = []
    for task in file_tasks:
        key = task["relative"].as_posix()
        record = completed.get(key)
        destination = output_path / task["relative"]
        if (
            record
            and record.get("source_size") == task["source_size"]
            and destination.is_file()
            and destination.stat().st_size == record.get("output_bytes")
        ):
            continue
        pending.append(task)
    print(f"Streaming removal: {len(completed):,} files checkpointed, {len(pending):,} pending")

    def filter_file(task: dict) -> tuple[str, dict]:
        source = task["source"]
        relative = task["relative"]
        destination = output_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_suffix(destination.suffix + ".tmp")
        if temp_path.exists():
            temp_path.unlink()

        source_rows = int(task["source_rows"])
        min_id = int(task["min_id"])
        first = int(np.searchsorted(duplicate_ids, min_id, side="left"))
        last = int(np.searchsorted(duplicate_ids, min_id + source_rows, side="left"))
        file_duplicate_ids = duplicate_ids[first:last]
        stats = {
            "docs": 0,
            "chars": 0,
            "hf_tokens": 0,
            "nanochat_tokens": 0,
            "original_occurrences": 0,
        }
        seen = {"hf_tokens": False, "nanochat_tokens": False, "original_occurrences": False}
        parquet = pq.ParquetFile(source)
        row_offset = 0
        with pq.ParquetWriter(temp_path, parquet.schema_arrow, compression="zstd") as writer:
            for record_batch in parquet.iter_batches(batch_size=batch_size):
                batch_min_id = min_id + row_offset
                batch_max_id = batch_min_id + record_batch.num_rows
                left = int(np.searchsorted(file_duplicate_ids, batch_min_id, side="left"))
                right = int(np.searchsorted(file_duplicate_ids, batch_max_id, side="left"))
                keep = np.ones(record_batch.num_rows, dtype=np.bool_)
                if right > left:
                    keep[file_duplicate_ids[left:right] - batch_min_id] = False
                filtered = record_batch.filter(pa.array(keep))
                writer.write_batch(filtered)
                stats["docs"] += filtered.num_rows
                text_column = filtered.column(filtered.schema.get_field_index("text"))
                stats["chars"] += int(pc.sum(pc.utf8_length(text_column)).as_py() or 0)
                for column_name, stats_name in (
                    ("hf_token_count", "hf_tokens"),
                    ("nanochat_token_count", "nanochat_tokens"),
                    ("count", "original_occurrences"),
                ):
                    column_index = filtered.schema.get_field_index(column_name)
                    if column_index >= 0:
                        column = filtered.column(column_index)
                        stats[stats_name] += int(pc.sum(column).as_py() or 0)
                        seen[stats_name] = seen[stats_name] or column.null_count < len(column)
                row_offset += record_batch.num_rows
        if row_offset != source_rows:
            raise RuntimeError(f"Read {row_offset:,}/{source_rows:,} rows from {source}")
        os.replace(temp_path, destination)
        return relative.as_posix(), {
            "source_size": int(task["source_size"]),
            "source_rows": source_rows,
            "output_rows": stats["docs"],
            "removed_rows": source_rows - stats["docs"],
            "output_bytes": destination.stat().st_size,
            "chars": stats["chars"],
            "hf_tokens": stats["hf_tokens"] if seen["hf_tokens"] else None,
            "nanochat_tokens": stats["nanochat_tokens"] if seen["nanochat_tokens"] else None,
            "original_occurrences": (
                stats["original_occurrences"] if seen["original_occurrences"] else None
            ),
        }

    since_checkpoint = 0
    last_checkpoint = time.monotonic()
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(filter_file, task): task for task in pending}
            for completed_count, future in enumerate(as_completed(futures), 1):
                key, record = future.result()
                completed[key] = record
                since_checkpoint += 1
                if since_checkpoint >= checkpoint_files or time.monotonic() - last_checkpoint >= 60:
                    checkpoint["updated_at"] = _utc_now()
                    _atomic_write_json(checkpoint_path, checkpoint)
                    print(
                        f"Streaming removal progress: {len(completed):,}/{len(file_tasks):,} files "
                        f"({100.0 * len(completed) / len(file_tasks):.2f}%)"
                    )
                    since_checkpoint = 0
                    last_checkpoint = time.monotonic()

    if len(completed) != len(file_tasks):
        raise RuntimeError(f"Removal checkpoint has {len(completed):,}/{len(file_tasks):,} files")
    records = list(completed.values())
    removed_rows = sum(int(record["removed_rows"]) for record in records)
    if removed_rows != expected_duplicates:
        raise RuntimeError(
            f"Streaming removal deleted {removed_rows:,} rows, expected {expected_duplicates:,}"
        )

    def sum_optional(name: str) -> int | None:
        values = [record.get(name) for record in records]
        return None if any(value is None for value in values) else sum(int(value) for value in values)

    output_stats = {
        "docs": sum(int(record["output_rows"]) for record in records),
        "chars": sum(int(record["chars"]) for record in records),
        "hf_tokens": sum_optional("hf_tokens"),
        "nanochat_tokens": sum_optional("nanochat_tokens"),
        "original_occurrences": sum_optional("original_occurrences"),
        "bytes": sum(int(record["output_bytes"]) for record in records),
        "files": len(records),
        "path": str(output_path),
        "nanochat_tokens_include_bos": sum_optional("nanochat_tokens") is not None,
        "computed_at": _utc_now(),
        "source": str(checkpoint_path),
    }
    checkpoint.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "num_duplicates_removed": removed_rows,
            "output_stats": output_stats,
        }
    )
    _atomic_write_json(checkpoint_path, checkpoint)
    return {
        "method": "streaming_curator_id_replay",
        "num_duplicates_removed": removed_rows,
        "files": len(records),
        "workers": workers,
        "batch_size": batch_size,
        "checkpoint_path": str(checkpoint_path),
        "elapsed_sec": time.time() - started,
        "output_stats": output_stats,
    }


def _run_exact(args: argparse.Namespace, manifest: dict, tokenizer) -> None:
    from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

    exact_ids_root = args.work_dir / "exact_identification"
    exact_data = args.work_dir / "post_exact"

    raw_stats = manifest["stats"].get("raw") or _get_input_stats(args, tokenizer)
    manifest["stats"]["raw"] = raw_stats

    started = time.time()
    identification = _load_identification_checkpoint(exact_ids_root) if args.resume else None
    if identification is None:
        _reset_generated_path(exact_ids_root)
        workflow = ExactDeduplicationWorkflow(
            input_path=str(args.input_data_dir),
            output_path=str(exact_ids_root),
            input_filetype="parquet",
            input_file_extensions=[".parquet"],
            input_blocksize=args.input_blocksize,
            identification_batchsize=args.exact_identification_batchsize,
            text_field="text",
            assign_id=True,
            total_nparts=args.exact_total_nparts,
            write_kwargs={"compression": "zstd"},
        )
        identification = _workflow_metadata(workflow.run(executor=_make_executor()))
        _write_identification_checkpoint(exact_ids_root, identification)
    else:
        print(f"Reusing exact identification checkpoint: {exact_ids_root}")
    duplicates = int(identification.get("num_duplicates", 0))
    id_generator_path = Path(identification["id_generator_path"])
    removal = None
    manifest["stages"]["exact"].update(
        {
            "status": "identification_complete",
            "num_duplicates": duplicates,
            "identification": identification,
        }
    )
    _atomic_write_json(args.work_dir / MANIFEST_NAME, manifest)
    if duplicates == 0:
        _reset_generated_path(exact_data)
        data_path = args.input_data_dir
        exact_stats = dict(raw_stats)
        exact_stats["path"] = str(data_path)
        skipped_removal = True
    else:
        removal = _run_removal(
            args.input_data_dir,
            exact_ids_root / "ExactDuplicateIds",
            id_generator_path,
            exact_data,
            args.input_blocksize,
            duplicates,
            args.removal_workers,
            args.removal_batch_size,
            args.removal_checkpoint_files,
            args.resume,
        )
        data_path = exact_data
        exact_stats = removal["output_stats"]
        skipped_removal = False

    observed = int(raw_stats["docs"]) - int(exact_stats["docs"])
    if observed != duplicates:
        raise RuntimeError(f"Exact count mismatch: Curator identified {duplicates:,}, stats observed {observed:,}")

    manifest["stats"]["post_exact"] = exact_stats
    manifest["stages"]["exact"].update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "elapsed_sec": time.time() - started,
            "num_duplicates": duplicates,
            "data_path": str(data_path),
            "ids_path": str(exact_ids_root / "ExactDuplicateIds"),
            "id_generator_path": str(id_generator_path),
            "removal_skipped_no_duplicates": skipped_removal,
            "identification": identification,
            "removal": removal,
        }
    )


def _run_fuzzy(args: argparse.Namespace, manifest: dict, tokenizer) -> None:
    from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

    exact_stage = manifest["stages"]["exact"]
    if exact_stage.get("status") != "complete":
        raise RuntimeError("Fuzzy stage requires a completed exact stage")
    input_path = Path(exact_stage["data_path"])
    fuzzy_cache = args.work_dir / "fuzzy_cache"
    fuzzy_ids_root = args.work_dir / "fuzzy_identification"

    started = time.time()
    identification = _load_identification_checkpoint(fuzzy_ids_root) if args.resume else None
    if identification is None:
        _reset_generated_path(fuzzy_cache)
        _reset_generated_path(fuzzy_ids_root)
        workflow = FuzzyDeduplicationWorkflow(
            input_path=str(input_path),
            cache_path=str(fuzzy_cache),
            output_path=str(fuzzy_ids_root),
            input_filetype="parquet",
            input_file_extensions=[".parquet"],
            input_blocksize=args.input_blocksize,
            text_field="text",
            seed=args.seed,
            char_ngrams=args.char_ngrams,
            num_bands=args.num_bands,
            minhashes_per_band=args.minhashes_per_band,
            use_64_bit_hash=args.use_64_bit_hash,
            bands_per_iteration=args.bands_per_iteration,
            lsh_num_output_partitions=args.lsh_num_output_partitions,
            write_kwargs={"compression": "zstd"},
        )
        identification = _workflow_metadata(workflow.run(executor=_make_executor()))
        _write_identification_checkpoint(fuzzy_ids_root, identification)
    else:
        print(f"Reusing fuzzy identification checkpoint: {fuzzy_ids_root}")
    duplicates = int(identification.get("num_duplicates", 0))
    id_generator_path = Path(identification["id_generator_path"])
    manifest["stages"]["fuzzy"].update(
        {
            "status": "identification_complete",
            "num_duplicates": duplicates,
            "identification": identification,
        }
    )
    _atomic_write_json(args.work_dir / MANIFEST_NAME, manifest)
    if duplicates == 0:
        _clone_corpus_with_hardlinks(input_path, args.output_data_dir)
        removal = {"num_duplicates_removed": 0, "hardlink_clone": True}
        final_stats = _scan_stats(
            args.output_data_dir,
            tokenizer,
            args.tokenizer_batch_size,
            args.tokenizer_threads,
        )
    else:
        removal = _run_removal(
            input_path,
            fuzzy_ids_root / "FuzzyDuplicateIds",
            id_generator_path,
            args.output_data_dir,
            args.input_blocksize,
            duplicates,
            args.removal_workers,
            args.removal_batch_size,
            args.removal_checkpoint_files,
            args.resume,
        )
        final_stats = removal["output_stats"]
    exact_stats = manifest["stats"]["post_exact"]
    observed = int(exact_stats["docs"]) - int(final_stats["docs"])
    if observed != duplicates:
        raise RuntimeError(f"Fuzzy count mismatch: Curator identified {duplicates:,}, stats observed {observed:,}")

    manifest["stats"]["post_fuzzy"] = final_stats
    manifest["stages"]["fuzzy"].update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "elapsed_sec": time.time() - started,
            "num_duplicates": duplicates,
            "data_path": str(args.output_data_dir),
            "ids_path": str(fuzzy_ids_root / "FuzzyDuplicateIds"),
            "id_generator_path": str(id_generator_path),
            "identification": identification,
            "removal": removal,
        }
    )

    if args.cleanup_intermediates:
        _reset_generated_path(fuzzy_cache)
        if input_path != args.input_data_dir:
            _reset_generated_path(input_path)
        manifest["stages"]["fuzzy"]["intermediates_cleaned"] = True


def _run_stats(args: argparse.Namespace, manifest: dict, tokenizer) -> None:
    if manifest["stages"]["exact"].get("status") != "complete":
        raise RuntimeError("Stats stage requires a completed exact stage")
    if manifest["stages"]["fuzzy"].get("status") != "complete":
        raise RuntimeError("Stats stage requires a completed fuzzy stage")

    if "raw" not in manifest["stats"]:
        manifest["stats"]["raw"] = _get_input_stats(args, tokenizer)
    if "post_exact" not in manifest["stats"]:
        exact_path = Path(manifest["stages"]["exact"]["data_path"])
        manifest["stats"]["post_exact"] = _scan_stats(
            exact_path, tokenizer, args.tokenizer_batch_size, args.tokenizer_threads
        )
    if "post_fuzzy" not in manifest["stats"]:
        manifest["stats"]["post_fuzzy"] = _scan_stats(
            args.output_data_dir, tokenizer, args.tokenizer_batch_size, args.tokenizer_threads
        )

    raw_docs = int(manifest["stats"]["raw"]["docs"])
    exact_docs = int(manifest["stats"]["post_exact"]["docs"])
    fuzzy_docs = int(manifest["stats"]["post_fuzzy"]["docs"])
    exact_removed = raw_docs - exact_docs
    fuzzy_removed = exact_docs - fuzzy_docs
    expected_exact = int(manifest["stages"]["exact"]["num_duplicates"])
    expected_fuzzy = int(manifest["stages"]["fuzzy"]["num_duplicates"])
    if exact_removed != expected_exact or fuzzy_removed != expected_fuzzy:
        raise RuntimeError(
            "Stats integrity check failed: "
            f"exact={exact_removed:,}/{expected_exact:,}, fuzzy={fuzzy_removed:,}/{expected_fuzzy:,}"
        )
    manifest["integrity"] = {
        "status": "passed",
        "raw_minus_exact_docs": exact_removed,
        "exact_minus_fuzzy_docs": fuzzy_removed,
        "checked_at": _utc_now(),
    }
    manifest["stages"]["stats"].update({"status": "complete", "completed_at": _utc_now()})


def _format_int(value) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def _ratio(value, baseline) -> str:
    if value is None or baseline in (None, 0):
        return "n/a"
    return f"{100.0 * float(value) / float(baseline):.4f}%"


def _removed(before, after) -> str:
    if before is None or after is None:
        return "n/a"
    return f"{int(before) - int(after):,}"


def _render_report(args: argparse.Namespace, manifest: dict) -> str:
    raw = manifest["stats"]["raw"]
    exact = manifest["stats"]["post_exact"]
    fuzzy = manifest["stats"]["post_fuzzy"]
    subsets = raw.get("subsets")
    scope = f"all {len(subsets)} Hugging Face configs" if subsets else "the materialized input corpus"
    capped = raw.get("max_docs_per_subset")
    cap_note = "full configs" if capped is None else f"at most {int(capped):,} rows per config"

    rows = []
    for label, stats in (("Raw Fortified", raw), ("After exact", exact), ("After fuzzy", fuzzy)):
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    _format_int(stats.get("docs")),
                    _ratio(stats.get("docs"), raw.get("docs")),
                    _format_int(stats.get("hf_tokens")),
                    _ratio(stats.get("hf_tokens"), raw.get("hf_tokens")),
                    _format_int(stats.get("nanochat_tokens")),
                    _ratio(stats.get("nanochat_tokens"), raw.get("nanochat_tokens")),
                    _format_int(stats.get("chars")),
                ]
            )
            + " |"
        )

    config = manifest["config"]
    num_bands = int(config["num_bands"])
    minhashes_per_band = int(config["minhashes_per_band"])
    total_minhashes = num_bands * minhashes_per_band
    candidate_midpoint = (1.0 - 0.5 ** (1.0 / num_bands)) ** (1.0 / minhashes_per_band)
    exact_removed = int(raw["docs"]) - int(exact["docs"])
    fuzzy_removed = int(exact["docs"]) - int(fuzzy["docs"])
    total_removed = int(raw["docs"]) - int(fuzzy["docs"])
    return "\n".join(
        [
            "# FineWeb-EDU-Fortified exact + fuzzy dedup report",
            "",
            f"Generated: {manifest.get('completed_at') or _utc_now()}",
            "",
            "## Experiment overview",
            "",
            "This experiment measures how much residual redundancy remains in the complete "
            "FineWeb-EDU-Fortified corpus after applying exact deduplication followed by MinHash-LSH fuzzy "
            "deduplication. The exact stage is retained as a residual check because the published corpus already "
            "applied global MD5 deduplication; the fuzzy stage targets near-duplicate documents that differ in "
            "formatting, boilerplate, or small text edits.",
            "",
            f"For fuzzy matching, each document is represented by `{config['char_ngrams']}`-character n-grams and "
            f"summarized by `{total_minhashes}` `{64 if config['use_64_bit_hash'] else 32}`-bit MinHashes generated "
            f"with seed `{config['seed']}`. The signature is divided into `{num_bands}` LSH bands with "
            f"`{minhashes_per_band}` hashes per band. Under the standard MinHash independence approximation, this "
            f"banding curve gives an individual pair a 50% candidate probability at character-ngram Jaccard "
            f"similarity approximately `{candidate_midpoint:.4f}`. This is a probabilistic candidate boundary, not "
            "a hard Jaccard threshold.",
            "",
            "Candidate relationships are merged into connected components, and one document is retained from each "
            "component. Because connected components are transitive, a removed document can have lower direct "
            "Jaccard similarity to the retained component representative than the pairwise LSH operating point. "
            f"Processing `{config['bands_per_iteration']}` bands per iteration only bounds execution resources; it "
            f"does not change the `{num_bands}`-band matching configuration. Input block size, removal worker count, "
            "and removal batch size are throughput settings rather than similarity parameters.",
            "",
            "## Result",
            "",
            "| Stage | Rows | Rows kept | HF tokens | HF tokens kept | NanoChat tokens (+BOS/row) | NanoChat tokens kept | Characters |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            f"Exact removed **{exact_removed:,} rows**. Fuzzy then removed **{fuzzy_removed:,} rows**. "
            f"Combined removal was **{total_removed:,} rows** ({100.0 * total_removed / int(raw['docs']):.4f}%).",
            "",
            "Token deltas:",
            "",
            f"- Exact: HF {_removed(raw.get('hf_tokens'), exact.get('hf_tokens'))}; "
            f"NanoChat {_removed(raw.get('nanochat_tokens'), exact.get('nanochat_tokens'))}",
            f"- Fuzzy: HF {_removed(exact.get('hf_tokens'), fuzzy.get('hf_tokens'))}; "
            f"NanoChat {_removed(exact.get('nanochat_tokens'), fuzzy.get('nanochat_tokens'))}",
            f"- Combined: HF {_removed(raw.get('hf_tokens'), fuzzy.get('hf_tokens'))}; "
            f"NanoChat {_removed(raw.get('nanochat_tokens'), fuzzy.get('nanochat_tokens'))}",
            "",
            "## Scope",
            "",
            f"Dataset: `airtrain-ai/fineweb-edu-fortified`; {scope}; {cap_note}. "
            "NanoChat counts use the local FineWeb-EDU tokenizer and include one BOS token per document.",
            "",
            "FineWeb-EDU-Fortified already applied global MD5 exact-match deduplication. "
            "The exact result here therefore measures only additional exact duplicates remaining in the published Fortified corpus.",
            "",
            "## Dedup configuration",
            "",
            f"- Order: exact identification/removal, then fuzzy identification/removal",
            f"- Curator input block size: `{config['input_blocksize']}`",
            f"- Removal: deterministic Curator-ID replay, `{config['removal_workers']}` streaming workers",
            f"- Removal record-batch size: `{config['removal_batch_size']:,}` rows",
            f"- Fuzzy seed: `{config['seed']}`",
            f"- Character n-grams: `{config['char_ngrams']}`",
            f"- LSH bands: `{num_bands}`",
            f"- MinHashes per band: `{minhashes_per_band}`",
            f"- Total MinHashes: `{total_minhashes}`",
            f"- Hash width: `{'64' if config['use_64_bit_hash'] else '32'}-bit`",
            f"- Bands per iteration: `{config['bands_per_iteration']}`",
            "- Retention policy: one document per Curator connected component",
            "",
            "## Artifacts",
            "",
            f"- Input: `{args.input_data_dir}`",
            f"- Final parquet: `{args.output_data_dir}`",
            f"- Machine-readable manifest: `{args.work_dir / MANIFEST_NAME}`",
            "",
            "Integrity check: passed; row deltas exactly match Curator's duplicate-ID counts.",
            "",
        ]
    )


def _run_report(args: argparse.Namespace, manifest: dict) -> None:
    if manifest["stages"]["stats"].get("status") != "complete":
        raise RuntimeError("Report stage requires completed stats")
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(_render_report(args, manifest), encoding="utf-8")
    manifest["stages"]["report"].update(
        {"status": "complete", "completed_at": _utc_now(), "path": str(args.report_path)}
    )


def _new_manifest(args: argparse.Namespace) -> dict:
    return {
        "format_version": 1,
        "status": "running",
        "created_at": _utc_now(),
        "input_data_dir": str(args.input_data_dir),
        "output_data_dir": str(args.output_data_dir),
        "work_dir": str(args.work_dir),
        "report_path": str(args.report_path),
        "config": {
            "input_blocksize": args.input_blocksize,
            "exact_identification_batchsize": args.exact_identification_batchsize,
            "exact_total_nparts": args.exact_total_nparts,
            "seed": args.seed,
            "char_ngrams": args.char_ngrams,
            "num_bands": args.num_bands,
            "minhashes_per_band": args.minhashes_per_band,
            "use_64_bit_hash": args.use_64_bit_hash,
            "bands_per_iteration": args.bands_per_iteration,
            "lsh_num_output_partitions": args.lsh_num_output_partitions,
            "tokenizer_dir": str(args.tokenizer_dir) if args.tokenizer_dir else None,
            "tokenizer_batch_size": args.tokenizer_batch_size,
            "tokenizer_threads": args.tokenizer_threads,
            "ray_num_cpus": args.ray_num_cpus,
            "ray_temp_dir": str(args.ray_temp_dir) if args.ray_temp_dir else None,
            "removal_workers": args.removal_workers,
            "removal_batch_size": args.removal_batch_size,
            "removal_checkpoint_files": args.removal_checkpoint_files,
        },
        "stages": {name: {"status": "pending"} for name in ("exact", "fuzzy", "stats", "report")},
        "stats": {},
    }


def _load_or_create_manifest(args: argparse.Namespace) -> tuple[dict, Path]:
    manifest_path = args.work_dir / MANIFEST_NAME
    if args.overwrite:
        _reset_generated_path(args.work_dir)
        _reset_generated_path(args.output_data_dir)
        if args.report_path.exists():
            args.report_path.unlink()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.is_file():
        if not args.resume and args.stage in ("all", "exact"):
            raise FileExistsError(f"{manifest_path} exists; pass --resume or --overwrite")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in (
            ("input_data_dir", str(args.input_data_dir)),
            ("output_data_dir", str(args.output_data_dir)),
            ("work_dir", str(args.work_dir)),
        ):
            if manifest.get(key) != value:
                raise ValueError(f"Manifest {key}={manifest.get(key)!r}, requested {value!r}")
    else:
        if args.resume:
            raise FileNotFoundError(f"Cannot resume without {manifest_path}")
        manifest = _new_manifest(args)
        _atomic_write_json(manifest_path, manifest)
    manifest["config"].update(
        {
            "removal_workers": args.removal_workers,
            "removal_batch_size": args.removal_batch_size,
            "removal_checkpoint_files": args.removal_checkpoint_files,
        }
    )
    return manifest, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact + fuzzy deduplication for FineWeb-EDU-Fortified")
    parser.add_argument("--input-data-dir", type=Path, required=True)
    parser.add_argument("--output-data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "exact", "fuzzy", "stats", "report"], default="all")
    parser.add_argument("--input-blocksize", default="512MiB")
    parser.add_argument("--exact-identification-batchsize", type=int, default=1)
    parser.add_argument("--exact-total-nparts", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--char-ngrams", type=int, default=24)
    parser.add_argument("--num-bands", type=int, default=20)
    parser.add_argument("--minhashes-per-band", type=int, default=13)
    parser.add_argument("--use-64-bit-hash", action="store_true")
    parser.add_argument("--bands-per-iteration", type=int, default=5)
    parser.add_argument("--lsh-num-output-partitions", type=int)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--tokenizer-threads", type=int, default=8)
    parser.add_argument("--ray-num-cpus", type=int, default=64)
    parser.add_argument("--ray-temp-dir", type=Path)
    parser.add_argument("--removal-workers", type=int, default=32)
    parser.add_argument("--removal-batch-size", type=int, default=8192)
    parser.add_argument("--removal-checkpoint-files", type=int, default=25)
    parser.add_argument("--cleanup-intermediates", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("input_data_dir", "output_data_dir", "work_dir", "report_path"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.tokenizer_dir is not None:
        args.tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    if args.ray_temp_dir is not None:
        args.ray_temp_dir = args.ray_temp_dir.expanduser().resolve()
    for name in ("removal_workers", "removal_batch_size", "removal_checkpoint_files"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not args.input_data_dir.is_dir():
        raise FileNotFoundError(args.input_data_dir)
    if args.input_blocksize != "512MiB":
        print(
            "Warning: Curator recommends partitions no larger than 512 MiB; "
            f"requested {args.input_blocksize}"
        )

    manifest, manifest_path = _load_or_create_manifest(args)
    tokenizer = _load_tokenizer(args.tokenizer_dir)
    _configure_local_ray(args)
    targets = ["exact", "fuzzy", "stats", "report"] if args.stage == "all" else [args.stage]
    runners = {
        "exact": lambda: _run_exact(args, manifest, tokenizer),
        "fuzzy": lambda: _run_fuzzy(args, manifest, tokenizer),
        "stats": lambda: _run_stats(args, manifest, tokenizer),
        "report": lambda: _run_report(args, manifest),
    }

    for stage in targets:
        if args.resume and manifest["stages"][stage].get("status") == "complete":
            print(f"Skipping completed stage: {stage}")
            continue
        previous = manifest["stages"][stage] if args.resume else {}
        manifest["stages"][stage] = {
            **previous,
            "status": "running",
            "started_at": _utc_now(),
        }
        manifest["stages"][stage].pop("error", None)
        manifest["stages"][stage].pop("failed_at", None)
        _atomic_write_json(manifest_path, manifest)
        print(f"Starting stage: {stage}")
        try:
            runners[stage]()
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["stages"][stage].update(
                {"status": "failed", "failed_at": _utc_now(), "error": f"{type(exc).__name__}: {exc}"}
            )
            _atomic_write_json(manifest_path, manifest)
            raise
        _atomic_write_json(manifest_path, manifest)
        print(f"Completed stage: {stage}")

    if all(manifest["stages"][stage].get("status") == "complete" for stage in manifest["stages"]):
        manifest["status"] = "complete"
        manifest["completed_at"] = _utc_now()
        if manifest["stages"]["report"].get("status") == "complete":
            args.report_path.write_text(_render_report(args, manifest), encoding="utf-8")
    else:
        manifest["status"] = "partial"
    _atomic_write_json(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")
    if args.report_path.is_file():
        print(f"Wrote report: {args.report_path}")


if __name__ == "__main__":
    main()
