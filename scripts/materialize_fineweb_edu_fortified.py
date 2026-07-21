"""Materialize FineWeb-EDU-Fortified as dedup-ready parquet shards.

With no corpus-mode flags this keeps the original NanoChat pilot behavior: one
Common Crawl subset, train shards, and a final validation shard. Corpus mode
(``--all-subsets``, repeatable ``--subset``, ``--max-docs-per-subset``, or
``--resume``) streams the selected Hugging Face configs into resumable parquet
shards while dropping the large precomputed embedding column.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import get_dataset_config_names, load_dataset


ROW_GROUP_SIZE = 1024
DEFAULT_SUBSET = "CC-MAIN-2024-10"
SOURCE_COLUMNS = ["text", "id", "dump", "url", "count", "token_count"]
CORPUS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("subset", pa.string(), nullable=False),
        pa.field("source_id", pa.string()),
        pa.field("dump", pa.string()),
        pa.field("url", pa.string()),
        pa.field("count", pa.int64()),
        pa.field("hf_token_count", pa.int64()),
        pa.field("nanochat_token_count", pa.int64()),
        pa.field("text", pa.string(), nullable=False),
    ]
)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def _write_texts(path: Path, texts: list[str]) -> dict:
    table = pa.table({"text": texts})
    pq.write_table(table, path, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    return {
        "file": path.name,
        "docs": len(texts),
        "chars": sum(len(text or "") for text in texts),
        "bytes": path.stat().st_size,
    }


def _reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        parquet_files = list(output_dir.glob("*.parquet"))
        if parquet_files and not overwrite:
            raise FileExistsError(f"{output_dir} already contains parquet files; pass --overwrite")
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_tokenizer(tokenizer_dir: Path | None):
    if tokenizer_dir is None:
        return None
    from nanochat.tokenizer import RustBPETokenizer

    resolved = tokenizer_dir.expanduser().resolve()
    if not (resolved / "tokenizer.pkl").is_file():
        raise FileNotFoundError(f"NanoChat tokenizer.pkl not found in {resolved}")
    return RustBPETokenizer.from_directory(str(resolved))


def _token_lengths(tokenizer, texts: list[str], batch_size: int, num_threads: int) -> list[int | None]:
    if tokenizer is None:
        return [None] * len(texts)
    bos = tokenizer.get_bos_token_id()
    lengths: list[int] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        encoded = tokenizer.encode(batch, prepend=bos, num_threads=num_threads)
        lengths.extend(len(token_ids) for token_ids in encoded)
    return lengths


def _as_optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _write_corpus_shard(
    path: Path,
    source_table: pa.Table,
    tokenizer,
    tokenizer_batch_size: int,
    tokenizer_threads: int,
    subset: str,
    source_row_start: int,
) -> dict:
    num_rows = source_table.num_rows

    def values(name: str):
        if name not in source_table.column_names:
            return [None] * num_rows
        return source_table[name].combine_chunks().to_pylist()

    texts = [str(text or "") for text in values("text")]
    source_ids = values("id")
    nanochat_lengths = _token_lengths(tokenizer, texts, tokenizer_batch_size, tokenizer_threads)
    columns = {
        "doc_id": [
            f"{subset}:{source_id if source_id is not None else f'row-{source_row_start + offset:012d}'}"
            for offset, source_id in enumerate(source_ids)
        ],
        "subset": [subset] * num_rows,
        "source_id": [str(value) if value is not None else None for value in source_ids],
        "dump": [str(value) if value is not None else None for value in values("dump")],
        "url": [str(value) if value is not None else None for value in values("url")],
        "count": [_as_optional_int(value) for value in values("count")],
        "hf_token_count": [_as_optional_int(value) for value in values("token_count")],
        "nanochat_token_count": nanochat_lengths,
        "text": texts,
    }
    metadata = {
        b"fineweb_edu_fortified_subset": subset.encode("utf-8"),
        b"source_row_start": str(source_row_start).encode("ascii"),
    }
    table = pa.Table.from_pydict(columns, schema=CORPUS_SCHEMA).replace_schema_metadata(metadata)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temp_path, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    os.replace(temp_path, path)

    hf_tokens = sum(value or 0 for value in columns["hf_token_count"])
    nanochat_tokens = None if tokenizer is None else sum(int(value) for value in nanochat_lengths)
    return {
        "file": path.name,
        "subset": subset,
        "source_row_start": source_row_start,
        "docs": num_rows,
        "chars": sum(len(text) for text in texts),
        "hf_tokens": hf_tokens,
        "nanochat_tokens": nanochat_tokens,
        "original_occurrences": sum(value or 0 for value in columns["count"]),
        "bytes": path.stat().st_size,
    }


def _manifest_totals(shards: list[dict]) -> dict:
    nanochat_values = [shard.get("nanochat_tokens") for shard in shards]
    return {
        "docs": sum(shard["docs"] for shard in shards),
        "chars": sum(shard["chars"] for shard in shards),
        "hf_tokens": sum(shard.get("hf_tokens", 0) for shard in shards),
        "nanochat_tokens": (
            None if any(value is None for value in nanochat_values) else sum(int(value) for value in nanochat_values)
        ),
        "original_occurrences": sum(shard.get("original_occurrences", 0) for shard in shards),
        "bytes": sum(shard["bytes"] for shard in shards),
        "files": len(shards),
    }


def _save_manifest(path: Path, manifest: dict) -> None:
    manifest["totals"] = _manifest_totals(manifest["shards"])
    _atomic_write_json(path, manifest)


def _resolve_subsets(args: argparse.Namespace) -> list[str]:
    if args.all_subsets:
        if args.subset:
            raise ValueError("--all-subsets cannot be combined with --subset")
        subsets = get_dataset_config_names(args.dataset)
    else:
        subsets = args.subset or [DEFAULT_SUBSET]
    subsets = list(dict.fromkeys(subsets))
    if not subsets:
        raise ValueError("No dataset subsets selected")
    return subsets


def _validate_resume_manifest(manifest: dict, args: argparse.Namespace, subsets: list[str], output_dir: Path) -> None:
    tokenizer_dir = str(args.tokenizer_dir.expanduser().resolve()) if args.tokenizer_dir else None
    expected = {
        "dataset": args.dataset,
        "split": args.split,
        "subsets": subsets,
        "docs_per_shard": args.docs_per_shard,
        "max_docs_per_subset": args.max_docs_per_subset,
        "tokenizer_dir": tokenizer_dir,
        "include_bos": tokenizer_dir is not None,
        "output_dir": str(output_dir),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Resume argument mismatch for {key}: manifest has {manifest.get(key)!r}, requested {value!r}"
            )
    listed_files = {shard["file"] for shard in manifest.get("shards", [])}
    missing = sorted(name for name in listed_files if not (output_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Resume manifest references missing shards: {missing[:5]}")
    unexpected_paths = sorted(path for path in output_dir.glob("*.parquet") if path.name not in listed_files)
    unexpected = [path.name for path in unexpected_paths]
    if unexpected:
        recoverable = all(path.name.startswith("shard_") for path in unexpected_paths)
        if args.recover_orphan_shards and recoverable:
            for path in unexpected_paths:
                path.unlink()
            print(f"Removed {len(unexpected_paths)} unmanifested shard(s): {unexpected[:5]}")
        else:
            raise ValueError(
                f"Found parquet shards not recorded in the resume manifest: {unexpected[:5]}. "
                "Inspect them, pass --recover-orphan-shards, or restart with --overwrite."
            )


def _new_manifest(args: argparse.Namespace, subsets: list[str], output_dir: Path) -> dict:
    return {
        "format_version": 2,
        "mode": "corpus",
        "status": "running",
        "dataset": args.dataset,
        "split": args.split,
        "subsets": subsets,
        "docs_per_shard": args.docs_per_shard,
        "max_docs_per_subset": args.max_docs_per_subset,
        "tokenizer_dir": str(args.tokenizer_dir.expanduser().resolve()) if args.tokenizer_dir else None,
        "include_bos": args.tokenizer_dir is not None,
        "output_dir": str(output_dir),
        "shards": [],
        "subset_stats": {
            subset: {"status": "pending", "docs": 0, "chars": 0, "hf_tokens": 0, "nanochat_tokens": 0}
            for subset in subsets
        },
    }


def _run_corpus(args: argparse.Namespace) -> None:
    if args.docs_per_shard <= 0:
        raise ValueError("--docs-per-shard must be positive")
    if args.max_docs_per_subset is not None and args.max_docs_per_subset <= 0:
        raise ValueError("--max-docs-per-subset must be positive when set")
    if args.tokenizer_batch_size <= 0 or args.tokenizer_threads <= 0:
        raise ValueError("Tokenizer batch size and thread count must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "materialize_manifest.json"
    subsets = _resolve_subsets(args)

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume_manifest(manifest, args, subsets, output_dir)
        manifest["status"] = "running"
        manifest.pop("error", None)
    else:
        if list(output_dir.glob("*.parquet")) or manifest_path.exists():
            raise FileExistsError(f"{output_dir} already contains materialized data; pass --resume or --overwrite")
        manifest = _new_manifest(args, subsets, output_dir)
    _save_manifest(manifest_path, manifest)

    tokenizer = _load_tokenizer(args.tokenizer_dir)
    shard_index = len(manifest["shards"])

    try:
        for subset in subsets:
            subset_state = manifest["subset_stats"][subset]
            if subset_state.get("status") == "complete":
                print(f"Skipping completed subset {subset}: {subset_state['docs']:,} docs")
                continue

            already_written = int(subset_state.get("docs", 0))
            dataset = load_dataset(args.dataset, name=subset, split=args.split, streaming=True)
            available_columns = set(dataset.column_names or (dataset.features or {}).keys())
            selected_columns = [column for column in SOURCE_COLUMNS if column in available_columns]
            if "text" not in selected_columns:
                raise ValueError(f"Subset {subset} does not expose a text column")
            if hasattr(dataset, "select_columns"):
                dataset = dataset.select_columns(selected_columns)
            if already_written:
                dataset = dataset.skip(already_written)
                print(f"Resuming {subset} after {already_written:,} docs")
            else:
                print(f"Materializing {subset}")
            dataset = dataset.with_format("arrow")

            max_docs = args.max_docs_per_subset
            remaining = None if max_docs is None else max_docs - already_written
            if remaining is not None and remaining <= 0:
                subset_state["status"] = "complete"
                subset_state["capped"] = True
                _save_manifest(manifest_path, manifest)
                continue

            source_row_index = already_written
            dataset_iterator = dataset.iter(batch_size=args.docs_per_shard)
            for source_table in dataset_iterator:
                if remaining is not None and remaining == 0:
                    break
                if remaining is not None and source_table.num_rows > remaining:
                    source_table = source_table.slice(0, remaining)
                if source_table.num_rows == 0:
                    continue

                shard_path = output_dir / f"shard_{shard_index:06d}.parquet"
                shard = _write_corpus_shard(
                    shard_path,
                    source_table,
                    tokenizer,
                    args.tokenizer_batch_size,
                    args.tokenizer_threads,
                    subset,
                    source_row_index,
                )
                manifest["shards"].append(shard)
                subset_state["docs"] += shard["docs"]
                subset_state["chars"] += shard["chars"]
                subset_state["hf_tokens"] += shard["hf_tokens"]
                if shard["nanochat_tokens"] is None:
                    subset_state["nanochat_tokens"] = None
                elif subset_state.get("nanochat_tokens") is not None:
                    subset_state["nanochat_tokens"] += shard["nanochat_tokens"]
                subset_state["bytes"] = subset_state.get("bytes", 0) + shard["bytes"]
                source_row_index += shard["docs"]
                if remaining is not None:
                    remaining -= shard["docs"]
                shard_index += 1
                _save_manifest(manifest_path, manifest)
                print(
                    f"Wrote {shard_path.name}: {shard['docs']:,} docs; "
                    f"total={manifest['totals']['docs']:,}"
                )
            close_iterator = getattr(dataset_iterator, "close", None)
            if close_iterator is not None:
                close_iterator()

            subset_state["status"] = "complete"
            subset_state["capped"] = max_docs is not None and subset_state["docs"] >= max_docs
            _save_manifest(manifest_path, manifest)
            print(f"Completed {subset}: {subset_state['docs']:,} docs")

        manifest["status"] = "complete"
        _save_manifest(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _save_manifest(manifest_path, manifest)
        raise

    totals = manifest["totals"]
    nanochat_tokens = totals["nanochat_tokens"]
    nanochat_display = "unavailable" if nanochat_tokens is None else f"{nanochat_tokens:,}"
    print(
        f"Completed corpus materialization: docs={totals['docs']:,} "
        f"hf_tokens={totals['hf_tokens']:,} nanochat_tokens={nanochat_display}"
    )
    print(f"Wrote manifest: {manifest_path}")


def _run_legacy(args: argparse.Namespace) -> None:
    if args.train_docs <= 0:
        raise ValueError("--train-docs must be positive")
    if args.val_docs <= 0:
        raise ValueError("--val-docs must be positive")
    if args.docs_per_shard <= 0:
        raise ValueError("--docs-per-shard must be positive")

    subset = (args.subset or [DEFAULT_SUBSET])[0]
    output_dir = args.output_dir.expanduser().resolve()
    _reset_output_dir(output_dir, args.overwrite)

    dataset = load_dataset(args.dataset, name=subset, split=args.split, streaming=True)
    if hasattr(dataset, "select_columns"):
        dataset = dataset.select_columns(["text"])

    needed = args.train_docs + args.val_docs
    texts = (row.get("text") or "" for _, row in zip(range(needed), dataset))

    shards = []
    train_written = 0
    shard_idx = 0
    text_iter = iter(texts)
    while train_written < args.train_docs:
        take = min(args.docs_per_shard, args.train_docs - train_written)
        batch = []
        for _ in range(take):
            try:
                batch.append(next(text_iter))
            except StopIteration as exc:
                raise RuntimeError(f"Dataset ended after {train_written} train docs") from exc
        path = output_dir / f"shard_{shard_idx:05d}.parquet"
        shards.append(_write_texts(path, batch))
        train_written += len(batch)
        shard_idx += 1
        print(f"Wrote train shard {path.name}: {len(batch):,} docs")

    val_batch = []
    for _ in range(args.val_docs):
        try:
            val_batch.append(next(text_iter))
        except StopIteration as exc:
            raise RuntimeError(f"Dataset ended before writing {args.val_docs} validation docs") from exc
    val_path = output_dir / "shard_99999.parquet"
    val_info = _write_texts(val_path, val_batch)
    print(f"Wrote validation shard {val_path.name}: {len(val_batch):,} docs")

    manifest = {
        "dataset": args.dataset,
        "subset": subset,
        "split": args.split,
        "train_docs": train_written,
        "val_docs": len(val_batch),
        "docs_per_shard": args.docs_per_shard,
        "train_shards": shards,
        "val_shard": val_info,
        "output_dir": str(output_dir),
    }
    manifest_path = output_dir / "materialize_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize airtrain-ai/fineweb-edu-fortified")
    parser.add_argument("--dataset", default="airtrain-ai/fineweb-edu-fortified")
    parser.add_argument("--subset", action="append", help="Dataset config; repeat to select multiple configs")
    parser.add_argument("--all-subsets", action="store_true", help="Materialize every Hugging Face dataset config")
    parser.add_argument("--corpus", action="store_true", help="Force resumable corpus mode for one config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-docs", type=int, default=200_000, help="Legacy pilot train rows")
    parser.add_argument("--val-docs", type=int, default=5_000, help="Legacy pilot validation rows")
    parser.add_argument("--docs-per-shard", type=int, default=100_000)
    parser.add_argument(
        "--max-docs-per-subset",
        type=int,
        help="Corpus-mode cap per config; omit for the full config",
    )
    parser.add_argument("--tokenizer-dir", type=Path, help="Store NanoChat+BOS token count per document")
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--tokenizer-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Resume corpus mode from materialize_manifest.json")
    parser.add_argument(
        "--recover-orphan-shards",
        action="store_true",
        help="Delete materializer-named parquet shards not committed to the resume manifest",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fast-exit",
        action="store_true",
        help="After a successful corpus run, flush output and bypass native library teardown",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_mode = bool(
        args.corpus
        or args.all_subsets
        or args.resume
        or args.max_docs_per_subset is not None
        or (args.subset is not None and len(args.subset) > 1)
    )
    if corpus_mode:
        _run_corpus(args)
        if args.fast_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    else:
        _run_legacy(args)


if __name__ == "__main__":
    main()
