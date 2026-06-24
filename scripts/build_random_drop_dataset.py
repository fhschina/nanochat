"""
Build a random-drop parquet directory for NanoChat controls.

This is a non-semantic control for SemDeDup experiments: remove a deterministic
uniform random subset of train documents while keeping the validation shard
unchanged. The output schema is NanoChat-compatible parquet with only a text
column, plus analysis artifacts that mirror the SemDeDup builder.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_semdedup_dataset import (
    ROW_GROUP_SIZE,
    _parquet_files,
    _quantiles,
    _reset_dir,
    _select_train_val,
    _token_lengths,
)


def _default_base_dir() -> Path:
    return Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat"))


def _default_dataset_tag() -> str:
    return os.environ.get("DATASET_TAG", "climbmix")


def _total_docs(paths: list[Path], max_docs: int) -> int:
    remaining = None if max_docs < 0 else max_docs
    total = 0
    for path in paths:
        if remaining == 0:
            break
        rows = pq.ParquetFile(path).metadata.num_rows
        if remaining is not None:
            rows = min(rows, remaining)
            remaining -= rows
        total += rows
    return total


def _sample_positions(total_docs: int, sample_size: int, seed: int, excluded: set[int] | None = None) -> set[int]:
    if sample_size <= 0 or total_docs <= 0:
        return set()
    excluded = excluded or set()
    available = total_docs - len(excluded)
    sample_size = min(sample_size, available)
    rng = random.Random(seed)
    out: set[int] = set()
    while len(out) < sample_size:
        pos = rng.randrange(total_docs)
        if pos in excluded or pos in out:
            continue
        out.add(pos)
    return out


def _length_stats(char_lengths: list[int], token_lengths: list[int] | None = None) -> dict:
    stats = {"char_length_quantiles": _quantiles(char_lengths)}
    if token_lengths is not None:
        stats["token_length_quantiles"] = _quantiles(token_lengths)
    return stats


def _write_jsonl_record(handle, doc_id: str, text: str, tokenizer, tokenizer_threads: int) -> None:
    token_len = None
    if tokenizer is not None:
        token_len = len(tokenizer.encode(text, prepend=tokenizer.get_bos_token_id(), num_threads=tokenizer_threads))
    handle.write(json.dumps({
        "doc_id": doc_id,
        "source_file": doc_id.split(":", 1)[0],
        "char_len": len(text),
        "token_len": token_len,
        "text_preview": " ".join(text[:1000].split()),
    }, ensure_ascii=False) + "\n")


def build_random_drop(args: argparse.Namespace) -> dict:
    args.input_data_dir = args.input_data_dir.expanduser().resolve()
    args.output_data_dir = args.output_data_dir.expanduser().resolve()
    args.analysis_output_dir = args.analysis_output_dir.expanduser().resolve()
    args.analysis_output_dir.mkdir(parents=True, exist_ok=True)
    _reset_dir(args.output_data_dir, overwrite=args.overwrite)

    tokenizer = None
    if not args.skip_token_stats:
        from nanochat.tokenizer import get_tokenizer

        tokenizer = get_tokenizer()

    train_paths, val_path = _select_train_val(args.input_data_dir, args.num_train_shards)
    total_docs = _total_docs(train_paths, args.max_docs)
    if args.target_removed_docs < 0:
        raise ValueError("--target-removed-docs must be >= 0")
    if args.target_removed_docs >= total_docs:
        raise ValueError(f"Cannot remove {args.target_removed_docs} docs from only {total_docs} selected docs")

    rng = random.Random(args.random_seed)
    removed_positions = set(rng.sample(range(total_docs), args.target_removed_docs))
    removed_sample_positions = set(rng.sample(sorted(removed_positions), min(args.audit_sample_size, len(removed_positions))))
    kept_sample_positions = _sample_positions(total_docs, args.audit_sample_size, args.audit_seed, excluded=removed_positions)

    kept_ids_path = args.analysis_output_dir / "kept_ids.txt"
    removed_ids_path = args.analysis_output_dir / "removed_ids.txt"
    kept_samples_path = args.analysis_output_dir / "kept_samples.jsonl"
    removed_samples_path = args.analysis_output_dir / "removed_samples.jsonl"

    input_docs = output_docs = 0
    input_chars = output_chars = 0
    input_tokens = output_tokens = None if tokenizer is None else 0
    input_char_lengths: list[int] = []
    output_char_lengths: list[int] = []
    input_token_lengths: list[int] = []
    output_token_lengths: list[int] = []
    input_files = []
    output_files = []

    global_pos = 0
    out_file_idx = 0
    remaining = None if args.max_docs < 0 else args.max_docs

    with kept_ids_path.open("w", encoding="utf-8") as kept_ids_f, \
        removed_ids_path.open("w", encoding="utf-8") as removed_ids_f, \
        kept_samples_path.open("w", encoding="utf-8") as kept_samples_f, \
        removed_samples_path.open("w", encoding="utf-8") as removed_samples_f:

        for src_path in train_paths:
            if remaining == 0:
                break
            pf = pq.ParquetFile(src_path)
            dst_path = args.output_data_dir / f"shard_{out_file_idx:05d}.parquet"
            writer = None
            file_input_docs = file_output_docs = 0
            file_input_chars = file_output_chars = 0
            file_input_tokens = file_output_tokens = None if tokenizer is None else 0
            row_offset = 0
            try:
                for rg_idx in range(pf.num_row_groups):
                    if remaining == 0:
                        break
                    table = pf.read_row_group(rg_idx, columns=["text"])
                    texts = [(text or "") for text in table.column("text").to_pylist()]
                    if remaining is not None and len(texts) > remaining:
                        texts = texts[:remaining]

                    char_lengths = [len(text) for text in texts]
                    token_lengths = _token_lengths(tokenizer, texts, args.tokenizer_batch_size, args.tokenizer_threads) if tokenizer is not None else []

                    keep_texts = []
                    for idx, text in enumerate(texts):
                        doc_id = f"{src_path.name}:{row_offset + idx}"
                        char_len = char_lengths[idx]
                        token_len = token_lengths[idx] if tokenizer is not None else None
                        is_removed = global_pos in removed_positions

                        input_docs += 1
                        input_chars += char_len
                        file_input_docs += 1
                        file_input_chars += char_len
                        input_char_lengths.append(char_len)
                        if tokenizer is not None:
                            input_tokens += token_len
                            file_input_tokens += token_len
                            input_token_lengths.append(token_len)

                        if is_removed:
                            removed_ids_f.write(doc_id + "\n")
                            if global_pos in removed_sample_positions:
                                _write_jsonl_record(removed_samples_f, doc_id, text, tokenizer, args.tokenizer_threads)
                        else:
                            kept_ids_f.write(doc_id + "\n")
                            keep_texts.append(text)
                            output_docs += 1
                            output_chars += char_len
                            file_output_docs += 1
                            file_output_chars += char_len
                            output_char_lengths.append(char_len)
                            if tokenizer is not None:
                                output_tokens += token_len
                                file_output_tokens += token_len
                                output_token_lengths.append(token_len)
                            if global_pos in kept_sample_positions:
                                _write_jsonl_record(kept_samples_f, doc_id, text, tokenizer, args.tokenizer_threads)
                        global_pos += 1

                    if keep_texts:
                        out = pa.table({"text": keep_texts})
                        if writer is None:
                            writer = pq.ParquetWriter(dst_path, out.schema, compression="zstd")
                        writer.write_table(out, row_group_size=ROW_GROUP_SIZE)

                    row_offset += len(texts)
                    if remaining is not None:
                        remaining -= len(texts)
            finally:
                if writer is not None:
                    writer.close()

            input_file = {"file": src_path.name, "docs": file_input_docs, "chars": file_input_chars, "bytes": src_path.stat().st_size}
            if tokenizer is not None:
                input_file["tokens"] = file_input_tokens
            input_files.append(input_file)
            if dst_path.exists():
                output_file = {"file": dst_path.name, "source_file": src_path.name, "docs": file_output_docs, "chars": file_output_chars, "bytes": dst_path.stat().st_size}
                if tokenizer is not None:
                    output_file["tokens"] = file_output_tokens
                output_files.append(output_file)
                out_file_idx += 1

    val_output_path = args.output_data_dir / f"shard_{out_file_idx:05d}.parquet"
    shutil.copy2(val_path, val_output_path)

    input_train = {
        "docs": input_docs,
        "chars": input_chars,
        "bytes": sum(item["bytes"] for item in input_files),
        "files": input_files,
        **_length_stats(input_char_lengths, input_token_lengths if tokenizer is not None else None),
    }
    output_train = {
        "docs": output_docs,
        "chars": output_chars,
        "bytes": sum(item["bytes"] for item in output_files),
        "files": output_files,
        **_length_stats(output_char_lengths, output_token_lengths if tokenizer is not None else None),
    }
    if tokenizer is not None:
        input_train["tokens"] = input_tokens
        output_train["tokens"] = output_tokens

    from scripts.build_semdedup_dataset import _count_texts

    output_val = _count_texts([val_output_path], tokenizer, args.tokenizer_batch_size, args.tokenizer_threads)
    removed_docs = input_docs - output_docs
    removed_chars = input_chars - output_chars
    removed_tokens = None if tokenizer is None else input_tokens - output_tokens

    manifest = {
        "backend": "random-drop",
        "semantic_dedup": False,
        "input_data_dir": str(args.input_data_dir),
        "output_data_dir": str(args.output_data_dir),
        "analysis_output_dir": str(args.analysis_output_dir),
        "train_files": [p.name for p in train_paths],
        "val_file": val_path.name,
        "output_val_file": val_output_path.name,
        "random_drop_config": {
            "random_seed": args.random_seed,
            "audit_seed": args.audit_seed,
            "target_removed_docs": args.target_removed_docs,
            "selected_docs": total_docs,
        },
        "token_stats": {
            "enabled": tokenizer is not None,
            "include_bos_per_document": True,
            "tokenizer_batch_size": args.tokenizer_batch_size,
            "tokenizer_threads": args.tokenizer_threads,
        },
        "input_train": input_train,
        "output_train": output_train,
        "output_val": output_val,
        "removed_docs": removed_docs,
        "removed_chars": removed_chars,
        "removed_tokens": removed_tokens,
        "keep_ratio_docs": output_docs / input_docs if input_docs else None,
        "keep_ratio_chars": output_chars / input_chars if input_chars else None,
        "keep_ratio_tokens": output_tokens / input_tokens if tokenizer is not None and input_tokens else None,
        "analysis_artifacts": {
            "doc_id_available": True,
            "kept_ids_path": str(kept_ids_path),
            "removed_ids_path": str(removed_ids_path),
            "kept_samples_path": str(kept_samples_path),
            "removed_samples_path": str(removed_samples_path),
            "kept_ids": output_docs,
            "removed_ids": removed_docs,
            "kept_sample_count": len(kept_sample_positions),
            "removed_sample_count": len(removed_sample_positions),
        },
    }
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path = args.output_data_dir / "random_drop_manifest.json"
    manifest_path.write_text(manifest_json)
    analysis_manifest_path = args.analysis_output_dir / "random_drop_manifest.json"
    if analysis_manifest_path != manifest_path:
        analysis_manifest_path.write_text(manifest_json)
    print(f"Wrote random-drop manifest: {manifest_path}")
    print(f"Random-drop keep ratio: docs={manifest['keep_ratio_docs']:.4f} chars={manifest['keep_ratio_chars']:.4f}")
    if manifest["keep_ratio_tokens"] is not None:
        print(f"Random-drop token keep ratio: {manifest['keep_ratio_tokens']:.4f}")
    return manifest


def parse_args() -> argparse.Namespace:
    base_dir = _default_base_dir()
    dataset_tag = _default_dataset_tag()
    parser = argparse.ArgumentParser(description="Build a random-drop NanoChat control data directory")
    parser.add_argument("--input-data-dir", type=Path, default=base_dir / f"base_data_{dataset_tag}")
    parser.add_argument("--output-data-dir", type=Path, default=None)
    parser.add_argument("--analysis-output-dir", type=Path, default=None)
    parser.add_argument("--num-train-shards", type=int, default=8, help="Number of train shards to process; -1 means all available train shards")
    parser.add_argument("--max-docs", type=int, default=-1, help="Optional cap for smoke/pilot runs; -1 means no cap")
    parser.add_argument("--target-removed-docs", type=int, default=291374, help="Exact number of selected train docs to remove")
    parser.add_argument("--random-seed", type=int, default=9001, help="Seed for selecting removed docs")
    parser.add_argument("--audit-sample-size", type=int, default=1000, help="Number of kept/removed docs to sample for manual inspection")
    parser.add_argument("--audit-seed", type=int, default=1337, help="Seed for deterministic kept audit samples")
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--tokenizer-threads", type=int, default=4)
    parser.add_argument("--skip-token-stats", action="store_true", help="Skip tokenizer-based token stats for fast plumbing checks")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_data_dir is None:
        args.output_data_dir = base_dir / f"base_data_{dataset_tag}_randomdrop_drop{args.target_removed_docs}_seed{args.random_seed}_n{args.num_train_shards}"
    if args.analysis_output_dir is None:
        args.analysis_output_dir = args.output_data_dir
    return args


def main() -> None:
    args = parse_args()
    build_random_drop(args)


if __name__ == "__main__":
    main()
