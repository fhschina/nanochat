"""
Compute tokenizer-based statistics for NanoChat parquet data.

The train/val split follows nanochat.dataset/list_parquet_files convention:
all parquet files except the final sorted shard are train, and the final shard
is validation. Token counts use the active NanoChat tokenizer and include the
BOS token per document by default, matching the pretraining dataloader.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from nanochat.tokenizer import get_tokenizer


QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)


def _parquet_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix == ".parquet" and not p.name.endswith(".tmp"))


def _select_files(data_dir: Path, split: str, num_train_shards: int) -> list[Path]:
    paths = _parquet_files(data_dir)
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    if split == "train":
        selected = paths[:-1]
        if num_train_shards > 0:
            selected = selected[:num_train_shards]
    elif split == "val":
        selected = paths[-1:]
    elif split == "all":
        selected = paths
    else:
        raise ValueError(f"Unknown split: {split}")
    if not selected:
        raise ValueError(f"Split {split!r} selected zero parquet files from {data_dir}")
    return selected


def _quantiles(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {f"p{int(q * 100):02d}": None for q in QUANTILES}
    values = sorted(values)
    n = len(values)
    out: dict[str, int | float | None] = {}
    for q in QUANTILES:
        idx = int(round(q * (n - 1)))
        out[f"p{int(q * 100):02d}"] = values[idx]
    out["mean"] = sum(values) / n
    return out


def _count_tokens(tokenizer, texts: list[str], batch_size: int, num_threads: int, include_bos: bool) -> list[int]:
    prepend = tokenizer.get_bos_token_id() if include_bos else None
    lengths: list[int] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        tokenized = tokenizer.encode(batch, prepend=prepend, num_threads=num_threads)
        lengths.extend(len(tokens) for tokens in tokenized)
    return lengths


def compute_stats(
    paths: list[Path],
    tokenizer,
    tokenizer_batch_size: int,
    tokenizer_threads: int,
    include_bos: bool,
    max_docs: int,
) -> dict:
    docs = 0
    chars = 0
    tokens = 0
    bytes_ = 0
    char_lengths: list[int] = []
    token_lengths: list[int] = []
    files = []
    remaining = None if max_docs < 0 else max_docs

    for path in paths:
        if remaining == 0:
            break
        pf = pq.ParquetFile(path)
        file_docs = 0
        file_chars = 0
        file_tokens = 0

        for rg_idx in range(pf.num_row_groups):
            if remaining == 0:
                break
            table = pf.read_row_group(rg_idx, columns=["text"])
            texts = [(text or "") for text in table.column("text").to_pylist()]
            if remaining is not None and len(texts) > remaining:
                texts = texts[:remaining]

            lengths = [len(text) for text in texts]
            token_lens = _count_tokens(tokenizer, texts, tokenizer_batch_size, tokenizer_threads, include_bos)

            batch_docs = len(texts)
            batch_chars = sum(lengths)
            batch_tokens = sum(token_lens)

            docs += batch_docs
            chars += batch_chars
            tokens += batch_tokens
            file_docs += batch_docs
            file_chars += batch_chars
            file_tokens += batch_tokens
            char_lengths.extend(lengths)
            token_lengths.extend(token_lens)

            if remaining is not None:
                remaining -= batch_docs

        size = path.stat().st_size
        bytes_ += size
        files.append(
            {
                "file": path.name,
                "docs": file_docs,
                "chars": file_chars,
                "tokens": file_tokens,
                "bytes": size,
            }
        )

    return {
        "docs": docs,
        "chars": chars,
        "tokens": tokens,
        "bytes": bytes_,
        "char_length_quantiles": _quantiles(char_lengths),
        "token_length_quantiles": _quantiles(token_lengths),
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute parquet data stats with the active NanoChat tokenizer")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--num-train-shards", type=int, default=-1, help="Limit train files before the val shard; -1 means all train files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=-1, help="Optional cap for smoke runs; -1 means all selected docs")
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--tokenizer-threads", type=int, default=4)
    parser.add_argument("--no-bos", action="store_true", help="Do not include one BOS token per document in token counts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer()
    selected = _select_files(data_dir, args.split, args.num_train_shards)
    stats = compute_stats(
        selected,
        tokenizer=tokenizer,
        tokenizer_batch_size=args.tokenizer_batch_size,
        tokenizer_threads=args.tokenizer_threads,
        include_bos=not args.no_bos,
        max_docs=args.max_docs,
    )
    payload = {
        "data_dir": str(data_dir),
        "split": args.split,
        "num_train_shards": args.num_train_shards,
        "selected_files": [p.name for p in selected],
        "include_bos": not args.no_bos,
        "tokenizer_batch_size": args.tokenizer_batch_size,
        "tokenizer_threads": args.tokenizer_threads,
        "max_docs": args.max_docs,
        **stats,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote data stats: {output}")
    print(f"docs={stats['docs']:,} chars={stats['chars']:,} tokens={stats['tokens']:,}")


if __name__ == "__main__":
    main()
