"""Create a tiny corpus with known exact and near-duplicate documents."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pyarrow as pa

from scripts.materialize_fineweb_edu_fortified import _load_tokenizer, _save_manifest, _write_corpus_shard


def _documents() -> list[str]:
    exact = (
        "Exact duplicate fixture. Distributed data curation should preserve stable identifiers, "
        "reproducible counts, and auditable outputs. "
    ) * 12
    fuzzy_a = (
        "Fuzzy duplicate fixture. A careful data pipeline hashes character windows, groups likely matches, "
        "and keeps one representative from every connected component. "
    ) * 16 + "This is revision alpha."
    fuzzy_b = (
        "Fuzzy duplicate fixture. A careful data pipeline hashes character windows, groups likely matches, "
        "and keeps one representative from every connected component. "
    ) * 16 + "This is revision beta."
    unique_topics = [
        "algebraic topology and compact manifolds",
        "marine navigation and celestial observations",
        "renaissance printing and movable type",
        "semiconductor fabrication and photolithography",
        "classical music counterpoint and harmony",
        "urban hydrology and watershed modeling",
        "comparative linguistics and sound change",
        "public health surveillance and sampling",
    ]
    unique = [
        (f"Unique fixture {index}: {topic}. " * 28) + f"Terminal marker {index}."
        for index, topic in enumerate(unique_topics)
    ]
    return [exact, exact, exact, fuzzy_a, fuzzy_b, *unique]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    tokenizer = _load_tokenizer(args.tokenizer_dir)
    records = []
    for index, text in enumerate(_documents()):
        records.append(
            {
                "id": f"doc-{index:04d}",
                "dump": "synthetic-smoke",
                "url": f"https://example.invalid/{index}",
                "count": 1,
                "token_count": len(text.split()),
                "text": text,
            }
        )

    shard = _write_corpus_shard(
        output_dir / "shard_000000.parquet",
        pa.Table.from_pylist(records),
        tokenizer,
        tokenizer_batch_size=16,
        tokenizer_threads=4,
        subset="synthetic-smoke",
        source_row_start=0,
    )
    manifest = {
        "format_version": 2,
        "mode": "corpus",
        "status": "complete",
        "dataset": "synthetic-smoke",
        "split": "train",
        "subsets": ["synthetic-smoke"],
        "docs_per_shard": len(records),
        "max_docs_per_subset": len(records),
        "tokenizer_dir": str(args.tokenizer_dir.expanduser().resolve()),
        "include_bos": True,
        "output_dir": str(output_dir),
        "shards": [shard],
        "subset_stats": {
            "synthetic-smoke": {
                "status": "complete",
                "capped": True,
                "docs": shard["docs"],
                "chars": shard["chars"],
                "hf_tokens": shard["hf_tokens"],
                "nanochat_tokens": shard["nanochat_tokens"],
                "bytes": shard["bytes"],
            }
        },
    }
    _save_manifest(output_dir / "materialize_manifest.json", manifest)
    print(f"Wrote smoke fixture: {output_dir} ({len(records)} docs; 2 known exact removals)")


if __name__ == "__main__":
    main()
