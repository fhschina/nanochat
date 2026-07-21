"""Rebuild and retain NeMo Curator fuzzy-dedup graph artifacts.

The production exact+fuzzy job may delete MinHash, LSH edge, and connected-
component caches after removal. This identification-only driver recreates those
artifacts without writing another filtered copy of the corpus.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from scripts.build_exact_fuzzy_dedup_dataset import (
    _atomic_write_json,
    _configure_local_ray,
    _make_executor,
    _utc_now,
    _workflow_metadata,
)


MANIFEST_NAME = "fuzzy_audit_artifacts_manifest.json"


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return sum(pq.ParquetFile(file).metadata.num_rows for file in files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retain fuzzy-dedup graph artifacts for audit")
    parser.add_argument("--input-data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected-duplicates", type=int)
    parser.add_argument("--input-blocksize", default="512MiB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--char-ngrams", type=int, default=24)
    parser.add_argument("--num-bands", type=int, default=20)
    parser.add_argument("--minhashes-per-band", type=int, default=13)
    parser.add_argument("--use-64-bit-hash", action="store_true")
    parser.add_argument("--bands-per-iteration", type=int, default=5)
    parser.add_argument("--lsh-num-output-partitions", type=int)
    parser.add_argument("--ray-num-cpus", type=int, default=64)
    parser.add_argument("--ray-temp-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

    args = parse_args()
    args.input_data_dir = args.input_data_dir.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    if args.ray_temp_dir is not None:
        args.ray_temp_dir = args.ray_temp_dir.expanduser().resolve()
    if not args.input_data_dir.is_dir():
        raise FileNotFoundError(args.input_data_dir)
    if args.expected_duplicates is not None and args.expected_duplicates < 0:
        raise ValueError("--expected-duplicates must be non-negative")

    manifest_path = args.work_dir / MANIFEST_NAME
    cache_path = args.work_dir / "fuzzy_cache"
    identification_path = args.work_dir / "fuzzy_identification"
    if manifest_path.is_file() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            print(f"Fuzzy audit artifacts are already complete: {manifest_path}")
            return
        raise FileExistsError(f"Incomplete run exists at {args.work_dir}; pass --overwrite to restart")
    if args.overwrite and args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input_blocksize": args.input_blocksize,
        "seed": args.seed,
        "char_ngrams": args.char_ngrams,
        "num_bands": args.num_bands,
        "minhashes_per_band": args.minhashes_per_band,
        "use_64_bit_hash": args.use_64_bit_hash,
        "bands_per_iteration": args.bands_per_iteration,
        "lsh_num_output_partitions": args.lsh_num_output_partitions,
        "ray_num_cpus": args.ray_num_cpus,
    }
    manifest = {
        "format_version": 1,
        "status": "running",
        "created_at": _utc_now(),
        "input_data_dir": str(args.input_data_dir),
        "work_dir": str(args.work_dir),
        "config": config,
    }
    _atomic_write_json(manifest_path, manifest)

    started = time.time()
    try:
        _configure_local_ray(args)
        workflow = FuzzyDeduplicationWorkflow(
            input_path=str(args.input_data_dir),
            cache_path=str(cache_path),
            output_path=str(identification_path),
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
        metadata = _workflow_metadata(workflow.run(executor=_make_executor()))
        duplicates = int(metadata.get("num_duplicates", 0))
        duplicate_rows = _parquet_rows(identification_path / "FuzzyDuplicateIds")
        component_rows = _parquet_rows(cache_path / "ConnectedComponentsStage")
        if duplicates != duplicate_rows:
            raise RuntimeError(
                f"Curator reported {duplicates:,} duplicate IDs but wrote {duplicate_rows:,} rows"
            )
        if args.expected_duplicates is not None and duplicates != args.expected_duplicates:
            raise RuntimeError(
                f"Expected {args.expected_duplicates:,} duplicate IDs but rebuilt {duplicates:,}"
            )
        manifest.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "elapsed_sec": time.time() - started,
                "metadata": metadata,
                "artifacts": {
                    "duplicate_ids_path": str(identification_path / "FuzzyDuplicateIds"),
                    "id_generator_path": str(identification_path / "fuzzy_id_generator.json"),
                    "component_path": str(cache_path / "ConnectedComponentsStage"),
                    "edge_path": str(cache_path / "BucketsToEdgesStage"),
                    "duplicate_rows": duplicate_rows,
                    "component_vertex_rows": component_rows,
                },
            }
        )
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "elapsed_sec": time.time() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_write_json(manifest_path, manifest)
        raise

    _atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
