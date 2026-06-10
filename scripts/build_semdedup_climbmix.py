"""
Build a SemDeDup-reduced ClimbMix parquet directory for nanochat.

The default backend calls NeMo Curator's TextSemanticDeduplicationWorkflow and
then normalizes its output to nanochat's expected parquet schema: one `text`
column, with the original ClimbMix validation shard copied as the final sorted
shard. The `exact-smoke` backend is dependency-free and exists only to test the
nanochat data plumbing on a tiny sample; it is not a semantic deduplication
experiment.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROW_GROUP_SIZE = 1024


def _default_base_dir() -> Path:
    return Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat"))


def _eps_slug(eps: float) -> str:
    return str(eps).replace(".", "p")


def _parquet_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix == ".parquet" and not p.name.endswith(".tmp"))


def _count_texts(paths: list[Path]) -> dict:
    docs = 0
    chars = 0
    bytes_ = 0
    files = []
    for path in paths:
        pf = pq.ParquetFile(path)
        file_docs = 0
        file_chars = 0
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["text"])
            texts = table.column("text").to_pylist()
            file_docs += len(texts)
            file_chars += sum(len(text or "") for text in texts)
        size = path.stat().st_size
        docs += file_docs
        chars += file_chars
        bytes_ += size
        files.append({"file": path.name, "docs": file_docs, "chars": file_chars, "bytes": size})
    return {"docs": docs, "chars": chars, "bytes": bytes_, "files": files}


def _select_train_val(input_data_dir: Path, num_train_shards: int) -> tuple[list[Path], Path]:
    paths = _parquet_files(input_data_dir)
    if len(paths) < 2:
        raise ValueError(f"Need at least one train parquet plus one val parquet in {input_data_dir}")
    val_path = paths[-1]
    train_paths = paths[:-1]
    if num_train_shards > 0:
        train_paths = train_paths[:num_train_shards]
    if not train_paths:
        raise ValueError("No train shards selected")
    return train_paths, val_path


def _reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _stage_train_inputs(train_paths: list[Path], staged_dir: Path, max_docs: int, overwrite: bool) -> dict:
    _reset_dir(staged_dir, overwrite=overwrite)
    docs = 0
    chars = 0
    bytes_ = 0
    remaining = None if max_docs < 0 else max_docs
    files = []

    for shard_idx, src_path in enumerate(train_paths):
        if remaining == 0:
            break
        dst_path = staged_dir / f"shard_{shard_idx:05d}.parquet"
        pf = pq.ParquetFile(src_path)
        writer = None
        file_docs = 0
        file_chars = 0
        row_offset = 0
        try:
            for rg_idx in range(pf.num_row_groups):
                if remaining == 0:
                    break
                table = pf.read_row_group(rg_idx, columns=["text"])
                texts = table.column("text").to_pylist()
                if remaining is not None and len(texts) > remaining:
                    texts = texts[:remaining]
                ids = [f"{src_path.name}:{row_offset + i}" for i in range(len(texts))]
                row_offset += len(texts)
                out = pa.table({"doc_id": ids, "text": texts})
                if writer is None:
                    writer = pq.ParquetWriter(dst_path, out.schema, compression="zstd")
                writer.write_table(out, row_group_size=ROW_GROUP_SIZE)
                batch_chars = sum(len(text or "") for text in texts)
                docs += len(texts)
                chars += batch_chars
                file_docs += len(texts)
                file_chars += batch_chars
                if remaining is not None:
                    remaining -= len(texts)
        finally:
            if writer is not None:
                writer.close()
        if file_docs > 0:
            size = dst_path.stat().st_size
            bytes_ += size
            files.append({"file": dst_path.name, "source_file": src_path.name, "docs": file_docs, "chars": file_chars, "bytes": size})

    if docs == 0:
        raise ValueError("Staging produced zero documents")
    return {"docs": docs, "chars": chars, "bytes": bytes_, "files": files}


def _preinit_local_ray(args) -> None:
    if args.no_ray_preinit:
        return
    try:
        import ray
    except ImportError:
        return
    if ray.is_initialized():
        return
    ray_temp_dir = args.ray_temp_dir.expanduser().resolve()
    ray_temp_dir.mkdir(parents=True, exist_ok=True)
    ray.init(
        address="local",
        _temp_dir=str(ray_temp_dir),
        include_dashboard=False,
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
        },
    )
    print(f"Initialized isolated local Ray at {ray.get_runtime_context().gcs_address} (temp_dir={ray_temp_dir})")


def _run_curator(args, staged_input_dir: Path, curator_output_dir: Path) -> dict:
    try:
        from nemo_curator.stages.text.deduplication.semantic import TextSemanticDeduplicationWorkflow
    except ImportError as exc:
        raise RuntimeError(
            "NeMo Curator semantic deduplication is not installed in this environment. "
            "Install the text CUDA extra in a Curator-capable environment, for example: "
            "uv pip install --extra-index-url https://pypi.nvidia.com 'nemo-curator[text_cuda12]'"
        ) from exc

    _preinit_local_ray(args)

    kwargs = {
        "input_path": str(staged_input_dir),
        "output_path": str(curator_output_dir),
        "cache_path": str(args.cache_dir / "curator_cache"),
        "text_field": "text",
        "id_field": "doc_id",
        "model_identifier": args.model_identifier,
        "n_clusters": args.n_clusters,
        "eps": args.eps,
        "distance_metric": args.distance_metric,
        "which_to_keep": args.which_to_keep,
        "pairwise_batch_size": args.pairwise_batch_size,
        "perform_removal": True,
    }
    if args.embedding_max_chars is not None:
        kwargs["embedding_max_chars"] = args.embedding_max_chars
    if args.model_cache_dir:
        kwargs["model_cache_dir"] = args.model_cache_dir

    workflow = TextSemanticDeduplicationWorkflow(**kwargs)
    t0 = time.time()
    result = workflow.run()
    elapsed = time.time() - t0
    metadata = getattr(result, "metadata", None)
    return {"elapsed_sec": elapsed, "metadata": metadata}


def _run_exact_smoke(staged_input_dir: Path, dedup_dir: Path, overwrite: bool) -> dict:
    _reset_dir(dedup_dir, overwrite=overwrite)
    seen = set()
    removed = 0
    kept = 0
    writer = None
    out_path = dedup_dir / "shard_00000.parquet"
    try:
        for src_path in _parquet_files(staged_input_dir):
            pf = pq.ParquetFile(src_path)
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=["doc_id", "text"])
                ids = table.column("doc_id").to_pylist()
                texts = table.column("text").to_pylist()
                keep_ids = []
                keep_texts = []
                for doc_id, text in zip(ids, texts):
                    digest = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
                    if digest in seen:
                        removed += 1
                        continue
                    seen.add(digest)
                    keep_ids.append(doc_id)
                    keep_texts.append(text)
                if keep_texts:
                    out = pa.table({"doc_id": keep_ids, "text": keep_texts})
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, out.schema, compression="zstd")
                    writer.write_table(out, row_group_size=ROW_GROUP_SIZE)
                    kept += len(keep_texts)
    finally:
        if writer is not None:
            writer.close()
    if kept == 0:
        raise ValueError("exact-smoke backend kept zero documents")
    return {"kept": kept, "removed": removed}


def _find_curator_dedup_dir(curator_output_dir: Path) -> Path:
    candidates = [curator_output_dir / "deduplicated", curator_output_dir]
    candidates.extend(p for p in curator_output_dir.rglob("deduplicated") if p.is_dir())
    for candidate in candidates:
        if candidate.exists() and list(candidate.rglob("*.parquet")):
            return candidate
    raise FileNotFoundError(f"Could not find deduplicated parquet output under {curator_output_dir}")


def _normalize_to_nanochat(dedup_dir: Path, val_path: Path, output_data_dir: Path, overwrite: bool) -> dict:
    _reset_dir(output_data_dir, overwrite=overwrite)
    dedup_paths = sorted(dedup_dir.rglob("*.parquet"))
    if not dedup_paths:
        raise FileNotFoundError(f"No parquet files found in {dedup_dir}")

    train_out_paths = []
    for idx, src_path in enumerate(dedup_paths):
        dst_path = output_data_dir / f"shard_{idx:05d}.parquet"
        pf = pq.ParquetFile(src_path)
        writer = None
        try:
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=["text"])
                out = pa.table({"text": table.column("text")})
                if writer is None:
                    writer = pq.ParquetWriter(dst_path, out.schema, compression="zstd")
                writer.write_table(out, row_group_size=ROW_GROUP_SIZE)
        finally:
            if writer is not None:
                writer.close()
        if dst_path.exists():
            train_out_paths.append(dst_path)

    if not train_out_paths:
        raise ValueError("Normalized train output contains zero parquet files")

    shutil.copy2(val_path, output_data_dir / val_path.name)
    train_stats = _count_texts(train_out_paths)
    val_stats = _count_texts([output_data_dir / val_path.name])
    return {"train": train_stats, "val": val_stats}


def parse_args() -> argparse.Namespace:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Build a SemDeDup-reduced ClimbMix data directory for nanochat")
    parser.add_argument("--input-data-dir", type=Path, default=base_dir / "base_data_climbmix")
    parser.add_argument("--output-data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--num-train-shards", type=int, default=8, help="Number of train shards to process; -1 means all available train shards")
    parser.add_argument("--max-docs", type=int, default=-1, help="Optional cap for smoke/pilot runs; -1 means no cap")
    parser.add_argument("--backend", choices=["curator", "exact-smoke"], default="curator")
    parser.add_argument("--model-identifier", type=str, default="google/embeddinggemma-300m")
    parser.add_argument("--model-cache-dir", type=str, default="")
    parser.add_argument("--embedding-max-chars", type=int, default=None)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.07)
    parser.add_argument("--distance-metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--which-to-keep", choices=["hard", "easy", "random"], default="hard")
    parser.add_argument("--pairwise-batch-size", type=int, default=1024)
    parser.add_argument("--ray-temp-dir", type=Path, default=None, help="Ray temp dir for an isolated local Curator/Xenna run")
    parser.add_argument("--no-ray-preinit", action="store_true", help="Disable isolated local Ray pre-initialization")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_data_dir is None:
        args.output_data_dir = base_dir / f"base_data_climbmix_semdedup_eps{_eps_slug(args.eps)}_n{args.num_train_shards}"
    if args.cache_dir is None:
        args.cache_dir = base_dir / "semdedup_cache" / f"eps{_eps_slug(args.eps)}_n{args.num_train_shards}"
    if args.ray_temp_dir is None:
        args.ray_temp_dir = args.cache_dir / "ray"
    return args


def main() -> None:
    args = parse_args()
    args.input_data_dir = args.input_data_dir.expanduser().resolve()
    args.output_data_dir = args.output_data_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    train_paths, val_path = _select_train_val(args.input_data_dir, args.num_train_shards)
    staged_input_dir = args.cache_dir / "input_with_ids"
    curator_output_dir = args.cache_dir / "curator_output"
    exact_output_dir = args.cache_dir / "exact_smoke_deduplicated"

    print(f"Input train shards: {len(train_paths)} from {args.input_data_dir}")
    print(f"Validation shard: {val_path.name}")
    print(f"Output data dir: {args.output_data_dir}")
    print(f"Cache dir: {args.cache_dir}")

    input_stats = _stage_train_inputs(train_paths, staged_input_dir, args.max_docs, overwrite=args.overwrite)
    print(f"Staged {input_stats['docs']:,} docs / {input_stats['chars']:,} chars")

    backend_result = None
    if args.backend == "curator":
        _reset_dir(curator_output_dir, overwrite=args.overwrite)
        backend_result = _run_curator(args, staged_input_dir, curator_output_dir)
        dedup_dir = _find_curator_dedup_dir(curator_output_dir)
    else:
        backend_result = _run_exact_smoke(staged_input_dir, exact_output_dir, overwrite=args.overwrite)
        dedup_dir = exact_output_dir

    output_stats = _normalize_to_nanochat(dedup_dir, val_path, args.output_data_dir, overwrite=args.overwrite)
    train_out = output_stats["train"]
    keep_ratio_docs = train_out["docs"] / input_stats["docs"] if input_stats["docs"] else None
    keep_ratio_chars = train_out["chars"] / input_stats["chars"] if input_stats["chars"] else None

    manifest = {
        "backend": args.backend,
        "semantic_dedup": args.backend == "curator",
        "note": "exact-smoke is only a plumbing smoke test and is not SemDeDup." if args.backend == "exact-smoke" else "",
        "input_data_dir": str(args.input_data_dir),
        "output_data_dir": str(args.output_data_dir),
        "cache_dir": str(args.cache_dir),
        "train_files": [p.name for p in train_paths],
        "val_file": val_path.name,
        "curator_config": {
            "model_identifier": args.model_identifier,
            "n_clusters": args.n_clusters,
            "eps": args.eps,
            "distance_metric": args.distance_metric,
            "which_to_keep": args.which_to_keep,
            "pairwise_batch_size": args.pairwise_batch_size,
            "embedding_max_chars": args.embedding_max_chars,
        },
        "input_train": input_stats,
        "output_train": train_out,
        "output_val": output_stats["val"],
        "keep_ratio_docs": keep_ratio_docs,
        "keep_ratio_chars": keep_ratio_chars,
        "backend_result": backend_result,
    }
    manifest_path = args.output_data_dir / "semdedup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote manifest: {manifest_path}")
    print(f"Train keep ratio: docs={keep_ratio_docs:.4f} chars={keep_ratio_chars:.4f}")


if __name__ == "__main__":
    main()
