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
import random
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROW_GROUP_SIZE = 1024
QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)


def _default_base_dir() -> Path:
    return Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat"))


def _eps_slug(eps: float) -> str:
    return str(eps).replace(".", "p")


def _parquet_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix == ".parquet" and not p.name.endswith(".tmp"))


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


def _token_lengths(tokenizer, texts: list[str], batch_size: int, num_threads: int) -> list[int]:
    if tokenizer is None:
        return []
    bos = tokenizer.get_bos_token_id()
    lengths: list[int] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        tokenized = tokenizer.encode(batch, prepend=bos, num_threads=num_threads)
        lengths.extend(len(tokens) for tokens in tokenized)
    return lengths


def _count_texts(paths: list[Path], tokenizer=None, tokenizer_batch_size: int = 128, tokenizer_threads: int = 4) -> dict:
    docs = 0
    chars = 0
    tokens = 0 if tokenizer is not None else None
    bytes_ = 0
    files = []
    char_lengths: list[int] = []
    token_lengths_all: list[int] = []

    for path in paths:
        pf = pq.ParquetFile(path)
        file_docs = 0
        file_chars = 0
        file_tokens = 0 if tokenizer is not None else None
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["text"])
            texts = [(text or "") for text in table.column("text").to_pylist()]
            lengths = [len(text) for text in texts]
            token_lengths = _token_lengths(tokenizer, texts, tokenizer_batch_size, tokenizer_threads)

            file_docs += len(texts)
            file_chars += sum(lengths)
            char_lengths.extend(lengths)
            if tokenizer is not None:
                batch_tokens = sum(token_lengths)
                file_tokens += batch_tokens
                tokens += batch_tokens
                token_lengths_all.extend(token_lengths)

        size = path.stat().st_size
        docs += file_docs
        chars += file_chars
        bytes_ += size
        file_stats = {"file": path.name, "docs": file_docs, "chars": file_chars, "bytes": size}
        if tokenizer is not None:
            file_stats["tokens"] = file_tokens
        files.append(file_stats)

    stats = {
        "docs": docs,
        "chars": chars,
        "bytes": bytes_,
        "files": files,
        "char_length_quantiles": _quantiles(char_lengths),
    }
    if tokenizer is not None:
        stats["tokens"] = tokens
        stats["token_length_quantiles"] = _quantiles(token_lengths_all)
    return stats


def _column_names(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def _collect_doc_ids(paths: list[Path]) -> set[str] | None:
    ids: set[str] = set()
    for path in paths:
        if "doc_id" not in _column_names(path):
            return None
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["doc_id"])
            ids.update(str(doc_id) for doc_id in table.column("doc_id").to_pylist())
    return ids


def _write_id_file(path: Path, ids: set[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc_id in sorted(ids):
            f.write(f"{doc_id}\n")


def _sample_ids(ids: set[str], sample_size: int, seed: int) -> set[str]:
    if sample_size <= 0 or not ids:
        return set()
    rng = random.Random(seed)
    ordered = sorted(ids)
    if len(ordered) <= sample_size:
        return set(ordered)
    return set(rng.sample(ordered, sample_size))


def _write_sample_records(
    staged_paths: list[Path],
    sample_ids: set[str],
    output_path: Path,
    tokenizer=None,
    tokenizer_batch_size: int = 128,
    tokenizer_threads: int = 4,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wanted = set(sample_ids)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        if not wanted:
            return 0
        for path in staged_paths:
            if "doc_id" not in _column_names(path):
                continue
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=["doc_id", "text"])
                ids = [str(doc_id) for doc_id in table.column("doc_id").to_pylist()]
                texts = [(text or "") for text in table.column("text").to_pylist()]
                for doc_id, text in zip(ids, texts):
                    if doc_id not in wanted:
                        continue
                    token_len = None
                    if tokenizer is not None:
                        token_len = len(tokenizer.encode(text, prepend=tokenizer.get_bos_token_id(), num_threads=tokenizer_threads))
                    preview = " ".join(text[:1000].split())
                    rec = {
                        "doc_id": doc_id,
                        "source_file": doc_id.split(":", 1)[0],
                        "char_len": len(text),
                        "token_len": token_len,
                        "text_preview": preview,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    wanted.remove(doc_id)
                if not wanted:
                    return written
    return written


def _write_analysis_artifacts(
    staged_paths: list[Path],
    dedup_dir: Path,
    analysis_dir: Path,
    tokenizer,
    audit_sample_size: int,
    audit_seed: int,
    tokenizer_batch_size: int,
    tokenizer_threads: int,
) -> dict:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    dedup_paths = sorted(dedup_dir.rglob("*.parquet"))
    kept_ids = _collect_doc_ids(dedup_paths)
    input_ids = _collect_doc_ids(staged_paths)
    if kept_ids is None or input_ids is None:
        return {
            "doc_id_available": False,
            "reason": "doc_id column missing from staged input or deduplicated output",
        }

    removed_ids = input_ids - kept_ids
    kept_ids_path = analysis_dir / "kept_ids.txt"
    removed_ids_path = analysis_dir / "removed_ids.txt"
    _write_id_file(kept_ids_path, kept_ids)
    _write_id_file(removed_ids_path, removed_ids)

    kept_sample = _sample_ids(kept_ids, audit_sample_size, audit_seed)
    removed_sample = _sample_ids(removed_ids, audit_sample_size, audit_seed + 1)
    kept_samples_path = analysis_dir / "kept_samples.jsonl"
    removed_samples_path = analysis_dir / "removed_samples.jsonl"
    kept_written = _write_sample_records(
        staged_paths, kept_sample, kept_samples_path, tokenizer, tokenizer_batch_size, tokenizer_threads
    )
    removed_written = _write_sample_records(
        staged_paths, removed_sample, removed_samples_path, tokenizer, tokenizer_batch_size, tokenizer_threads
    )

    return {
        "doc_id_available": True,
        "kept_ids_path": str(kept_ids_path),
        "removed_ids_path": str(removed_ids_path),
        "kept_samples_path": str(kept_samples_path),
        "removed_samples_path": str(removed_samples_path),
        "kept_ids": len(kept_ids),
        "removed_ids": len(removed_ids),
        "kept_sample_count": kept_written,
        "removed_sample_count": removed_written,
    }


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


def _normalize_to_nanochat(
    dedup_dir: Path,
    val_path: Path,
    output_data_dir: Path,
    overwrite: bool,
    tokenizer=None,
    tokenizer_batch_size: int = 128,
    tokenizer_threads: int = 4,
) -> dict:
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

    val_output_path = output_data_dir / f"shard_{len(train_out_paths):05d}.parquet"
    shutil.copy2(val_path, val_output_path)
    train_stats = _count_texts(train_out_paths, tokenizer, tokenizer_batch_size, tokenizer_threads)
    val_stats = _count_texts([val_output_path], tokenizer, tokenizer_batch_size, tokenizer_threads)
    return {"train": train_stats, "val": val_stats, "val_output_file": val_output_path.name}


def parse_args() -> argparse.Namespace:
    base_dir = _default_base_dir()
    parser = argparse.ArgumentParser(description="Build a SemDeDup-reduced ClimbMix data directory for nanochat")
    parser.add_argument("--input-data-dir", type=Path, default=base_dir / "base_data_climbmix")
    parser.add_argument("--output-data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--analysis-output-dir", type=Path, default=None, help="Directory for SemDeDup audit artifacts and an extra manifest copy")
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
    parser.add_argument("--audit-sample-size", type=int, default=100, help="Number of kept/removed docs to sample for manual inspection")
    parser.add_argument("--audit-seed", type=int, default=1337, help="Seed for deterministic audit samples")
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--tokenizer-threads", type=int, default=4)
    parser.add_argument("--skip-token-stats", action="store_true", help="Skip tokenizer-based token stats for fast plumbing checks")
    parser.add_argument("--ray-temp-dir", type=Path, default=None, help="Ray temp dir for an isolated local Curator/Xenna run")
    parser.add_argument("--no-ray-preinit", action="store_true", help="Disable isolated local Ray pre-initialization")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_data_dir is None:
        args.output_data_dir = base_dir / f"base_data_climbmix_semdedup_eps{_eps_slug(args.eps)}_n{args.num_train_shards}"
    if args.cache_dir is None:
        args.cache_dir = base_dir / "semdedup_cache" / f"eps{_eps_slug(args.eps)}_n{args.num_train_shards}"
    if args.analysis_output_dir is None:
        args.analysis_output_dir = args.output_data_dir
    if args.ray_temp_dir is None:
        args.ray_temp_dir = args.cache_dir / "ray"
    return args


def main() -> None:
    args = parse_args()
    args.input_data_dir = args.input_data_dir.expanduser().resolve()
    args.output_data_dir = args.output_data_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.analysis_output_dir = args.analysis_output_dir.expanduser().resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if not args.skip_token_stats:
        from nanochat.tokenizer import get_tokenizer

        tokenizer = get_tokenizer()

    train_paths, val_path = _select_train_val(args.input_data_dir, args.num_train_shards)
    staged_input_dir = args.cache_dir / "input_with_ids"
    curator_output_dir = args.cache_dir / "curator_output"
    exact_output_dir = args.cache_dir / "exact_smoke_deduplicated"

    print(f"Input train shards: {len(train_paths)} from {args.input_data_dir}")
    print(f"Validation shard: {val_path.name}")
    print(f"Output data dir: {args.output_data_dir}")
    print(f"Cache dir: {args.cache_dir}")

    input_stats = _stage_train_inputs(train_paths, staged_input_dir, args.max_docs, overwrite=args.overwrite)
    staged_paths = _parquet_files(staged_input_dir)
    if tokenizer is not None:
        input_stats = _count_texts(staged_paths, tokenizer, args.tokenizer_batch_size, args.tokenizer_threads)
    print(f"Staged {input_stats['docs']:,} docs / {input_stats['chars']:,} chars")
    if tokenizer is not None:
        print(f"Staged token count: {input_stats['tokens']:,}")

    backend_result = None
    if args.backend == "curator":
        _reset_dir(curator_output_dir, overwrite=args.overwrite)
        backend_result = _run_curator(args, staged_input_dir, curator_output_dir)
        dedup_dir = _find_curator_dedup_dir(curator_output_dir)
    else:
        backend_result = _run_exact_smoke(staged_input_dir, exact_output_dir, overwrite=args.overwrite)
        dedup_dir = exact_output_dir

    analysis_artifacts = _write_analysis_artifacts(
        staged_paths,
        dedup_dir,
        args.analysis_output_dir,
        tokenizer,
        args.audit_sample_size,
        args.audit_seed,
        args.tokenizer_batch_size,
        args.tokenizer_threads,
    )
    output_stats = _normalize_to_nanochat(
        dedup_dir,
        val_path,
        args.output_data_dir,
        overwrite=args.overwrite,
        tokenizer=tokenizer,
        tokenizer_batch_size=args.tokenizer_batch_size,
        tokenizer_threads=args.tokenizer_threads,
    )
    train_out = output_stats["train"]
    keep_ratio_docs = train_out["docs"] / input_stats["docs"] if input_stats["docs"] else None
    keep_ratio_chars = train_out["chars"] / input_stats["chars"] if input_stats["chars"] else None
    keep_ratio_tokens = None
    removed_tokens = None
    if "tokens" in input_stats and "tokens" in train_out and input_stats["tokens"]:
        keep_ratio_tokens = train_out["tokens"] / input_stats["tokens"]
        removed_tokens = input_stats["tokens"] - train_out["tokens"]
    removed_docs = input_stats["docs"] - train_out["docs"]
    removed_chars = input_stats["chars"] - train_out["chars"]

    manifest = {
        "backend": args.backend,
        "semantic_dedup": args.backend == "curator",
        "note": "exact-smoke is only a plumbing smoke test and is not SemDeDup." if args.backend == "exact-smoke" else "",
        "input_data_dir": str(args.input_data_dir),
        "output_data_dir": str(args.output_data_dir),
        "cache_dir": str(args.cache_dir),
        "analysis_output_dir": str(args.analysis_output_dir),
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
        "token_stats": {
            "enabled": tokenizer is not None,
            "include_bos_per_document": True,
            "tokenizer_batch_size": args.tokenizer_batch_size,
            "tokenizer_threads": args.tokenizer_threads,
        },
        "input_train": input_stats,
        "output_train": train_out,
        "output_val": output_stats["val"],
        "output_val_file": output_stats.get("val_output_file"),
        "removed_docs": removed_docs,
        "removed_chars": removed_chars,
        "removed_tokens": removed_tokens,
        "keep_ratio_docs": keep_ratio_docs,
        "keep_ratio_chars": keep_ratio_chars,
        "keep_ratio_tokens": keep_ratio_tokens,
        "analysis_artifacts": analysis_artifacts,
        "elapsed_sec": backend_result.get("elapsed_sec") if isinstance(backend_result, dict) else None,
        "backend_result": backend_result,
    }
    manifest_path = args.output_data_dir / "semdedup_manifest.json"
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path.write_text(manifest_json)
    analysis_manifest_path = args.analysis_output_dir / "semdedup_manifest.json"
    if analysis_manifest_path != manifest_path:
        analysis_manifest_path.write_text(manifest_json)
    print(f"Wrote manifest: {manifest_path}")
    if analysis_manifest_path != manifest_path:
        print(f"Wrote analysis manifest: {analysis_manifest_path}")
    keep_msg = f"Train keep ratio: docs={keep_ratio_docs:.4f} chars={keep_ratio_chars:.4f}"
    if keep_ratio_tokens is not None:
        keep_msg += f" tokens={keep_ratio_tokens:.4f}"
    print(keep_msg)


if __name__ == "__main__":
    main()
