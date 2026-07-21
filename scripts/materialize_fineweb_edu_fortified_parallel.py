"""Materialize independent FineWeb-EDU-Fortified configs concurrently."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from datasets import get_dataset_config_names

from scripts.materialize_fineweb_edu_fortified import _atomic_write_json, _manifest_totals


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_subset_name(subset: str) -> str:
    return subset.replace("/", "__")


def _child_dir(args: argparse.Namespace, subset: str) -> Path:
    return args.output_dir / "subsets" / _safe_subset_name(subset)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_manifest(args: argparse.Namespace, subsets: list[str], status: str) -> dict:
    root_path = args.output_dir / "materialize_manifest.json"
    previous = _load_json(root_path) if root_path.is_file() else {}
    shards: list[dict] = []
    subset_stats: dict[str, dict] = {}
    completed = 0

    for subset in subsets:
        child_dir = _child_dir(args, subset)
        child_manifest_path = child_dir / "materialize_manifest.json"
        if not child_manifest_path.is_file():
            subset_stats[subset] = {"status": "pending", "docs": 0}
            continue
        child = _load_json(child_manifest_path)
        child_state = dict(child.get("subset_stats", {}).get(subset, {"status": child.get("status", "pending")}))
        subset_stats[subset] = child_state
        if child.get("status") == "complete":
            completed += 1
        relative_dir = child_dir.relative_to(args.output_dir)
        for shard in child.get("shards", []):
            merged_shard = dict(shard)
            merged_shard["file"] = str(relative_dir / shard["file"])
            shards.append(merged_shard)

    manifest = {
        "format_version": 3,
        "mode": "corpus",
        "status": status,
        "created_at": previous.get("created_at", _utc_now()),
        "updated_at": _utc_now(),
        "dataset": args.dataset,
        "split": args.split,
        "subsets": subsets,
        "docs_per_shard": args.docs_per_shard,
        "max_docs_per_subset": args.max_docs_per_subset,
        "tokenizer_dir": str(args.tokenizer_dir),
        "include_bos": True,
        "output_dir": str(args.output_dir),
        "parallel_workers": args.workers,
        "completed_subsets": completed,
        "shards": shards,
        "subset_stats": subset_stats,
        "totals": _manifest_totals(shards),
    }
    if status == "complete":
        manifest["completed_at"] = _utc_now()
    _atomic_write_json(root_path, manifest)
    return manifest


def _run_subset(args: argparse.Namespace, subset: str) -> str:
    child_dir = _child_dir(args, subset)
    manifest_path = child_dir / "materialize_manifest.json"
    if manifest_path.is_file() and _load_json(manifest_path).get("status") == "complete":
        return subset

    child_dir.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "_materialize_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    base_command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.materialize_fineweb_edu_fortified",
        "--corpus",
        "--dataset",
        args.dataset,
        "--subset",
        subset,
        "--split",
        args.split,
        "--output-dir",
        str(child_dir),
        "--docs-per-shard",
        str(args.docs_per_shard),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
        "--tokenizer-batch-size",
        str(args.tokenizer_batch_size),
        "--tokenizer-threads",
        str(args.tokenizer_threads),
        "--fast-exit",
        "--recover-orphan-shards",
    ]
    if args.max_docs_per_subset is not None:
        base_command.extend(["--max-docs-per-subset", str(args.max_docs_per_subset)])

    log_path = log_dir / f"{_safe_subset_name(subset)}.log"
    print(f"Starting {subset}; log={log_path}", flush=True)
    for attempt in range(1, args.retries + 1):
        command = list(base_command)
        if manifest_path.is_file():
            command.append("--resume")
        elif args.overwrite:
            command.append("--overwrite")
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n=== attempt {attempt}/{args.retries} at {_utc_now()} ===\n")
            log_file.flush()
            result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        if result.returncode == 0:
            child = _load_json(manifest_path)
            if child.get("status") == "complete":
                print(f"Completed {subset}: {child['totals']['docs']:,} docs", flush=True)
                return subset
        if result.returncode in (130, -2, -15):
            raise RuntimeError(f"Subset {subset} was interrupted; see {log_path}")
        if attempt < args.retries:
            delay = args.retry_backoff_sec * (2 ** (attempt - 1))
            print(
                f"Retrying {subset} after exit code {result.returncode} in {delay:.0f}s "
                f"(attempt {attempt + 1}/{args.retries})",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"Subset {subset} failed after {args.retries} attempts; see {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel FineWeb-EDU-Fortified materialization")
    parser.add_argument("--dataset", default="airtrain-ai/fineweb-edu-fortified")
    parser.add_argument("--subset", action="append")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docs-per-shard", type=int, default=100_000)
    parser.add_argument("--max-docs-per-subset", type=int)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--tokenizer-threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-backoff-sec", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fast-exit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.retries <= 0 or args.retry_backoff_sec < 0:
        raise ValueError("Retry count must be positive and backoff must be non-negative")
    if not (args.tokenizer_dir / "tokenizer.pkl").is_file():
        raise FileNotFoundError(args.tokenizer_dir / "tokenizer.pkl")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subsets = args.subset or get_dataset_config_names(args.dataset)
    subsets = list(dict.fromkeys(subsets))
    _merge_manifest(args, subsets, "running")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_subset, args, subset): subset for subset in subsets}
            errors: list[str] = []
            for future in as_completed(futures):
                subset = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    error = f"{subset}: {type(exc).__name__}: {exc}"
                    errors.append(error)
                    print(f"ERROR: {error}", flush=True)
                progress = _merge_manifest(args, subsets, "running")
                print(
                    f"Progress: {progress['completed_subsets']}/{len(subsets)} configs; "
                    f"{progress['totals']['docs']:,} docs",
                    flush=True,
                )
        if errors:
            raise RuntimeError("Materialization failures:\n" + "\n".join(errors))
    except BaseException:
        _merge_manifest(args, subsets, "interrupted")
        raise

    manifest = _merge_manifest(args, subsets, "complete")
    print(
        f"Completed parallel materialization: configs={len(subsets)} "
        f"docs={manifest['totals']['docs']:,} hf_tokens={manifest['totals']['hf_tokens']:,} "
        f"nanochat_tokens={manifest['totals']['nanochat_tokens']:,}",
        flush=True,
    )
    if args.fast_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
