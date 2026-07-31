"""Create a hardlinked training+evaluation corpus for contamination identification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


MANIFEST_NAME = "contamination_union_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parquet_files(root: Path) -> list[Path]:
    files = sorted(path for path in root.rglob("*.parquet") if not path.name.endswith(".tmp"))
    if not files:
        raise FileNotFoundError(f"No parquet files below {root}")
    return files


def link_tree(source: Path, output: Path, prefix: str, link_mode: str = "hardlink") -> list[dict[str, Any]]:
    rows = []
    for index, source_path in enumerate(parquet_files(source)):
        relative = source_path.relative_to(source)
        destination = output / prefix / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if link_mode == "copy":
            shutil.copy2(source_path, destination)
            method = "copy"
        else:
            try:
                os.link(source_path, destination)
                method = "hardlink"
            except OSError as error:
                if link_mode == "hardlink":
                    raise RuntimeError(
                        f"Hardlink required but failed for {source_path} -> {destination}: {error}"
                    ) from error
                shutil.copy2(source_path, destination)
                method = "copy"
        rows.append({
            "index": index,
            "source": str(source_path),
            "file": str(destination.relative_to(output)),
            "bytes": destination.stat().st_size,
            "method": method,
        })
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    training = args.training_root.expanduser().resolve()
    evaluation = args.eval_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (training, evaluation):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if output in {training, evaluation, Path("/")}:
        raise ValueError(f"Unsafe output path: {output}")
    manifest_path = output / MANIFEST_NAME
    link_mode = getattr(args, "link_mode", "hardlink")
    config = {
        "training_root": str(training),
        "eval_root": str(evaluation),
        "link_mode": link_mode,
    }
    if manifest_path.is_file() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("config") == config:
            return existing
        raise FileExistsError(output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    started = utc_now()
    atomic_json(manifest_path, {"status": "running", "config": config, "started_at": started})
    training_files = link_tree(training, output, "training", link_mode=link_mode)
    eval_files = link_tree(evaluation, output, "eval", link_mode=link_mode)
    if link_mode == "hardlink" and any(
        row["method"] != "hardlink" for row in training_files + eval_files
    ):
        raise RuntimeError("Hardlink-only union unexpectedly contains copied files")
    source_manifests = []
    for candidate in (
        training / "materialize_manifest.json",
        training.parent / "work" / "exact_fuzzy_manifest.json",
        evaluation / "eval_contamination_manifest.json",
    ):
        if candidate.is_file():
            source_manifests.append({"file": str(candidate), "sha256": sha256_file(candidate)})
    payload = {
        "format_version": 1,
        "status": "complete",
        "config": config,
        "started_at": started,
        "completed_at": utc_now(),
        "training_prefix": "training",
        "eval_prefix": "eval",
        "training_files": training_files,
        "eval_files": eval_files,
        "source_manifests": source_manifests,
        "totals": {
            "training_files": len(training_files),
            "eval_files": len(eval_files),
            "bytes": sum(row["bytes"] for row in training_files + eval_files),
        },
    }
    atomic_json(manifest_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "auto", "copy"),
        default="hardlink",
        help="hardlink is the safe default; auto permits a copy fallback",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args())["totals"], indent=2))


if __name__ == "__main__":
    main()
