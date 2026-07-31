"""Export frozen NanoChat validation and CORE examples as dedup query documents."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SCHEMA = pa.schema([
    pa.field("text", pa.string(), nullable=False),
    pa.field("subset", pa.string(), nullable=False),
    pa.field("doc_id", pa.string(), nullable=False),
    pa.field("nanochat_token_count", pa.int64(), nullable=False),
    pa.field("eval_source", pa.string(), nullable=False),
    pa.field("source_path", pa.string(), nullable=False),
])


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


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key in sorted(value):
            values.extend(flatten_strings(value[key]))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(flatten_strings(item))
        return values
    return []


def canonical_core_text(row: dict[str, Any]) -> str:
    parts = [part.strip() for part in flatten_strings(row) if part.strip()]
    return "\n".join(parts)


def export(args: argparse.Namespace) -> dict[str, Any]:
    validation = args.validation_file.expanduser().resolve()
    core_bundle = args.core_bundle.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not validation.is_file():
        raise FileNotFoundError(validation)
    config_path = core_bundle / "core.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    manifest_path = output / "eval_contamination_manifest.json"
    config = {
        "validation_file": str(validation),
        "core_bundle": str(core_bundle),
        "canonicalization": "recursive string leaves sorted by dictionary key and joined with newline",
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

    records: list[dict[str, Any]] = []
    validation_parquet = pq.ParquetFile(validation)
    if "text" not in validation_parquet.schema_arrow.names:
        raise ValueError(f"Validation file has no text column: {validation}")
    validation_index = 0
    for row_group in range(validation_parquet.num_row_groups):
        texts = validation_parquet.read_row_group(row_group, columns=["text"])["text"].to_pylist()
        for text in texts:
            records.append({
                "text": str(text),
                "subset": "__validation__",
                "doc_id": f"validation:{validation_index:08d}",
                "nanochat_token_count": 0,
                "eval_source": "validation",
                "source_path": str(validation),
            })
            validation_index += 1

    core_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    core_files: list[Path] = []
    core_docs = 0
    for task in core_config.get("icl_tasks", []):
        label = str(task["label"])
        data_path = core_bundle / "eval_data" / task["dataset_uri"]
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        core_files.append(data_path)
        with data_path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                row = json.loads(line)
                text = canonical_core_text(row)
                if not text:
                    continue
                records.append({
                    "text": text,
                    "subset": "__core__",
                    "doc_id": f"core:{label}:{row_index:08d}",
                    "nanochat_token_count": 0,
                    "eval_source": f"core:{label}",
                    "source_path": str(data_path),
                })
                core_docs += 1
    if not records:
        raise RuntimeError("No evaluation documents were exported")

    data_path = output / "eval_documents.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), data_path, compression="zstd", row_group_size=1024)
    payload = {
        "format_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "config": config,
        "validation": {
            "file": str(validation),
            "sha256": sha256_file(validation),
            "docs": validation_index,
        },
        "core": {
            "config_file": str(config_path),
            "config_sha256": sha256_file(config_path),
            "files": [{"file": str(path), "sha256": sha256_file(path)} for path in core_files],
            "docs": core_docs,
        },
        "output_file": data_path.name,
        "output_sha256": sha256_file(data_path),
        "docs": len(records),
    }
    atomic_json(manifest_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--core-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = export(parse_args())
    print(json.dumps({"docs": result["docs"], "validation": result["validation"], "core": result["core"]["docs"]}, indent=2))


if __name__ == "__main__":
    main()
