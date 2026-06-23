#!/usr/bin/env bash

set -euo pipefail

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_b200_climbmix}"
SOURCE_SEMDEDUP_RUN="${SOURCE_SEMDEDUP_RUN:-$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/full-n170-r9p5-eps0p07-20260612T080000Z_semdedup}"
BASELINE_RUN="${BASELINE_RUN:-$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_quality/full-n170-r9p5-eps0p07-20260611T163000Z_baseline}"
ORIG_DATA_DIR="${ORIG_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_climbmix}"
TASK_DELTA_ROOT="${TASK_DELTA_ROOT:-$NANOCHAT_BASE_DIR/experiments/climbmix_semdedup_task_delta}"
ORDER_ROOT="${ORDER_ROOT:-$TASK_DELTA_ROOT/order_preserving}"
ORDER_DATA_DIR="${ORDER_DATA_DIR:-$ORDER_ROOT/data/base_data_climbmix_semdedup_eps0p07_orderpreserve_n170}"
ORDER_REBUILD="${ORDER_REBUILD:-0}"

mkdir -p "$ORDER_ROOT/logs" "$ORDER_ROOT/pids" "$(dirname "$ORDER_DATA_DIR")"

build_order_preserving_data() {
  if [[ "$ORDER_REBUILD" != "1" && -d "$ORDER_DATA_DIR" && -f "$ORDER_DATA_DIR/shard_00170.parquet" ]]; then
    echo "Order-preserving data dir already exists: $ORDER_DATA_DIR"
  else
    PYTHON_BIN="${PYTHON_BIN:-python}" \
    SOURCE_SEMDEDUP_RUN="$SOURCE_SEMDEDUP_RUN" \
    ORIG_DATA_DIR="$ORIG_DATA_DIR" \
    ORDER_DATA_DIR="$ORDER_DATA_DIR" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROW_GROUP_SIZE = 1024

source_run = Path(os.environ["SOURCE_SEMDEDUP_RUN"]).expanduser().resolve()
orig_data_dir = Path(os.environ["ORIG_DATA_DIR"]).expanduser().resolve()
out_dir = Path(os.environ["ORDER_DATA_DIR"]).expanduser().resolve()
input_dir = source_run / "semdedup_cache" / "input_with_ids"
duplicates_dir = source_run / "semdedup_cache" / "curator_output" / "duplicates"
source_manifest_path = source_run / "semdedup_manifest.json"

if not input_dir.exists():
    raise FileNotFoundError(f"Missing input_with_ids: {input_dir}")
if not duplicates_dir.exists():
    raise FileNotFoundError(f"Missing duplicates dir: {duplicates_dir}")

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)

removed: set[str] = set()
for path in sorted(duplicates_dir.glob("*.parquet")):
    pf = pq.ParquetFile(path)
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["id"])
        removed.update(table.column("id").to_pylist())
removed_values = pa.array(sorted(removed))

train_files = sorted(input_dir.glob("shard_*.parquet"))
if len(train_files) != 170:
    raise ValueError(f"Expected 170 train shards in {input_dir}, found {len(train_files)}")

output_files = []
output_docs = 0
for shard_idx, src_path in enumerate(train_files):
    expected_name = f"shard_{shard_idx:05d}.parquet"
    if src_path.name != expected_name:
        raise ValueError(f"Unexpected input shard order: {src_path.name} != {expected_name}")
    dst_path = out_dir / expected_name
    pf = pq.ParquetFile(src_path)
    writer = None
    shard_docs = 0
    try:
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["doc_id", "text"])
            keep_mask = pc.invert(pc.is_in(table.column("doc_id"), value_set=removed_values))
            kept = table.filter(keep_mask)
            if kept.num_rows == 0:
                continue
            out = pa.table({"text": kept.column("text")})
            if writer is None:
                writer = pq.ParquetWriter(dst_path, out.schema, compression="zstd")
            writer.write_table(out, row_group_size=ROW_GROUP_SIZE)
            shard_docs += kept.num_rows
    finally:
        if writer is not None:
            writer.close()
    if not dst_path.exists():
        raise RuntimeError(f"No output written for {src_path}")
    output_docs += shard_docs
    output_files.append({"file": dst_path.name, "source_file": src_path.name, "docs": shard_docs})
    print(f"Wrote {dst_path.name}: {shard_docs} docs", flush=True)

val_src = orig_data_dir / "shard_00170.parquet"
val_dst = out_dir / "shard_00170.parquet"
if not val_src.exists():
    raise FileNotFoundError(f"Missing validation shard: {val_src}")
shutil.copy2(val_src, val_dst)

source_manifest = json.loads(source_manifest_path.read_text()) if source_manifest_path.exists() else {}
manifest = {
    **source_manifest,
    "order_preserving": True,
    "source_semdedup_run": str(source_run),
    "input_with_ids": str(input_dir),
    "duplicates_dir": str(duplicates_dir),
    "output_data_dir": str(out_dir),
    "removed_docs": len(removed),
    "output_train": {
        **source_manifest.get("output_train", {}),
        "docs": output_docs,
        "files": output_files,
    },
    "validation_shard": str(val_dst),
}
(out_dir / "semdedup_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(f"Wrote order-preserving SemDeDup data: {out_dir}", flush=True)
print(f"Removed docs: {len(removed)} output train docs: {output_docs}", flush=True)
PY
  fi

  PYTHON_BIN="${PYTHON_BIN:-python}" \
  SOURCE_SEMDEDUP_RUN="$SOURCE_SEMDEDUP_RUN" \
  ORDER_DATA_DIR="$ORDER_DATA_DIR" \
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import pyarrow.parquet as pq

source_run = Path(os.environ["SOURCE_SEMDEDUP_RUN"]).expanduser().resolve()
out_dir = Path(os.environ["ORDER_DATA_DIR"]).expanduser().resolve()
input_dir = source_run / "semdedup_cache" / "input_with_ids"
duplicates_dir = source_run / "semdedup_cache" / "curator_output" / "duplicates"

removed = set()
for path in sorted(duplicates_dir.glob("*.parquet")):
    pf = pq.ParquetFile(path)
    for rg_idx in range(pf.num_row_groups):
        removed.update(pf.read_row_group(rg_idx, columns=["id"]).column("id").to_pylist())

def first_output(shard_idx: int, n: int = 10):
    vals = []
    pf = pq.ParquetFile(out_dir / f"shard_{shard_idx:05d}.parquet")
    for rg_idx in range(pf.num_row_groups):
        vals.extend(pf.read_row_group(rg_idx, columns=["text"]).column("text").to_pylist())
        if len(vals) >= n:
            return vals[:n]
    return vals

def first_expected(shard_idx: int, n: int = 10):
    vals = []
    pf = pq.ParquetFile(input_dir / f"shard_{shard_idx:05d}.parquet")
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["doc_id", "text"])
        for doc_id, text in zip(table.column("doc_id").to_pylist(), table.column("text").to_pylist()):
            if doc_id not in removed:
                vals.append(text)
                if len(vals) >= n:
                    return vals
    return vals

for shard_idx in [0, 1, 2, 3, 4, 126, 169]:
    out = first_output(shard_idx)
    exp = first_expected(shard_idx)
    matches = sum(a == b for a, b in zip(out, exp))
    print(f"verify shard_{shard_idx:05d}: prefix matches {matches}/10")
    if matches != min(10, len(out), len(exp)):
        raise RuntimeError(f"Order-preserving verification failed for shard_{shard_idx:05d}")
print("Order-preserving verification passed.")
PY
}

build_order_preserving_data

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${RUN_ID:-full-semdedup-eps0p07-orderpreserve-seed42-n170-i6612-$RUN_TIMESTAMP}"
RUN_TAG_SEMDEDUP="${RUN_TAG_SEMDEDUP:-d24-climbmix-semdedup-eps0p07-orderpreserve-seed42-n170-i6612-$RUN_TIMESTAMP}"

SEMD_OUTPUT_DIR="$ORDER_DATA_DIR" \
RUN_ROOT="$ORDER_ROOT" \
RUN_ID="$RUN_ID" \
RUN_TAG_SEMDEDUP="$RUN_TAG_SEMDEDUP" \
WANDB_RUN="${WANDB_RUN:-climbmix-semdedup-eps0p07-orderpreserve-seed42-n170-i6612}" \
NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
NUM_GPUS="${NUM_GPUS:-8}" \
DEPTH="${DEPTH:-24}" \
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}" \
NUM_TRAIN_SHARDS="${NUM_TRAIN_SHARDS:-170}" \
NUM_ITERATIONS="${NUM_ITERATIONS:-6612}" \
PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-9.5}" \
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" \
SEMD_MODEL="${SEMD_MODEL:-google/embeddinggemma-300m}" \
SEMD_EPS="${SEMD_EPS:-0.07}" \
SEMD_N_CLUSTERS="${SEMD_N_CLUSTERS:-100}" \
SEMD_DISTANCE_METRIC="${SEMD_DISTANCE_METRIC:-cosine}" \
SEMD_WHICH_TO_KEEP="${SEMD_WHICH_TO_KEEP:-hard}" \
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-2000}" \
CORE_METRIC_MAX_PER_TASK="${CORE_METRIC_MAX_PER_TASK:-500}" \
FINAL_CORE_MAX_PER_TASK="${FINAL_CORE_MAX_PER_TASK:--1}" \
BASE_EVAL_MODES="${BASE_EVAL_MODES:-core,bpb,sample}" \
SKIP_UV_SYNC="${SKIP_UV_SYNC:-1}" \
PREP_DATASET="${PREP_DATASET:-0}" \
DO_SEMDEDUP=0 \
DO_TRAIN="${DO_TRAIN:-1}" \
DO_EVAL="${DO_EVAL:-1}" \
bash runs/climbmix_semdedup_quality_b200.sh semdedup

PYTHON_BIN="${PYTHON_BIN:-python}" \
BASELINE_RUN="$BASELINE_RUN" \
SOURCE_SEMDEDUP_RUN="$SOURCE_SEMDEDUP_RUN" \
ORDER_RUN="$ORDER_ROOT/$RUN_ID" \
ORDER_ROOT="$ORDER_ROOT" \
"$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

baseline_run = Path(os.environ["BASELINE_RUN"])
source_semdedup_run = Path(os.environ["SOURCE_SEMDEDUP_RUN"])
order_run = Path(os.environ["ORDER_RUN"])
order_root = Path(os.environ["ORDER_ROOT"])

def read_summary(run: Path):
    path = run / "run_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}

def read_core(run: Path):
    path = run / "base_eval_core.csv"
    out = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            task = row[0].strip()
            centered = row[2].strip() if len(row) > 2 else ""
            out[task] = float(centered) if centered else None
    return out

runs = {
    "baseline": baseline_run,
    "semdedup_hash_order": source_semdedup_run,
    "semdedup_order_preserving": order_run,
}
rows = []
for name, run in runs.items():
    summary = read_summary(run)
    core = read_core(run)
    rows.append({
        "arm": name,
        "run_dir": str(run),
        "val_bpb": summary.get("metrics", {}).get("base_eval_val_bpb") or summary.get("metrics", {}).get("final_train_val_bpb"),
        "CORE": core.get("CORE") or summary.get("metrics", {}).get("final_core"),
        "commonsense_qa": core.get("commonsense_qa"),
        "winograd": core.get("winograd"),
        "winogrande": core.get("winogrande"),
    })

report = order_root / "order_preserving_comparison.md"
lines = [
    "# ClimbMix Order-Preserving SemDeDup Control",
    "",
    "| Arm | Val BPB | CORE | commonsense_qa | winograd | winogrande |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        f"| {row['arm']} | {row['val_bpb']} | {row['CORE']} | "
        f"{row['commonsense_qa']} | {row['winograd']} | {row['winogrande']} |"
    )
lines.extend(["", "## Run Dirs", ""])
for row in rows:
    lines.append(f"- {row['arm']}: `{row['run_dir']}`")
report.write_text("\n".join(lines) + "\n")
print(f"Wrote comparison report: {report}")
PY
