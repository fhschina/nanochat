"""Distributed BOS-aligned best-fit parquet dataloaders."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pyarrow.parquet as pq
import torch

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

DATALOADER_STATE_VERSION = 2

STAT_KEYS = (
    "source_docs",
    "source_tokens",
    "removed_source_docs",
    "removed_source_tokens",
    "consumed_source_tokens",
    "input_tokens",
    "target_tokens",
    "removed_target_tokens",
    "row_boundary_input_only_tokens",
    "continuation_fragments",
    "discarded_source_tokens",
    # Legacy counters. Version-2 packers never increment these.
    "cropped_tokens",
    "removed_cropped_tokens",
)


def resolve_parquet_paths(path: str | Path) -> list[str]:
    """Resolve a parquet file or a flat parquet directory into sorted paths."""
    source = Path(path).expanduser().resolve()
    if source.is_file():
        if source.suffix != ".parquet":
            raise ValueError(f"Expected a parquet file, got: {source}")
        return [str(source)]
    if not source.is_dir():
        raise FileNotFoundError(source)
    paths = sorted(
        str(item)
        for item in source.iterdir()
        if item.is_file() and item.suffix == ".parquet" and not item.name.endswith(".tmp")
    )
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {source}")
    return paths


def _selected_parquet_paths(split, data_dir, parquet_paths, warn_on_legacy):
    if parquet_paths is not None:
        paths = [str(Path(path).expanduser().resolve()) for path in parquet_paths]
        if not paths:
            raise ValueError("parquet_paths must not be empty")
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        return paths
    paths = list_parquet_files(data_dir=data_dir, warn_on_legacy=warn_on_legacy)
    if not paths:
        raise FileNotFoundError("No dataset parquet files found, did you run dataset.py?")
    return paths[:-1] if split == "train" else paths[-1:]


def _document_batches(
    split,
    resume_state_dict,
    tokenizer_batch_size,
    data_dir=None,
    parquet_paths: Sequence[str] | None = None,
):
    """Yield document records and an exact cursor after each source batch."""
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    _, ddp_rank, _, ddp_world_size = get_dist_info()
    warn = ddp_rank == 0 and split == "train" and data_dir is None and parquet_paths is None
    paths = _selected_parquet_paths(split, data_dir, parquet_paths, warn)
    source_state = (resume_state_dict or {}).get("source_state", resume_state_dict or {})
    pq_idx = int(source_state.get("pq_idx", 0))
    rg_idx = int(source_state.get("rg_idx", -1))
    doc_offset = int(source_state.get("doc_offset", 0))
    epoch = int(source_state.get("epoch", 1))

    while True:
        if pq_idx >= len(paths):
            pq_idx, rg_idx, doc_offset, epoch = 0, -1, 0, epoch + 1
        pf = pq.ParquetFile(paths[pq_idx])
        if rg_idx < 0:
            rg_idx = ddp_rank
        if rg_idx >= pf.num_row_groups:
            pq_idx, rg_idx, doc_offset = pq_idx + 1, -1, 0
            continue

        available = set(pf.schema_arrow.names)
        columns = ["text"]
        if "removed_by_fuzzy" in available:
            columns.append("removed_by_fuzzy")
        if "nanochat_token_count" in available:
            columns.append("nanochat_token_count")
        rows = pf.read_row_group(rg_idx, columns=columns).to_pydict()
        texts = rows["text"]
        removed = rows.get("removed_by_fuzzy")
        declared_tokens = rows.get("nanochat_token_count")

        while doc_offset < len(texts):
            end = min(doc_offset + tokenizer_batch_size, len(texts))
            records = [
                {
                    "text": texts[index],
                    "removed_by_fuzzy": bool(removed[index]) if removed is not None else False,
                    "has_removed_label": removed is not None,
                    "declared_tokens": int(declared_tokens[index]) if declared_tokens is not None else None,
                    "source_epoch": epoch,
                }
                for index in range(doc_offset, end)
            ]
            next_pq, next_rg, next_offset, next_epoch = pq_idx, rg_idx, end, epoch
            if next_offset >= len(texts):
                next_offset, next_rg = 0, rg_idx + ddp_world_size
                if next_rg >= pf.num_row_groups:
                    next_pq, next_rg = pq_idx + 1, -1
                    if next_pq >= len(paths):
                        next_pq, next_epoch = 0, epoch + 1
            next_state = {
                "pq_idx": next_pq,
                "rg_idx": next_rg,
                "doc_offset": next_offset,
                "epoch": next_epoch,
            }
            yield records, next_state
            pq_idx, rg_idx = next_state["pq_idx"], next_state["rg_idx"]
            doc_offset, epoch = next_state["doc_offset"], next_state["epoch"]
            if doc_offset == 0:
                break


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    B,
    T,
    split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    data_dir=None,
    buffer_size=1000,
    parquet_paths: Sequence[str] | None = None,
):
    """Yield packed tensors plus an exact post-batch source/buffer state."""
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive")
    resume = resume_state_dict or {}
    row_capacity = T + 1
    batches = _document_batches(
        split,
        resume,
        tokenizer_batch_size,
        data_dir=data_dir,
        parquet_paths=parquet_paths,
    )
    bos_token = tokenizer.get_bos_token_id()
    state_version = int(resume.get("state_version", 1))
    doc_buffer = [
        (
            list(item["tokens"]),
            bool(item["removed_by_fuzzy"]),
            bool(item.get("continuation", False)),
            int(item.get("source_epoch", resume.get("consumed_epoch", 1))),
        )
        for item in resume.get("doc_buffer", [])
    ]
    stats = {key: int(resume.get("data_stats", {}).get(key, 0)) for key in STAT_KEYS}
    if state_version < DATALOADER_STATE_VERSION and "discarded_source_tokens" not in resume.get("data_stats", {}):
        # Version 1 called silently discarded source tokens "cropped_tokens".
        stats["discarded_source_tokens"] = stats["cropped_tokens"]
    stats["has_removed_labels"] = bool(
        resume.get("data_stats", {}).get("has_removed_labels", False)
    )
    source_state = dict(
        resume.get(
            "source_state",
            {"pq_idx": 0, "rg_idx": -1, "doc_offset": 0, "epoch": 1},
        )
    )
    consumed_epoch = int(resume.get("consumed_epoch", 1))

    def serialize_buffer():
        return [
            {
                "tokens": tokens,
                "removed_by_fuzzy": removed,
                "continuation": continuation,
                "source_epoch": source_epoch,
            }
            for tokens, removed, continuation, source_epoch in doc_buffer
        ]

    def snapshot_state():
        return {
            **source_state,
            "state_version": DATALOADER_STATE_VERSION,
            "packer": "bos_bestfit_continuation",
            "source_state": dict(source_state),
            "consumed_epoch": consumed_epoch,
            "doc_buffer": serialize_buffer(),
            "data_stats": dict(stats),
        }

    def refill_buffer():
        nonlocal source_state
        records, source_state = next(batches)
        token_lists = tokenizer.encode(
            [record["text"] for record in records],
            prepend=bos_token,
            num_threads=tokenizer_threads,
        )
        for record, tokens in zip(records, token_lists, strict=True):
            tokens = list(tokens)
            removed = bool(record["removed_by_fuzzy"])
            stats["has_removed_labels"] = bool(
                stats["has_removed_labels"] or record["has_removed_label"]
            )
            stats["source_docs"] += 1
            stats["source_tokens"] += len(tokens)
            if removed:
                stats["removed_source_docs"] += 1
                stats["removed_source_tokens"] += len(tokens)
            doc_buffer.append((tokens, removed, False, int(record["source_epoch"])))

    use_cuda = device == "cuda" or str(device).startswith("cuda:")
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    origin_buffer = torch.empty((B, row_capacity), dtype=torch.bool)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    while True:
        resume_state_before_batch = snapshot_state()
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                active_epoch = min(item[3] for item in doc_buffer)
                best_idx = -1
                best_len = 0
                for index, (tokens, _, _, source_epoch) in enumerate(doc_buffer):
                    if (
                        source_epoch == active_epoch
                        and len(tokens) <= remaining
                        and len(tokens) > best_len
                    ):
                        best_idx = index
                        best_len = len(tokens)

                if best_idx >= 0:
                    tokens, removed, continuation, source_epoch = doc_buffer.pop(best_idx)
                    used = len(tokens)
                else:
                    shortest_idx = min(
                        (
                            index
                            for index, item in enumerate(doc_buffer)
                            if item[3] == active_epoch
                        ),
                        key=lambda index: len(doc_buffer[index][0]),
                    )
                    tokens, removed, continuation, source_epoch = doc_buffer[shortest_idx]
                    used = remaining
                    remainder = tokens[used:]
                    if not remainder:
                        raise RuntimeError("best-fit continuation split produced an empty remainder")
                    doc_buffer[shortest_idx] = (remainder, removed, True, source_epoch)
                    stats["continuation_fragments"] += 1
                row_buffer[row_idx, pos : pos + used] = torch.tensor(tokens[:used], dtype=torch.long)
                origin_buffer[row_idx, pos : pos + used] = removed
                pos += used
                consumed_epoch = max(consumed_epoch, source_epoch)

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        batch_removed_targets = int(origin_buffer[:, 1:].sum().item())
        stats["consumed_source_tokens"] += B * row_capacity
        stats["input_tokens"] += B * T
        stats["target_tokens"] += B * T
        stats["removed_target_tokens"] += batch_removed_targets
        stats["row_boundary_input_only_tokens"] += B
        state_dict = {
            **snapshot_state(),
            "resume_state_before_batch": resume_state_before_batch,
            "batch_stats": {
                "target_tokens": B * T,
                "removed_target_tokens": batch_removed_targets,
                "has_removed_labels": bool(stats["has_removed_labels"]),
                "consumed_epoch": consumed_epoch,
                "discarded_source_tokens": stats["discarded_source_tokens"],
            },
        }
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, _ in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
