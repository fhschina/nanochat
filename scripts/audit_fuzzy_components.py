"""Audit fuzzy-dedup component sizes and sampled removed-to-keeper pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CURATOR_ID = "_curator_dedup_id"
COMPONENT_ID = "_duplicate_group_id"
SIZE_BINS = (
    ("size_2", 2, 2),
    ("size_3", 3, 3),
    ("size_4_10", 4, 10),
    ("size_11_100", 11, 100),
    ("size_101_1000", 101, 1000),
    ("size_1001_plus", 1001, None),
)
QUANTILES = (0.0, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parquet_files(path: Path) -> list[Path]:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return files


def _load_sorted_column(files: list[Path], field: str, path: Path) -> np.memmap:
    rows = sum(pq.ParquetFile(file).metadata.num_rows for file in files)
    values = np.memmap(path, dtype=np.int64, mode="w+", shape=(rows,))
    offset = 0
    for file_index, file in enumerate(files, 1):
        parquet = pq.ParquetFile(file)
        if parquet.metadata.num_rows == 0:
            print(f"Loaded {field}: {file_index}/{len(files)} files, {offset:,}/{rows:,} rows")
            continue
        for batch in parquet.iter_batches(batch_size=2_000_000, columns=[field]):
            array = batch.column(0).to_numpy(zero_copy_only=False)
            values[offset : offset + len(array)] = array
            offset += len(array)
        print(f"Loaded {field}: {file_index}/{len(files)} files, {offset:,}/{rows:,} rows")
    if offset != rows:
        raise RuntimeError(f"Loaded {offset:,}/{rows:,} values for {field}")
    values.sort(kind="quicksort")
    values.flush()
    return values


def _bin_mask(sizes: np.ndarray, lower: int, upper: int | None) -> np.ndarray:
    mask = sizes >= lower
    if upper is not None:
        mask &= sizes <= upper
    return mask


def _component_statistics(
    labels: np.memmap,
    sample_per_bin: int,
    top_components: int,
    seed: int,
    chunk_rows: int,
) -> tuple[dict, dict[int, set[str]]]:
    rng = np.random.default_rng(seed)
    histogram: dict[int, int] = defaultdict(int)
    reservoirs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    top_groups = np.empty(0, dtype=np.int64)
    top_sizes = np.empty(0, dtype=np.int64)

    def emit(groups: np.ndarray, sizes: np.ndarray) -> None:
        nonlocal top_groups, top_sizes
        if len(groups) == 0:
            return
        unique_sizes, counts = np.unique(sizes, return_counts=True)
        for size, count in zip(unique_sizes.tolist(), counts.tolist()):
            histogram[int(size)] += int(count)

        candidates = min(top_components, len(groups))
        if candidates:
            indexes = np.argpartition(sizes, len(sizes) - candidates)[-candidates:]
            merged_groups = np.concatenate((top_groups, groups[indexes]))
            merged_sizes = np.concatenate((top_sizes, sizes[indexes]))
            keep = min(top_components, len(merged_groups))
            keep_indexes = np.argpartition(merged_sizes, len(merged_sizes) - keep)[-keep:]
            top_groups = merged_groups[keep_indexes]
            top_sizes = merged_sizes[keep_indexes]

        for name, lower, upper in SIZE_BINS:
            mask = _bin_mask(sizes, lower, upper)
            bin_groups = groups[mask]
            if len(bin_groups) == 0 or sample_per_bin == 0:
                continue
            priorities = rng.random(len(bin_groups))
            bin_sizes = sizes[mask]
            if name in reservoirs:
                old_groups, old_sizes, old_priorities = reservoirs[name]
                bin_groups = np.concatenate((old_groups, bin_groups))
                bin_sizes = np.concatenate((old_sizes, bin_sizes))
                priorities = np.concatenate((old_priorities, priorities))
            keep = min(sample_per_bin, len(bin_groups))
            indexes = np.argpartition(priorities, keep - 1)[:keep]
            reservoirs[name] = (bin_groups[indexes], bin_sizes[indexes], priorities[indexes])

    carry_label: int | None = None
    carry_count = 0
    total_rows = len(labels)
    for offset in range(0, total_rows, chunk_rows):
        end = min(total_rows, offset + chunk_rows)
        chunk = np.asarray(labels[offset:end])
        boundaries = np.flatnonzero(chunk[1:] != chunk[:-1]) + 1
        starts = np.concatenate((np.array([0]), boundaries))
        ends = np.concatenate((boundaries, np.array([len(chunk)])))
        groups = chunk[starts].copy()
        sizes = (ends - starts).astype(np.int64, copy=False)
        if carry_label is not None:
            if int(groups[0]) != carry_label:
                emit(np.array([carry_label], dtype=np.int64), np.array([carry_count], dtype=np.int64))
            else:
                sizes[0] += carry_count
        continues = end < total_rows and int(chunk[-1]) == int(labels[end])
        if continues:
            carry_label = int(groups[-1])
            carry_count = int(sizes[-1])
            emit(groups[:-1], sizes[:-1])
        else:
            carry_label = None
            carry_count = 0
            emit(groups, sizes)
        print(f"Component size scan: {end:,}/{total_rows:,} vertices")
    if carry_label is not None:
        emit(np.array([carry_label], dtype=np.int64), np.array([carry_count], dtype=np.int64))

    component_count = sum(histogram.values())
    duplicate_rows = total_rows - component_count
    cumulative = 0
    sorted_histogram = sorted(histogram.items())
    quantiles = {}
    targets = {q: int(round(q * (component_count - 1))) for q in QUANTILES}
    for size, count in sorted_histogram:
        next_cumulative = cumulative + count
        for q, target in targets.items():
            key = f"p{100 * q:g}"
            if key not in quantiles and target < next_cumulative:
                quantiles[key] = size
        cumulative = next_cumulative

    bin_counts = {}
    for name, lower, upper in SIZE_BINS:
        bin_counts[name] = sum(
            count for size, count in sorted_histogram if size >= lower and (upper is None or size <= upper)
        )
    order = np.argsort(top_sizes)[::-1]
    top = [
        {"component_id": int(top_groups[index]), "size": int(top_sizes[index])}
        for index in order
    ]
    selected: dict[int, set[str]] = defaultdict(set)
    for record in top:
        selected[record["component_id"]].add("top_component")
    sampled = {}
    for name, (groups, sizes, _) in reservoirs.items():
        records = []
        for group, size in zip(groups.tolist(), sizes.tolist()):
            selected[int(group)].add(name)
            records.append({"component_id": int(group), "size": int(size)})
        sampled[name] = sorted(records, key=lambda item: item["component_id"])

    summary = {
        "component_vertex_rows": total_rows,
        "component_count": component_count,
        "implied_duplicate_rows": duplicate_rows,
        "component_size_quantiles": quantiles,
        "component_size_bins": bin_counts,
        "component_size_histogram": {str(size): count for size, count in sorted_histogram},
        "top_components": top,
        "sampled_components": sampled,
    }
    return summary, selected


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    result = values.astype(np.uint64, copy=True) + np.uint64(seed)
    result = (result ^ (result >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    result = (result ^ (result >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return result ^ (result >> np.uint64(31))


def _collect_sampled_members(
    component_files: list[Path],
    removed_ids: np.memmap,
    selected: dict[int, set[str]],
    max_removed_per_component: int,
    seed: int,
) -> dict[int, dict]:
    groups_wanted = np.array(sorted(selected), dtype=np.int64)
    records = {
        int(group): {"seen": 0, "keepers": set(), "removed": [], "reasons": sorted(selected[int(group)])}
        for group in groups_wanted
    }
    for file_index, file in enumerate(component_files, 1):
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=1_000_000, columns=[CURATOR_ID, COMPONENT_ID]):
            ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            groups = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            positions = np.searchsorted(groups_wanted, groups)
            mask = positions < len(groups_wanted)
            mask[mask] &= groups_wanted[positions[mask]] == groups[mask]
            if not np.any(mask):
                continue
            ids = ids[mask]
            groups = groups[mask]
            removed_positions = np.searchsorted(removed_ids, ids)
            is_removed = removed_positions < len(removed_ids)
            is_removed[is_removed] &= removed_ids[removed_positions[is_removed]] == ids[is_removed]
            order = np.argsort(groups, kind="stable")
            ids = ids[order]
            groups = groups[order]
            is_removed = is_removed[order]
            boundaries = np.flatnonzero(groups[1:] != groups[:-1]) + 1
            starts = np.concatenate((np.array([0]), boundaries))
            ends = np.concatenate((boundaries, np.array([len(groups)])))
            for start, end in zip(starts.tolist(), ends.tolist()):
                group = int(groups[start])
                record = records[group]
                group_ids = ids[start:end]
                group_removed = is_removed[start:end]
                record["seen"] += len(group_ids)
                record["keepers"].update(int(value) for value in group_ids[~group_removed])
                removed = group_ids[group_removed]
                if len(removed):
                    priorities = _splitmix64(removed, seed ^ group)
                    old = record["removed"]
                    if old:
                        removed = np.concatenate((np.array([item[1] for item in old]), removed))
                        priorities = np.concatenate((np.array([item[0] for item in old], dtype=np.uint64), priorities))
                    keep = min(max_removed_per_component, len(removed))
                    indexes = np.argpartition(priorities, keep - 1)[:keep]
                    record["removed"] = [
                        (int(priorities[index]), int(removed[index])) for index in indexes
                    ]
        print(f"Sampled member scan: {file_index}/{len(component_files)} component files")
    return records


def _source_file_ranges(input_path: Path, id_generator_path: Path, input_blocksize: str) -> list[dict]:
    from nemo_curator.stages.deduplication.id_generator import IdGeneratorBase
    from nemo_curator.utils.file_utils import (
        _split_files_as_per_blocksize,
        get_all_file_paths_and_size_under,
        parse_bytes_string_to_int,
    )

    payload = json.loads(id_generator_path.read_text(encoding="utf-8"))
    generator = IdGeneratorBase(start_id=int(payload["next_id"]), batch_registry=payload["batch_registry"])
    file_records = get_all_file_paths_and_size_under(
        str(input_path),
        recurse_subdirectories=True,
        keep_extensions=[".parquet"],
        storage_options=None,
        sort_by_size=True,
    )
    groups = _split_files_as_per_blocksize(
        sorted(file_records, key=lambda item: item[1]), parse_bytes_string_to_int(input_blocksize)
    )
    ranges = []
    for group in groups:
        group_hash = str(uuid.uuid5(uuid.NAMESPACE_URL, ";".join(group)))
        if group_hash not in payload["batch_registry"]:
            raise KeyError(f"Input file group is absent from Curator registry: {group}")
        first, last = generator.get_batch_range(files=group, key=None)
        next_id = int(first)
        for source_name in group:
            source = Path(source_name).resolve()
            rows = pq.ParquetFile(source).metadata.num_rows
            ranges.append({"start": next_id, "end": next_id + rows, "source": source})
            next_id += rows
        if next_id - 1 != int(last):
            raise RuntimeError(f"Curator ID range does not match source rows for {group_hash}")
    return sorted(ranges, key=lambda item: item["start"])


def _extract_texts(target_ids: set[int], ranges: list[dict], input_path: Path) -> dict[int, dict]:
    sorted_targets = np.array(sorted(target_ids), dtype=np.int64)
    found: dict[int, dict] = {}
    for record in ranges:
        left = int(np.searchsorted(sorted_targets, record["start"], side="left"))
        right = int(np.searchsorted(sorted_targets, record["end"], side="left"))
        if right == left:
            continue
        source = record["source"]
        parquet = pq.ParquetFile(source)
        local_rows = sorted_targets[left:right] - int(record["start"])
        row_group_ends = np.cumsum(
            [parquet.metadata.row_group(index).num_rows for index in range(parquet.num_row_groups)]
        )
        row_groups = np.searchsorted(row_group_ends, local_rows, side="right")
        for row_group in np.unique(row_groups):
            table = parquet.read_row_group(int(row_group), columns=["doc_id", "text"])
            row_group_start = 0 if row_group == 0 else int(row_group_ends[row_group - 1])
            indexes = np.flatnonzero(row_groups == row_group)
            for index in indexes.tolist():
                local_row = int(local_rows[index])
                table_row = local_row - row_group_start
                curator_id = int(sorted_targets[left + index])
                found[curator_id] = {
                    "doc_id": str(table.column("doc_id")[table_row].as_py()),
                    "text": str(table.column("text")[table_row].as_py()),
                    "source_file": str(source.relative_to(input_path)),
                    "source_row": local_row,
                }
    missing = target_ids - set(found)
    if missing:
        raise RuntimeError(f"Failed to resolve {len(missing)} sampled Curator IDs to source rows")
    return found


def _char_ngram_jaccard(left: str, right: str, width: int) -> float:
    def shingles(text: str) -> set[str]:
        if len(text) < width:
            return {text}
        return {text[index : index + width] for index in range(len(text) - width + 1)}

    left_set = shingles(left)
    right_set = shingles(right)
    union = len(left_set | right_set)
    return 1.0 if union == 0 else len(left_set & right_set) / union


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {f"p{100 * q:g}": None for q in QUANTILES}
    return {
        f"p{100 * q:g}": float(np.quantile(np.asarray(values), q, method="nearest"))
        for q in QUANTILES
    }


def _render_report(summary: dict, pair_summary: dict, args: argparse.Namespace) -> str:
    bins = summary["component_size_bins"]
    quantiles = summary["component_size_quantiles"]
    top = summary["top_components"][:10]
    lines = [
        "# FineWeb-EDU-Fortified fuzzy component audit",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Component structure",
        "",
        f"- Vertices in non-singleton components: **{summary['component_vertex_rows']:,}**",
        f"- Components: **{summary['component_count']:,}**",
        f"- Implied removals (vertices - components): **{summary['implied_duplicate_rows']:,}**",
        f"- Median / p99 / max component size: **{quantiles.get('p50')} / {quantiles.get('p99')} / {quantiles.get('p100')}**",
        "",
        "| Component size | Components |",
        "| --- | ---: |",
    ]
    for name, _, _ in SIZE_BINS:
        lines.append(f"| `{name}` | {bins[name]:,} |")
    lines.extend(
        [
            "",
            "Largest components:",
            "",
            "| Component ID | Size |",
            "| ---: | ---: |",
        ]
    )
    lines.extend(f"| {item['component_id']} | {item['size']:,} |" for item in top)
    lines.extend(
        [
            "",
            "## Removed-to-keeper sample",
            "",
            f"- Pairs audited: **{pair_summary['pairs']}**",
            f"- Exact {args.char_ngrams}-character-ngram Jaccard p50 / p10 / min: "
            f"**{pair_summary['jaccard_quantiles'].get('p50')} / "
            f"{pair_summary['jaccard_quantiles'].get('p10')} / "
            f"{pair_summary['jaccard_quantiles'].get('p0')}**",
            f"- Pairs below Jaccard 0.5 / 0.7 / 0.9: "
            f"**{pair_summary['below_threshold']['0.5']} / "
            f"{pair_summary['below_threshold']['0.7']} / "
            f"{pair_summary['below_threshold']['0.9']}**",
            "",
            "The sampled pair JSONL contains source IDs, previews, component size, and exact character Jaccard. "
            "Embedding cosine is intentionally left pending until a real precomputed embedding path and ID schema are available.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit NeMo Curator fuzzy connected components")
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--fuzzy-removed-ids", type=Path, required=True)
    parser.add_argument("--id-generator-path", type=Path, required=True)
    parser.add_argument("--input-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-blocksize", default="512MiB")
    parser.add_argument("--char-ngrams", type=int, default=24)
    parser.add_argument("--sample-components-per-bin", type=int, default=20)
    parser.add_argument("--top-components", type=int, default=20)
    parser.add_argument("--max-removed-per-component", type=int, default=5)
    parser.add_argument("--scan-chunk-rows", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-duplicates", type=int)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("component_dir", "fuzzy_removed_ids", "id_generator_path", "input_data_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    temp_dir = args.output_dir / "_tmp"
    temp_dir.mkdir()

    component_files = _parquet_files(args.component_dir)
    removed_files = _parquet_files(args.fuzzy_removed_ids)
    labels = _load_sorted_column(component_files, COMPONENT_ID, temp_dir / "component_labels.int64")
    removed_ids = _load_sorted_column(removed_files, CURATOR_ID, temp_dir / "removed_ids.int64")
    summary, selected = _component_statistics(
        labels,
        args.sample_components_per_bin,
        args.top_components,
        args.seed,
        args.scan_chunk_rows,
    )
    if summary["implied_duplicate_rows"] != len(removed_ids):
        raise RuntimeError(
            "Component structure implies "
            f"{summary['implied_duplicate_rows']:,} removals, ID output has {len(removed_ids):,}"
        )
    if args.expected_duplicates is not None and len(removed_ids) != args.expected_duplicates:
        raise RuntimeError(f"Expected {args.expected_duplicates:,} removals, found {len(removed_ids):,}")

    members = _collect_sampled_members(
        component_files,
        removed_ids,
        selected,
        args.max_removed_per_component,
        args.seed,
    )
    target_ids = set()
    for group, record in members.items():
        if record["seen"] <= 1:
            raise RuntimeError(f"Selected component {group} has only {record['seen']} members")
        if len(record["keepers"]) != 1:
            raise RuntimeError(f"Selected component {group} has {len(record['keepers'])} keepers")
        target_ids.update(record["keepers"])
        target_ids.update(item[1] for item in record["removed"])
    ranges = _source_file_ranges(args.input_data_dir, args.id_generator_path, args.input_blocksize)
    texts = _extract_texts(target_ids, ranges, args.input_data_dir)

    size_by_group = {}
    for item in summary["top_components"]:
        size_by_group[item["component_id"]] = item["size"]
    for records in summary["sampled_components"].values():
        for item in records:
            size_by_group[item["component_id"]] = item["size"]
    pairs = []
    for group in sorted(members):
        record = members[group]
        keeper_id = next(iter(record["keepers"]))
        keeper = texts[keeper_id]
        for _, removed_id in sorted(record["removed"]):
            removed = texts[removed_id]
            jaccard = _char_ngram_jaccard(keeper["text"], removed["text"], args.char_ngrams)
            pairs.append(
                {
                    "component_id": group,
                    "component_size": size_by_group[group],
                    "selection_reasons": record["reasons"],
                    "keeper_curator_id": keeper_id,
                    "keeper_doc_id": keeper["doc_id"],
                    "keeper_source_file": keeper["source_file"],
                    "keeper_source_row": keeper["source_row"],
                    "keeper_chars": len(keeper["text"]),
                    "keeper_preview": keeper["text"][:500].replace("\n", " "),
                    "removed_curator_id": removed_id,
                    "removed_doc_id": removed["doc_id"],
                    "removed_source_file": removed["source_file"],
                    "removed_source_row": removed["source_row"],
                    "removed_chars": len(removed["text"]),
                    "removed_preview": removed["text"][:500].replace("\n", " "),
                    "char_ngram_width": args.char_ngrams,
                    "char_ngram_jaccard": jaccard,
                    "embedding_cosine": None,
                }
            )

    jaccards = [pair["char_ngram_jaccard"] for pair in pairs]
    pair_summary = {
        "pairs": len(pairs),
        "jaccard_quantiles": _quantiles(jaccards),
        "below_threshold": {
            str(threshold): sum(value < threshold for value in jaccards)
            for threshold in (0.5, 0.7, 0.9)
        },
        "embedding_cosine_status": "pending_precomputed_embeddings_and_id_schema",
    }
    payload = {
        "format_version": 1,
        "completed_at": _utc_now(),
        "config": {
            "char_ngrams": args.char_ngrams,
            "sample_components_per_bin": args.sample_components_per_bin,
            "top_components": args.top_components,
            "max_removed_per_component": args.max_removed_per_component,
            "seed": args.seed,
        },
        "components": summary,
        "pair_audit": pair_summary,
    }
    (args.output_dir / "component_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "removed_to_keeper_pairs.jsonl").open("w", encoding="utf-8") as output:
        for pair in pairs:
            output.write(json.dumps(pair, ensure_ascii=True, sort_keys=True) + "\n")
    (args.output_dir / "report.md").write_text(
        _render_report(summary, pair_summary, args), encoding="utf-8"
    )
    if not args.keep_temp:
        del labels
        del removed_ids
        shutil.rmtree(temp_dir)
    print(json.dumps(pair_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
