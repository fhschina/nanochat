"""Enforce correctness and throughput gates for one CW fuzzy A/B smoke log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import median
from typing import Any


def check(args: argparse.Namespace) -> dict[str, Any]:
    text = args.log.read_text(errors="replace")
    for pattern in (r"Traceback \(most recent call last\)", r"RuntimeError:", r"loss:\s+nan", r"Validation bpb:\s+nan"):
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise RuntimeError(f"Smoke log contains failure pattern: {pattern}")
    if "Using Flash Attention 3" not in text:
        raise RuntimeError("Smoke log does not prove Flash Attention 3 was active")
    rows = [
        (int(step), int(tokens.replace(",", "")), float(wait_fraction))
        for step, wait_fraction, tokens in re.findall(
            r"step\s+(\d+)/\d+.*?data wait:\s+[0-9.]+ms\s+\(([0-9.]+)\).*?tok/sec:\s+([0-9,]+)",
            text,
        )
        if int(step) > args.ignore_first_steps
    ]
    if not rows:
        raise RuntimeError("Smoke log contains no post-warmup performance rows")
    median_tokens_per_sec = float(median(row[1] for row in rows))
    median_data_wait_fraction = float(median(row[2] for row in rows))
    projected_hours = (
        args.full_training_tokens / median_tokens_per_sec * (1.0 + args.overhead_fraction) / 3600
    )
    loader_matches = re.findall(r"FINAL_DATA_LOADER_STATS\s+(\{[^\n]+\})", text)
    if not loader_matches:
        raise RuntimeError("Smoke log has no final dataloader audit")
    loader = json.loads(loader_matches[-1])
    expected_loader = {
        "state_version": 2,
        "packer": "bos_bestfit_continuation",
        "consumed_epoch": 1,
        "discarded_source_tokens": 0,
    }
    for key, expected in expected_loader.items():
        if loader.get(key) != expected:
            raise RuntimeError(f"Dataloader audit mismatch for {key}: {loader.get(key)!r} != {expected!r}")
    if median_tokens_per_sec < args.min_tokens_per_sec:
        raise RuntimeError(
            f"Median throughput {median_tokens_per_sec:,.0f} is below {args.min_tokens_per_sec:,.0f} tokens/s"
        )
    if median_data_wait_fraction > args.max_data_wait_fraction:
        raise RuntimeError(
            f"Median data wait {median_data_wait_fraction:.4f} exceeds {args.max_data_wait_fraction:.4f}"
        )
    if projected_hours > args.max_projected_hours:
        raise RuntimeError(
            f"Projected full runtime {projected_hours:.2f}h exceeds {args.max_projected_hours:.2f}h"
        )
    result = {
        "status": "passed",
        "performance_rows": len(rows),
        "median_tokens_per_sec": median_tokens_per_sec,
        "median_data_wait_fraction": median_data_wait_fraction,
        "projected_full_hours_with_overhead": projected_hours,
        "loader": loader,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ignore-first-steps", type=int, default=10)
    parser.add_argument("--full-training-tokens", type=int, default=27_732_738_048)
    parser.add_argument("--min-tokens-per-sec", type=float, default=200_000)
    parser.add_argument("--max-data-wait-fraction", type=float, default=0.10)
    parser.add_argument("--max-projected-hours", type=float, default=42.0)
    parser.add_argument("--overhead-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(check(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
