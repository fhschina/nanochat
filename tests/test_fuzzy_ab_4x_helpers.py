import argparse
import json
from pathlib import Path

import pytest

from scripts.check_fuzzy_ab_smoke import check
from scripts.compare_fortified_fuzzy_ab import paired_bpb_bootstrap


def test_paired_bpb_bootstrap_detects_lower_fuzzy_bpb() -> None:
    fuzzy = [{"nats": 90.0 + index, "bytes": 100} for index in range(16)]
    raw = [{"nats": 100.0 + index, "bytes": 100} for index in range(16)]
    result = paired_bpb_bootstrap(fuzzy, raw, repetitions=1_000, seed=42)
    assert result["observed_delta"] < 0
    assert result["ci95_high"] < 0
    assert result["paired_batches"] == 16


def test_smoke_gate_accepts_fa3_capacity_and_throughput(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    rows = [
        f"step {step:05d}/26448 | data wait: 10.0ms (0.0100) | tok/sec: 250,000"
        for step in range(11, 31)
    ]
    rows.extend([
        "Using Flash Attention 3 (Hopper GPU detected)",
        "FINAL_DATA_LOADER_STATS " + json.dumps({
            "state_version": 2,
            "packer": "bos_bestfit_continuation",
            "consumed_epoch": 1,
            "discarded_source_tokens": 0,
        }),
    ])
    log.write_text("\n".join(rows))
    output = tmp_path / "result.json"
    result = check(argparse.Namespace(
        log=log,
        output=output,
        ignore_first_steps=10,
        full_training_tokens=27_732_738_048,
        min_tokens_per_sec=200_000,
        max_data_wait_fraction=0.10,
        max_projected_hours=42.0,
        overhead_fraction=0.10,
    ))
    assert result["status"] == "passed"
    assert output.is_file()


def test_smoke_gate_rejects_slow_run(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text("\n".join([
        "Using Flash Attention 3 (Hopper GPU detected)",
        "step 00011/26448 | data wait: 10.0ms (0.0100) | tok/sec: 150,000",
        "FINAL_DATA_LOADER_STATS " + json.dumps({
            "state_version": 2,
            "packer": "bos_bestfit_continuation",
            "consumed_epoch": 1,
            "discarded_source_tokens": 0,
        }),
    ]))
    with pytest.raises(RuntimeError, match="below"):
        check(argparse.Namespace(
            log=log,
            output=tmp_path / "result.json",
            ignore_first_steps=10,
            full_training_tokens=27_732_738_048,
            min_tokens_per_sec=200_000,
            max_data_wait_fraction=0.10,
            max_projected_hours=42.0,
            overhead_fraction=0.10,
        ))
