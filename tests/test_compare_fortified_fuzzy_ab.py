import argparse
import json
from pathlib import Path

from scripts.compare_fortified_fuzzy_ab import generate_report
from scripts.generate_fortified_fuzzy_pioneer import generate_pioneer_report


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def summary(arm: str, seed: int, bpb: float, core: float):
    return {
        "arm": arm,
        "seed": seed,
        "config": {"validation_sha256": "same"},
        "curves": {
            "val_bpb": [
                {"step": 0, "tokens": 0, "bpb": 1.0},
                {"step": 6612, "tokens": 6933184512, "bpb": bpb},
            ],
            "train_loss": [
                {"step": 0, "tokens": 1048576, "loss": 5.0, "runtime_sec": 1.0},
                {"step": 6611, "tokens": 6933184512, "loss": 2.0, "runtime_sec": 5000.0},
            ],
            "online_core": [{"step": 6612, "tokens": 6933184512, "core": core}],
        },
        "metrics": {
            "num_iterations": 6612,
            "training_tokens": 6933184512,
            "final_train_val_bpb": bpb,
            "final_eval_val_bpb": bpb + 0.001,
            "final_core": core,
            "training_time_sec": 5000.0,
            "final_data_exposure": {"fuzzy_removed_target_fraction": 0.44} if arm == "raw" else None,
        },
        "core_tasks": {"task": {"accuracy": core, "centered": core - 0.2}},
    }


def test_report_fixture_generates_required_outputs(tmp_path: Path):
    run_root = tmp_path / "runs"
    write_json(run_root / "full/fuzzy/seed_42/run_summary.json", summary("fuzzy", 42, 0.80, 0.25))
    write_json(run_root / "full/raw/seed_42/run_summary.json", summary("raw", 42, 0.81, 0.24))
    view = {
        "selection": {"docs": 2, "tokens": 20, "excluded_validation_overlap_docs": 0, "removed_token_fraction": 0.4},
        "by_subset": {"s": {"docs": 2, "tokens": 20}},
        "by_length": {"lt128": {"docs": 2, "tokens": 20}},
    }
    fuzzy_manifest, raw_manifest = tmp_path / "fuzzy.json", tmp_path / "raw.json"
    write_json(fuzzy_manifest, view); write_json(raw_manifest, view)
    dedup = tmp_path / "dedup.json"; write_json(dedup, {"stats": {"raw": {"docs": 4, "nanochat_tokens": 40}, "post_fuzzy": {"docs": 2, "nanochat_tokens": 20}}})
    component = tmp_path / "component.json"; write_json(component, {"components": {"component_size_histogram": {"2": 3, "3": 1}}})
    pair = tmp_path / "pairs.jsonl"; pair.write_text(json.dumps({"char_ngram_jaccard": 0.8, "component_size": 2, "removed_preview": "r", "keeper_preview": "k"}) + "\n")
    output = tmp_path / "report"
    args = argparse.Namespace(
        run_root=run_root, output_dir=output, fuzzy_manifest=fuzzy_manifest,
        raw_manifest=raw_manifest, dedup_manifest=dedup, component_audit=component,
        pair_audit=pair, allow_incomplete=True,
    )
    result = generate_report(args)
    assert result["completed_pairs"] == 1
    for name in (
        "final_report.md", "results.json", "run_results.csv", "val_bpb_vs_tokens.svg",
        "bpb_delta_vs_tokens.svg", "train_loss_vs_tokens.svg", "online_core_vs_tokens.svg",
        "val_bpb_vs_time.svg", "final_core.svg", "core_task_delta.svg",
        "component_size_histogram.svg", "jaccard_ecdf.svg",
    ):
        assert (output / name).is_file(), name



def test_pioneer_report_fixture_generates_fuzzy_only_outputs(tmp_path: Path):
    run_root = tmp_path / "runs"
    write_json(run_root / "full/fuzzy/seed_42/run_summary.json", summary("fuzzy", 42, 0.80, 0.25))
    fuzzy_manifest = tmp_path / "fuzzy.json"
    write_json(fuzzy_manifest, {
        "selection": {
            "docs": 2,
            "tokens": 20,
            "excluded_validation_overlap_docs": 1,
            "excluded_validation_overlap_tokens": 3,
        },
        "by_subset": {"s": {"docs": 2, "tokens": 20}},
        "by_length": {"lt128": {"docs": 2, "tokens": 20}},
    })
    dedup = tmp_path / "dedup.json"
    write_json(dedup, {"stats": {"post_fuzzy": {"docs": 2, "nanochat_tokens": 20}}})
    component = tmp_path / "component.json"
    write_json(component, {"components": {"component_size_histogram": {"2": 3}}})
    pair = tmp_path / "pairs.jsonl"
    pair.write_text(json.dumps({
        "char_ngram_jaccard": 0.8,
        "component_size": 2,
        "removed_preview": "r",
        "keeper_preview": "k",
    }) + "\n")
    output = tmp_path / "pioneer"
    args = argparse.Namespace(
        run_root=run_root,
        output_dir=output,
        fuzzy_manifest=fuzzy_manifest,
        dedup_manifest=dedup,
        component_audit=component,
        pair_audit=pair,
        seed=42,
    )
    result = generate_pioneer_report(args)
    assert result["report_type"] == "pioneer_fuzzy_only"
    assert result["comparison_available"] is False
    report = (output / "pioneer_report.md").read_text()
    assert "does not estimate the causal effect" in report
    assert "raw NanoChat arm" in report
    for name in (
        "pioneer_report.md", "final_report.md", "results.json", "run_results.csv",
        "val_bpb_vs_tokens.svg", "train_loss_vs_tokens.svg", "online_core_vs_tokens.svg",
        "val_bpb_vs_time.svg", "final_core.svg", "core_task_scores.svg",
        "component_size_histogram.svg", "jaccard_ecdf.svg",
    ):
        assert (output / name).is_file(), name
