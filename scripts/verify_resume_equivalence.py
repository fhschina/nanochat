"""Verify an interrupted NanoChat run exactly matches its uninterrupted control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AssertionError(f"Tensor metadata differs: {left.shape}/{left.dtype} vs {right.shape}/{right.dtype}")
    if left.numel() == 0:
        return 0.0
    if left.is_floating_point() or left.is_complex():
        if not torch.equal(torch.isnan(left), torch.isnan(right)):
            return float("inf")
        return float((left.float() - right.float()).abs().nan_to_num().max().item())
    return 0.0 if torch.equal(left, right) else float("inf")


def recursive_max_abs(left: Any, right: Any, path: str = "root") -> float:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return tensor_max_abs(left, right)
    if type(left) is not type(right):
        raise AssertionError(f"{path}: type differs: {type(left)} vs {type(right)}")
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f"{path}: dict keys differ")
        return max((recursive_max_abs(left[key], right[key], f"{path}.{key}") for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: sequence lengths differ")
        return max((recursive_max_abs(a, b, f"{path}[{index}]") for index, (a, b) in enumerate(zip(left, right, strict=True))), default=0.0)
    if isinstance(left, float):
        return abs(left - right)
    if left != right:
        raise AssertionError(f"{path}: value differs: {left!r} vs {right!r}")
    return 0.0


def load_meta(checkpoint_dir: Path, step: int, rank: int) -> dict[str, Any]:
    ranked = checkpoint_dir / f"meta_{step:06d}_rank{rank}.json"
    path = ranked if ranked.is_file() else checkpoint_dir / f"meta_{step:06d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def final_bpb(path: Path) -> float:
    values = re.findall(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([0-9.]+)", path.read_text(errors="replace"))
    if not values:
        raise RuntimeError(f"No validation BPB in {path}")
    return float(values[-1][1])


def batch_audits(path: Path) -> dict[int, list[str]]:
    audits = {}
    for payload in re.findall(r"DATA_BATCH_AUDIT\s+(\{[^\n]+\})", path.read_text(errors="replace")):
        row = json.loads(payload)
        audits[int(row["step"])] = list(row["rank_sha256"])
    return audits


def optimizer_steps(state: Any) -> set[int]:
    steps: set[int] = set()
    if isinstance(state, dict):
        for key, value in state.items():
            if key == "step":
                if torch.is_tensor(value) and value.numel() == 1:
                    steps.add(int(value.item()))
                elif isinstance(value, (int, float)):
                    steps.add(int(value))
            else:
                steps.update(optimizer_steps(value))
    elif isinstance(state, (list, tuple)):
        for value in state:
            steps.update(optimizer_steps(value))
    return steps


def verify(args: argparse.Namespace) -> dict[str, Any]:
    reference = args.reference_checkpoint_dir.expanduser().resolve()
    resumed = args.resumed_checkpoint_dir.expanduser().resolve()
    tolerance = float(args.parameter_tolerance)
    reference_model = torch.load(reference / f"model_{args.step:06d}.pt", map_location="cpu")
    resumed_model = torch.load(resumed / f"model_{args.step:06d}.pt", map_location="cpu")
    model_max_abs = recursive_max_abs(reference_model, resumed_model, "model")
    if model_max_abs > tolerance:
        raise RuntimeError(f"Model max abs diff {model_max_abs} exceeds {tolerance}")

    optimizer_max_abs = 0.0
    per_rank = []
    for rank in range(args.world_size):
        reference_optimizer = torch.load(reference / f"optim_{args.step:06d}_rank{rank}.pt", map_location="cpu")
        resumed_optimizer = torch.load(resumed / f"optim_{args.step:06d}_rank{rank}.pt", map_location="cpu")
        rank_diff = recursive_max_abs(reference_optimizer, resumed_optimizer, f"optimizer.rank{rank}")
        optimizer_max_abs = max(optimizer_max_abs, rank_diff)
        ref_steps = optimizer_steps(reference_optimizer)
        resumed_steps = optimizer_steps(resumed_optimizer)
        if ref_steps != resumed_steps or (ref_steps and ref_steps != {args.step}):
            raise RuntimeError(f"rank {rank}: optimizer steps differ: {ref_steps} vs {resumed_steps}")
        ref_meta = load_meta(reference, args.step, rank)
        resumed_meta = load_meta(resumed, args.step, rank)
        if int(ref_meta["step"]) != args.step or int(resumed_meta["step"]) != args.step:
            raise RuntimeError(f"rank {rank}: checkpoint metadata step mismatch")
        if ref_meta["dataloader_state_dict"] != resumed_meta["dataloader_state_dict"]:
            raise RuntimeError(f"rank {rank}: dataloader cursor/buffer differs")
        if ref_meta["loop_state"]["consumed_data_stats"] != resumed_meta["loop_state"]["consumed_data_stats"]:
            raise RuntimeError(f"rank {rank}: consumed data stats differ")
        per_rank.append({"rank": rank, "optimizer_steps": sorted(ref_steps), "optimizer_max_abs": rank_diff})
    if optimizer_max_abs > tolerance:
        raise RuntimeError(f"Optimizer max abs diff {optimizer_max_abs} exceeds {tolerance}")

    reference_bpb = final_bpb(args.reference_log)
    resumed_bpb = final_bpb(args.resumed_log)
    if abs(reference_bpb - resumed_bpb) > tolerance:
        raise RuntimeError(f"Final validation BPB differs: {reference_bpb} vs {resumed_bpb}")
    reference_audits = batch_audits(args.reference_log)
    resumed_audits = batch_audits(args.resumed_log)
    missing = sorted(set(args.required_batch_step) - (set(reference_audits) & set(resumed_audits)))
    if missing:
        raise RuntimeError(f"Missing batch audits for steps: {missing}")
    for step in args.required_batch_step:
        if reference_audits[step] != resumed_audits[step]:
            raise RuntimeError(f"Step {step}: resumed batch hashes differ from uninterrupted control")

    result = {
        "format_version": 1,
        "status": "passed",
        "verified_at": utc_now(),
        "step": args.step,
        "world_size": args.world_size,
        "parameter_tolerance": tolerance,
        "model_max_abs_diff": model_max_abs,
        "optimizer_max_abs_diff": optimizer_max_abs,
        "final_validation_bpb": reference_bpb,
        "batch_audits": {str(step): reference_audits[step] for step in args.required_batch_step},
        "per_rank": per_rank,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resumed-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--resumed-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=954)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--required-batch-step", type=int, action="append")
    parser.add_argument("--parameter-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.required_batch_step is None:
        args.required_batch_step = [477]
    return args


def main() -> None:
    print(json.dumps(verify(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
