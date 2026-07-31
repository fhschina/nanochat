"""Capture a reproducibility fingerprint for one CW H100 training node."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
from typing import Any

import torch

from nanochat.flash_attention import HAS_FA3, USE_FA3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(*args: str) -> str:
    result = subprocess.run(args, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def stable_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
    if gpu_count != args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} visible GPUs, found {gpu_count}")
    unexpected = [name for name in gpu_names if args.expected_gpu_name not in name]
    if unexpected:
        raise RuntimeError(f"Expected {args.expected_gpu_name} GPUs, found {unexpected}")
    if args.require_fa3 and not USE_FA3:
        raise RuntimeError("Flash Attention 3 is required but is not active")
    shared_storage = None
    if args.shared_path:
        path = args.shared_path.expanduser().resolve()
        usage = shutil.disk_usage(path)
        shared_storage = {
            "path": str(path),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        }
        if usage.free < args.min_shared_free_gb * 1024**3:
            raise RuntimeError(
                f"Shared storage has {usage.free / 1024**3:.1f} GiB free; "
                f"at least {args.min_shared_free_gb} GiB is required"
            )
    driver = command(
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ).splitlines()
    nccl_version = torch.cuda.nccl.version()
    compatibility = {
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "compute_capabilities": [list(torch.cuda.get_device_capability(index)) for index in range(gpu_count)],
        "driver_versions": driver,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": list(nccl_version) if isinstance(nccl_version, tuple) else nccl_version,
        "torchao": package_version("torchao"),
        "kernels": package_version("kernels"),
        "has_fa3": HAS_FA3,
        "use_fa3": USE_FA3,
    }
    runtime = {
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_NODEID",
                "SLURM_NODELIST",
                "SLURM_PROCID",
            )
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_topology": command("nvidia-smi", "topo", "-m"),
        "shared_storage": shared_storage,
    }
    payload = {
        "format_version": 1,
        "compatibility": compatibility,
        "compatibility_sha256": stable_sha256(compatibility),
        "runtime": runtime,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--require-fa3", action="store_true")
    parser.add_argument("--shared-path", type=Path)
    parser.add_argument("--min-shared-free-gb", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    payload = capture(parse_args())
    print(json.dumps({
        "compatibility_sha256": payload["compatibility_sha256"],
        "hostname": payload["runtime"]["hostname"],
    }, indent=2))


if __name__ == "__main__":
    main()
