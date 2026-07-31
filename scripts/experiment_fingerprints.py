"""Stable content fingerprints for NanoChat experiment code."""

from __future__ import annotations

import hashlib
from pathlib import Path


def training_code_sha256(repo_root: Path | None = None) -> str:
    """Hash the Python training/evaluation implementation, including dirty edits."""
    root = repo_root or Path(__file__).resolve().parents[1]
    paths = sorted((root / "nanochat").glob("*.py"))
    paths.extend([root / "scripts" / "base_train.py", root / "scripts" / "base_eval.py"])
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()
