import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.prepare_contamination_union import build, link_tree


def write_parquet(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": [text]}), path)


def test_union_defaults_to_hardlinks(tmp_path: Path) -> None:
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    output = tmp_path / "union"
    write_parquet(training / "train.parquet", "train")
    write_parquet(evaluation / "eval.parquet", "eval")
    result = build(argparse.Namespace(
        training_root=training,
        eval_root=evaluation,
        output_dir=output,
        link_mode="hardlink",
        overwrite=False,
    ))
    assert {row["method"] for row in result["training_files"] + result["eval_files"]} == {"hardlink"}
    assert os.stat(training / "train.parquet").st_ino == os.stat(output / "training" / "train.parquet").st_ino
    manifest = json.loads((output / "contamination_union_manifest.json").read_text())
    assert manifest["config"]["link_mode"] == "hardlink"


def test_hardlink_mode_never_falls_back_to_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    write_parquet(source / "data.parquet", "data")

    def fail_link(_source, _destination):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(RuntimeError, match="Hardlink required"):
        link_tree(source, output, "training", link_mode="hardlink")
    assert not (output / "training" / "data.parquet").is_file()
