from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nanochat.dataloader import (
    _selected_parquet_paths,
    resolve_parquet_paths,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)


class FakeTokenizer:
    def get_bos_token_id(self):
        return 99

    def encode(self, texts, prepend=None, num_threads=1):
        return [[prepend] + list(range(1, int(text) + 1)) for text in texts]


def write_data(path: Path, lengths, removed=None, row_group_size=2):
    data = {
        "text": [str(value) for value in lengths],
        "nanochat_token_count": [value + 1 for value in lengths],
    }
    if removed is not None:
        data["removed_by_fuzzy"] = removed
    pq.write_table(pa.table(data), path, row_group_size=row_group_size)


def test_explicit_paths_use_every_file_while_legacy_splits_last(tmp_path: Path):
    write_data(tmp_path / "a.parquet", [1, 2])
    write_data(tmp_path / "b.parquet", [3, 4])
    paths = resolve_parquet_paths(tmp_path)
    assert len(paths) == 2
    assert _selected_parquet_paths("train", None, paths, False) == paths
    assert _selected_parquet_paths("val", None, paths, False) == paths
    assert _selected_parquet_paths("train", str(tmp_path), None, False) == paths[:-1]
    assert _selected_parquet_paths("val", str(tmp_path), None, False) == paths[-1:]


def test_removed_target_and_crop_attribution(tmp_path: Path):
    path = tmp_path / "data.parquet"
    write_data(path, [5, 6], [True, False])
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        FakeTokenizer(), B=1, T=4, split="train", device="cpu",
        parquet_paths=[str(path)], tokenizer_batch_size=2, buffer_size=2,
    )
    _, _, state = next(loader)
    assert state["batch_stats"]["target_tokens"] == 4
    assert state["batch_stats"]["removed_target_tokens"] == 4
    assert state["data_stats"]["cropped_tokens"] == 1
    assert state["data_stats"]["removed_cropped_tokens"] == 1
    assert state["data_stats"]["removed_source_docs"] == 1


def test_resume_state_produces_same_next_batch(tmp_path: Path):
    path = tmp_path / "data.parquet"
    write_data(path, list(range(1, 13)), [value % 2 == 0 for value in range(1, 13)], row_group_size=12)
    kwargs = dict(
        tokenizer=FakeTokenizer(), B=1, T=8, split="train", device="cpu",
        parquet_paths=[str(path)], tokenizer_batch_size=3, buffer_size=4,
    )
    continuous = tokenizing_distributed_data_loader_with_state_bos_bestfit(**kwargs)
    first_x, first_y, checkpoint_state = next(continuous)
    replay = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        **kwargs, resume_state_dict=checkpoint_state["resume_state_before_batch"]
    )
    replay_x, replay_y, _ = next(replay)
    assert torch.equal(replay_x, first_x)
    assert torch.equal(replay_y, first_y)
    expected_x, expected_y, expected_state = next(continuous)
    expected_x, expected_y = expected_x.clone(), expected_y.clone()

    resumed = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        **kwargs, resume_state_dict=checkpoint_state
    )
    actual_x, actual_y, actual_state = next(resumed)
    assert torch.equal(actual_x, expected_x)
    assert torch.equal(actual_y, expected_y)
    assert actual_state["source_state"] == expected_state["source_state"]
    assert actual_state["data_stats"] == expected_state["data_stats"]
