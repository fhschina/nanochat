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


def test_removed_target_and_continuation_attribution(tmp_path: Path):
    path = tmp_path / "data.parquet"
    write_data(path, [5, 6], [True, False])
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        FakeTokenizer(), B=1, T=4, split="train", device="cpu",
        parquet_paths=[str(path)], tokenizer_batch_size=2, buffer_size=2,
    )
    _, _, state = next(loader)
    assert state["state_version"] == 2
    assert state["batch_stats"]["target_tokens"] == 4
    assert state["batch_stats"]["removed_target_tokens"] == 4
    assert state["data_stats"]["cropped_tokens"] == 0
    assert state["data_stats"]["discarded_source_tokens"] == 0
    assert state["data_stats"]["continuation_fragments"] == 1
    assert state["data_stats"]["removed_source_docs"] == 1
    continuation = next(item for item in state["doc_buffer"] if item["continuation"])
    assert continuation["removed_by_fuzzy"] is True
    assert len(continuation["tokens"]) == 1


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


def test_long_document_remainders_are_never_discarded(tmp_path: Path):
    path = tmp_path / "long.parquet"
    write_data(path, [20], [True], row_group_size=1)
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        FakeTokenizer(), B=1, T=4, split="train", device="cpu",
        parquet_paths=[str(path)], tokenizer_batch_size=1, buffer_size=1,
    )
    states = []
    for _ in range(4):
        _, _, state = next(loader)
        states.append(state)
        buffered_tokens = sum(len(item["tokens"]) for item in state["doc_buffer"])
        assert state["data_stats"]["source_tokens"] == (
            state["data_stats"]["consumed_source_tokens"] + buffered_tokens
        )
        assert state["data_stats"]["consumed_source_tokens"] == (
            state["data_stats"]["input_tokens"]
            + state["data_stats"]["row_boundary_input_only_tokens"]
        )
        assert state["data_stats"]["discarded_source_tokens"] == 0
        assert state["consumed_epoch"] == 1
    assert states[-1]["epoch"] == 2  # source cursor may prefetch ahead
    _, _, rollover = next(loader)
    assert rollover["consumed_epoch"] == 2


def test_legacy_resume_state_is_upgraded_and_audited(tmp_path: Path):
    path = tmp_path / "legacy.parquet"
    write_data(path, [8], [False], row_group_size=1)
    legacy = {
        "source_state": {"pq_idx": 0, "rg_idx": 0, "doc_offset": 0, "epoch": 1},
        "doc_buffer": [{"tokens": [99, 1, 2, 3, 4], "removed_by_fuzzy": True}],
        "data_stats": {"cropped_tokens": 7, "removed_cropped_tokens": 7},
    }
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        FakeTokenizer(), B=1, T=4, split="train", device="cpu",
        parquet_paths=[str(path)], tokenizer_batch_size=1, buffer_size=1,
        resume_state_dict=legacy,
    )
    _, _, state = next(loader)
    assert state["state_version"] == 2
    assert state["data_stats"]["discarded_source_tokens"] == 7

def test_prefetched_next_epoch_is_not_consumed_early(tmp_path: Path):
    path = tmp_path / "epoch.parquet"
    write_data(path, [8], [False], row_group_size=1)
    resume = {
        "state_version": 2,
        "source_state": {"pq_idx": 0, "rg_idx": 0, "doc_offset": 0, "epoch": 2},
        "consumed_epoch": 1,
        "doc_buffer": [
            {
                "tokens": [99, 1, 2, 3, 4, 5],
                "removed_by_fuzzy": False,
                "continuation": False,
                "source_epoch": 1,
            },
            {
                "tokens": [99, 6, 7, 8, 9],
                "removed_by_fuzzy": False,
                "continuation": False,
                "source_epoch": 2,
            },
        ],
        "data_stats": {},
    }
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        FakeTokenizer(), B=1, T=4, split="train", device="cpu",
        parquet_paths=[str(path)], tokenizer_batch_size=1, buffer_size=2,
        resume_state_dict=resume,
    )
    _, _, state = next(loader)
    assert state["consumed_epoch"] == 1
    assert any(item["source_epoch"] == 2 for item in state["doc_buffer"])
