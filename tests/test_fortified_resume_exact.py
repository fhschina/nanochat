import copy
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit


class FakeTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, texts, prepend=None, num_threads=1):
        return [[prepend] + [1 + (index % 15) for index in range(int(text))] for text in texts]


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.output = torch.nn.Linear(8, 16)

    def forward(self, inputs, targets):
        logits = self.output(self.embedding(inputs))
        return torch.nn.functional.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))


def write_data(path: Path) -> None:
    lengths = list(range(1, 25))
    pq.write_table(pa.table({
        "text": [str(value) for value in lengths],
        "removed_by_fuzzy": [value % 2 == 0 for value in lengths],
        "nanochat_token_count": [value + 1 for value in lengths],
    }), path, row_group_size=24)


def make_loader(path: Path, state=None):
    return tokenizing_distributed_data_loader_with_state_bos_bestfit(
        FakeTokenizer(), B=1, T=8, split="train", device="cpu",
        parquet_paths=[str(path)], tokenizer_batch_size=4, buffer_size=5,
        resume_state_dict=state,
    )


def train_step(model, optimizer, batch):
    inputs, targets, _ = batch
    optimizer.zero_grad(set_to_none=True)
    loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def validation_bpb(model) -> float:
    inputs = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    targets = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    return float(model(inputs, targets).detach()) / math.log(2)


def optimizer_steps(optimizer) -> set[int]:
    return {
        int(value["step"].item() if torch.is_tensor(value["step"]) else value["step"])
        for value in optimizer.state.values()
        if "step" in value
    }


def test_interrupted_resume_matches_uninterrupted_exactly(tmp_path: Path) -> None:
    data = tmp_path / "data.parquet"
    write_data(data)
    torch.manual_seed(42)
    continuous_model = TinyLM()
    continuous_optimizer = torch.optim.AdamW(continuous_model.parameters(), lr=1e-3)
    continuous_loader = make_loader(data)

    checkpoint = None
    final_continuous_state = None
    for step in range(6):
        batch = next(continuous_loader)
        train_step(continuous_model, continuous_optimizer, batch)
        if step == 2:
            checkpoint = {
                "model": copy.deepcopy(continuous_model.state_dict()),
                "optimizer": copy.deepcopy(continuous_optimizer.state_dict()),
                "dataloader": copy.deepcopy(batch[2]),
                "step": step + 1,
            }
        final_continuous_state = batch[2]
    assert checkpoint is not None

    replay_loader = make_loader(data, checkpoint["dataloader"])
    expected_inputs, expected_targets, _ = next(replay_loader)

    torch.manual_seed(999)
    resumed_model = TinyLM()
    resumed_model.load_state_dict(checkpoint["model"])
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_loader = make_loader(data, checkpoint["dataloader"])
    final_resumed_state = None
    for resumed_step in range(checkpoint["step"], 6):
        batch = next(resumed_loader)
        if resumed_step == checkpoint["step"]:
            assert torch.equal(batch[0], expected_inputs)
            assert torch.equal(batch[1], expected_targets)
        train_step(resumed_model, resumed_optimizer, batch)
        final_resumed_state = batch[2]

    max_abs = max(
        float((left - right).abs().max().detach())
        for left, right in zip(continuous_model.parameters(), resumed_model.parameters(), strict=True)
    )
    assert max_abs <= 1e-6
    assert validation_bpb(continuous_model) == validation_bpb(resumed_model)
    assert optimizer_steps(continuous_optimizer) == optimizer_steps(resumed_optimizer) == {6}
    assert final_resumed_state["source_state"] == final_continuous_state["source_state"]
    assert final_resumed_state["doc_buffer"] == final_continuous_state["doc_buffer"]
    assert final_resumed_state["data_stats"] == final_continuous_state["data_stats"]
