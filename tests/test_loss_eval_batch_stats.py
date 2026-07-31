import math

import torch

from nanochat.loss_eval import evaluate_bpb


class ConstantLossModel:
    def get_device(self):
        return torch.device("cpu")

    def __call__(self, _x, y, loss_reduction="none"):
        assert loss_reduction == "none"
        return torch.ones_like(y, dtype=torch.float32)


def test_evaluate_bpb_can_return_reduced_batch_numerators() -> None:
    batches = [
        (torch.tensor([[1, 2]]), torch.tensor([[1, 2]])),
        (torch.tensor([[2, 1]]), torch.tensor([[2, 1]])),
    ]
    token_bytes = torch.tensor([0, 1, 2], dtype=torch.int64)
    value, details = evaluate_bpb(
        ConstantLossModel(),
        batches,
        steps=2,
        token_bytes=token_bytes,
        return_batch_stats=True,
    )
    assert details == [
        {"nats": 2.0, "bytes": 3},
        {"nats": 2.0, "bytes": 3},
    ]
    assert value == 4.0 / (math.log(2) * 6)
