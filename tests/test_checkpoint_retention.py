from pathlib import Path

from nanochat.checkpoint_manager import prune_checkpoints


def generation(root: Path, step: int, ranks: int = 2) -> None:
    (root / f"model_{step:06d}.pt").touch()
    (root / f"meta_{step:06d}.json").touch()
    for rank in range(ranks):
        (root / f"meta_{step:06d}_rank{rank}.json").touch()
        (root / f"optim_{step:06d}_rank{rank}.pt").touch()


def test_checkpoint_pruning_keeps_two_regular_generations_plus_final(tmp_path: Path) -> None:
    for step in (1000, 2000, 3000, 4000):
        generation(tmp_path, step)
    unrelated = tmp_path / "train.log"
    unrelated.touch()
    result = prune_checkpoints(
        tmp_path,
        keep_last=2,
        preserve_steps={4000},
        exclude_preserved_from_limit=True,
    )
    assert result["kept_steps"] == [2000, 3000, 4000]
    assert not list(tmp_path.glob("*001000*"))
    assert (tmp_path / "model_002000.pt").is_file()
    assert (tmp_path / "model_003000.pt").is_file()
    assert (tmp_path / "model_004000.pt").is_file()
    assert unrelated.is_file()


def test_checkpoint_pruning_keeps_all_milestones_plus_two_regular(tmp_path: Path) -> None:
    for step in (1102, 2204, 3306, 4408, 5510, 6612, 7714, 8816):
        generation(tmp_path, step)
    result = prune_checkpoints(
        tmp_path,
        keep_last=2,
        preserve_steps={2204, 6612},
        exclude_preserved_from_limit=True,
    )
    assert result["kept_steps"] == [2204, 6612, 7714, 8816]
    assert (tmp_path / "model_002204.pt").is_file()
    assert (tmp_path / "model_006612.pt").is_file()
    assert (tmp_path / "model_007714.pt").is_file()
    assert (tmp_path / "model_008816.pt").is_file()
