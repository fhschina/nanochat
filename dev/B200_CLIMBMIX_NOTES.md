# B200 ClimbMix Notes

This branch keeps Karpathy nanochat close to upstream while adding the pieces used
for reproducing the ClimbMix leaderboard-style run on an 8x NVIDIA B200 node.

## Changes relative to karpathy/nanochat master

- Added Sarah Yurick's local Megatron `.bin/.idx` training path from her `climb`
  branch, including:
  - `nanochat/megatron_dataset.py` for mmap-backed MMIDIDX `.bin/.idx` reads.
  - `nanochat/megatron_dataloader.py` for BOS-aligned distributed loading with
    optional per-domain weighting.
  - `nanochat/llama_tokenizer.py` and `scripts/build_llama_tokenizer.py` for
    Llama2 SentencePiece tokenizer artifacts.
  - `scripts/prepare_pile_val.py` for optional Pile validation BPB evaluation.
  - `runs/megatron_pretrain.sh` for B200 Megatron-data pretraining.
- Extended `scripts/base_train.py` with `--data-source=megatron`, `--data-dir`,
  `--domain-weights`, `--train-fraction`, and optional `--pile-val-dir` support.
- Extended `scripts/base_eval.py` with the same Megatron BPB evaluation flags so
  `runs/megatron_pretrain.sh` can complete its final eval stage.
- Added `runs/speedrun_b200.sh`, a B200-friendly ClimbMix HF parquet reproduction
  script that sets `NCCL_NVLS_ENABLE=0` and uses `--window-pattern=L` because
  Blackwell currently uses the SDPA fallback in this nanochat setup.

## Reproduction result on 8x B200

Using `runs/speedrun_b200.sh` with `PARAM_DATA_RATIO=9.5`:

- Hardware: 8x NVIDIA B200
- Model tag: `d24-climbmix-b200-r9.5`
- Depth: 24
- FP8: enabled
- Window pattern: `L`
- Training steps: 6612
- Training tokens: 6,933,184,512
- Total training time: 5299.19s / 88.32m / 1.47h
- Minimum validation BPB during training: 0.7111379
- Final validation BPB during training: 0.7111379
- Full base eval validation BPB: 0.7085
- Full CORE metric: 0.264189

The CORE score is above the GPT-2 leaderboard threshold of 0.256525.
