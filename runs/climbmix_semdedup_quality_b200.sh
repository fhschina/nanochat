#!/bin/bash

# ClimbMix entrypoint for the shared SemDeDup quality A/B runner.

set -euo pipefail

export DATASET_TAG="${DATASET_TAG:-climbmix}"
export NANOCHAT_DATASET_NAME="${NANOCHAT_DATASET_NAME:-climbmix}"

exec bash runs/semdedup_quality_b200.sh "$@"
