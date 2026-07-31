#!/usr/bin/env bash
#SBATCH --job-name=fwef-fuzzy-ab-4x-smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$repo_root/runs/fineweb_edu_fortified_fuzzy_ab_4x_cw.sh" paired-slurm smoke
