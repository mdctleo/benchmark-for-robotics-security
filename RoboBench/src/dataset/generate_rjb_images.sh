#!/bin/bash
#SBATCH --job-name=rjb-images
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH -p a100-80gb
#SBATCH -A antoniob
#SBATCH --qos=standby


#
# Usage, from the repo root:
#   sbatch src/dataset/generate_rjb_images.sh prompts [extra driver flags, e.g. --limit 3]
#   sbatch src/dataset/generate_rjb_images.sh images  [extra driver flags]
#
# Needs HF_TOKEN in .env (FLUX.2 weights are gated).
set -euo pipefail

STAGE="${1:?usage: $0 prompts|images [driver flags]}"
shift

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"
mkdir -p workdir/slurm

# uv sync --extra imagegen
uv run python src/dataset/generate_rjb_images.py data/main_dataset_50 --stage "$STAGE" "$@"
