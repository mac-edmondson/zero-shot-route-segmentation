#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
source /apps/python/3.12-miniforge/etc/profile.d/conda.sh
conda activate lit
cd "$ROOT_DIR"
python scripts/slurm/run_sam3_validation.py --root "$ROOT_DIR" --model-dir "$MODEL_DIR" --run-dir "$RUN_DIR"
