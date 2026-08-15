#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/vault/v123be/v123be56/LIT/models/sam3}"
RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$ROOT_DIR/logs/slurm/$RUN_ID"
mkdir -p "$RUN_DIR"
declare -a JOB_IDS=()
submit() {
  local partition="$1" gres="$2" job_id
  job_id="$(sbatch --parsable --hold --job-name="sam3-$partition" --partition="$partition" --gres="$gres" --time=01:00:00 --output="$RUN_DIR/%x-%j.out" --error="$RUN_DIR/%x-%j.err" --export="ALL,ROOT_DIR=$ROOT_DIR,RUN_DIR=$RUN_DIR,MODEL_DIR=$MODEL_DIR" "$ROOT_DIR/scripts/slurm/sam3_validation.sbatch")"
  JOB_IDS+=("${job_id%%;*}")
}
submit a40 gpu:a40:1
submit a100 gpu:a100:1
submit rtxpro6k gpu:rtxpro6k:1
submit a100mig gpu:a100med:1
printf '%s\n' "${JOB_IDS[@]}" > "$RUN_DIR/job_ids.txt"
for job_id in "${JOB_IDS[@]}"; do scontrol release "$job_id"; done
printf 'run_dir=%s\njob_ids=%s\n' "$RUN_DIR" "${JOB_IDS[*]}"
