#!/bin/bash
#SBATCH --job-name=YOLO_job
#SBATCH --output=training/yolov8-hold-detector/YOLO_job_%j.out
#SBATCH --error=training/yolov8-hold-detector/YOLO_job_%j.err

#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00

module purge
module add python
source "/etc/profile.d/conda.sh"
conda activate lit

cd "."

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Starting at: $(date)"

python training/yolov8-hold-detector/yolov8_trainer.py

echo "Finished at: $(date)"
