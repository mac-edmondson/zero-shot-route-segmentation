#!/usr/bin/env bash
#SBATCH --job-name=route-eval-a40
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/route-eval-a40-%j.out
#SBATCH --error=logs/route-eval-a40-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
python -u - <<'PYTHON'
from pathlib import Path
import json
import os
import numpy as np
import torch
from PIL import Image
from pipeline.evaluation import ComponentSpec, MATRIX_ROUTE_SPECS, load_benchmark_records, run_evaluation_matrix, run_independent_evaluations
if not torch.cuda.is_available():
    raise RuntimeError("A40 Slurm job does not expose CUDA")
root = Path("results/gpu_a40")
root.mkdir(parents=True, exist_ok=True)
metadata = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gpu_name": torch.cuda.get_device_name(0), "cuda_device": torch.cuda.current_device(), "device": "cuda"}
(root / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
records = load_benchmark_records("data/evaluation/ground_truth_labels", split="test")

def gpu_routes():
    return tuple(ComponentSpec(s.name, {**s.config, "device": "cuda"}) for s in MATRIX_ROUTE_SPECS)

def exemplar_detector():
    p = Path("data/evaluation/exemplars")
    return ComponentSpec("SAM 3", {"device": "cuda", "exemplars": tuple((Image.open(p / f"exemplar_image_{i}.png").convert("RGB"), np.asarray(Image.open(p / f"exemplar_mask_{i}.png")) > 0) for i in range(1, 6))})

def run(label, detectors, independent_dir, matrix_dir):
    run_independent_evaluations(records, detectors=detectors, route_discriminators=gpu_routes(), output_dir=root / independent_dir, run_id=f"a40-{label}-independent")
    manifest = run_evaluation_matrix(records, detectors=detectors, route_discriminators=gpu_routes(), output_dir=root / matrix_dir, run_id=f"a40-{label}-matrix")
    print(label, "combinations", len(manifest["combinations"]), "errors", len(manifest["errors"]), flush=True)

run("standard", tuple(ComponentSpec(name, {"device": "cuda"}) for name in ("Mask-RCNN", "YOLO", "SAM 3")), "standard/independent", "standard/matrix")
run("text-exemplar", (exemplar_detector(),), "text_exemplar/independent", "text_exemplar/matrix")
PYTHON
