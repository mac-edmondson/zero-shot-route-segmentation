"""Run local SAM wrapper and hold evaluation inside a SLURM job."""
from __future__ import annotations
import argparse, os, subprocess, sys, time, traceback
from pathlib import Path

def log(message: str) -> None:
    print(f"[sam3-validation] {message}", flush=True)

def command(args: list[str], cwd: Path) -> None:
    log(f"Running: {' '.join(args)}")
    subprocess.run(args, cwd=cwd, check=True)

def environment(model_dir: Path) -> None:
    import torch, transformers
    log(f"host={os.uname().nodename} job={os.environ.get('SLURM_JOB_ID')} partition={os.environ.get('SLURM_JOB_PARTITION')}")
    log(f"torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__} cuda_available={torch.cuda.is_available()}")
    for index in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(index)
        log(f"gpu[{index}]={gpu.name} memory_gb={gpu.total_memory / 1024**3:.1f}")
    log(f"model_dir={model_dir} model_size_bytes={(model_dir / 'model.safetensors').stat().st_size}")
    subprocess.run(["nvidia-smi"], check=False)

def wrapper_smoke(root: Path, model_dir: Path) -> None:
    from PIL import Image
    sys.path.insert(0, str(root / "src" / "SAM"))
    from sam_wrapper import SAMWrapper
    from test_sam_wrapper import HOLD_CLICKS
    path = root / "src" / "SAM" / "data" / "climbing-wall-climbing-wall-with-colorful-rocks-photo.jpg"
    with Image.open(path) as image:
        sam = SAMWrapper(model_dir=model_dir, device="cuda")
        started = time.monotonic(); sam.load_model()
        log(f"wrapper_model_load_seconds={time.monotonic() - started:.2f}")
        started = time.monotonic(); masks = sam.generate_mask(image, HOLD_CLICKS)
        log(f"wrapper_inference_seconds={time.monotonic() - started:.2f} masks_shape={masks.shape}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        environment(args.model_dir)
        wrapper_smoke(args.root, args.model_dir)
        command([sys.executable, "-m", "pytest", "-q", "src/SAM_hold/test_dataset.py"], args.root)
        command([sys.executable, "-m", "src.SAM_hold.evaluate", "--device", "cuda", "--model-dir", str(args.model_dir), "--output-dir", str(args.run_dir / "results")], args.root)
    except Exception:
        log("VALIDATION_FAILED"); traceback.print_exc(); raise
    (args.run_dir / "SUCCESS").write_text(f"job_id={os.environ.get('SLURM_JOB_ID', 'local')}\n")
    log(f"VALIDATION_SUCCEEDED total_seconds={time.monotonic() - started:.2f}")
    current = os.environ.get("SLURM_JOB_ID")
    ids = (args.run_dir / "job_ids.txt").read_text().split() if (args.run_dir / "job_ids.txt").is_file() else []
    siblings = [job_id for job_id in ids if job_id != current]
    if siblings:
        log(f"Cancelling siblings: {' '.join(siblings)}"); subprocess.run(["scancel", *siblings], check=False)
if __name__ == "__main__":
    main()
