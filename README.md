# Robust Indoor Climbing Route Recognition

**ROUTNet** is a modular computer-vision pipeline and interactive web demo that finds climbing holds in a photo of a bouldering wall and groups them into routes. It also measures how robust that grouping is to chalk, lighting, and similar-color distortions.

Joswin Dsouza · Macallyster Edmondson · Praveen Narayan Hegde

Department of Computer Science, UTN (2026)

---

For detailed methodology, experimental setup, and benchmark evaluation results, refer to [our report](report/report.pdf).

---

## Interactive Dashboard

![Interactive Dashboard Demo](docs/diagrams/dashboard.gif)

### Capabilities & Usage

The interactive web dashboard lets you test the end-to-end route segmentation pipeline directly on climbing wall images:

1. **Upload & Sample Images:** Load wall images via file upload, webcam, or from the built-in sample gallery.
2. **Interactive Hold Segmentation & Augmentations:** Click on holds to segment them with SAM 3 point prompts. Apply realistic distortions (per-hold chalking, recoloring, and global lighting shifts) to test model robustness.
3. **Route Recognition Pipeline:** Select any registered hold detector (e.g., YOLOv8, SAM 3, Mask R-CNN, or ground truth) and route discriminator (e.g., DINOv3 clustering, DINO learning pair-head, or color-only baseline), run inference, and inspect the resulting hold polygons and grouped route overlays.

### Launching Locally

#### Prerequisites

- Docker and Docker Compose
- **If pulled from GitHub:** Git LFS (pull model weights via `git lfs pull`)

#### Run with Docker Compose

**Linux / Standard:**

```bash
docker compose up --build
```

**macOS:** *(Necessary workaround for CORS browser issues on some versions of MacOS.)*

```bash
./compose-up-mac.sh
```

Once running, open **<http://localhost:8080>** in your browser.

---

## Evaluation

The evaluation suite benchmarks hold detection accuracy (mAP, Precision, Recall) and route discrimination clustering quality (Pairwise F1, ARI, NMI) across clean and augmented wall images.

### Reproducing Benchmarks

```bash
# Independent component evaluation:
# detector mAP and route discrimination using ground-truth holds
PYTHONPATH=src python -m pipeline.evaluation.run_all_eval \
  --independent \
  --dataset-root data/evaluation/ground_truth_labels \
  --split test \
  --output-dir results/independent

# Clean end-to-end matrix:
# detector predictions paired with every configured route discriminator
PYTHONPATH=src python -m pipeline.evaluation.run_all_eval \
  --dataset-root data/evaluation/ground_truth_labels \
  --split test \
  --output results/evaluation_matrix/manifest.json
```

### Environment Setup

Perform one of the following depending on your system.

#### Using venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### Using uv

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

### Running Tests

```bash
PYTHONPATH=.:src uv run pytest src/tests
```

## Repository Structure

```text
zero-shot-route-segmentation/
├── README.md                     # Project overview and quickstart
├── docker-compose*.yaml          # Compose configurations and mac overrides
├── pyproject.toml, uv.lock       # Python dependencies managed with uv
├── data/                         # Evaluation datasets, ground-truth labels, and exemplars
├── docs/                         # Specifications, evaluation notes, and architecture diagrams
├── models/                       # Tracked model weights (Git LFS)
├── src/
│   ├── backend/                  # FastAPI service powering the web demo
│   ├── frontend/                 # Next.js / React interactive dashboard
│   ├── nginx/                    # Reverse-proxy configurations
│   ├── pipeline/                 # Core computer vision pipeline
│   │   ├── hold_detector/        # Hold detector implementations (YOLOv8, SAM 3, Mask R-CNN, etc.)
│   │   ├── route_discriminator/  # Route grouping models (DINOv3, color baseline, etc.)
│   │   ├── preprocessing/        # Augmentation suite (chalk, lighting, color)
│   │   ├── utility/              # Backbones, embeddings, and loaders
│   │   └── evaluation/           # Benchmark runners and metrics calculation
│   └── tests/                    # Pytest test suite
└── training/                     # Training scripts for detectors and discriminators
```
