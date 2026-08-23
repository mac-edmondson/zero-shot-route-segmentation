FROM python:3.12 AS builder

# Install git for fetching anything which requires it in the requirements file.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/backend/*requirements.txt ./
RUN python -m venv /venv
RUN /venv/bin/pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /venv/bin/pip uninstall -y opencv-python opencv-python-headless \
    && /venv/bin/pip install --no-cache-dir opencv-python-headless
RUN /venv/bin/pip install --no-cache-dir --no-build-isolation -r detectron.requirements.txt


FROM python:3.12-slim AS base

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Ensure models needed for runtime inference are baked into the image
ENV TORCH_HOME=/opt/torch-cache
RUN python - <<'PY'
from torchvision.models import resnet50, ResNet50_Weights
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
PY

COPY models ./models
COPY src ./src

EXPOSE 8000
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS production
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

