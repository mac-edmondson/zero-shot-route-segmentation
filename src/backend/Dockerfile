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
# Do some tom foolery to try and ensure detectron2 install get's cached cause it takes a little while.
RUN /venv/bin/pip install --no-cache-dir -r pre-detectron.requirements.txt
RUN /venv/bin/pip install --no-cache-dir --no-build-isolation -r detectron.requirements.txt

# Ensure models needed for runtime inference are baked into the image
ENV TORCH_HOME=/opt/torch-cache
RUN /venv/bin/python - <<'PY'
from torchvision.models import resnet50, ResNet50_Weights
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
PY

# Install the rest of the packages
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS base

WORKDIR /app

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

COPY --from=builder /opt/torch-cache /opt/torch-cache
ENV TORCH_HOME=/opt/torch-cache


COPY src ./src

# VOLUME /app/models # A volume must be bind-mounted into this image to work!

EXPOSE 8000
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS production
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

