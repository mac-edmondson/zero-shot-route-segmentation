FROM python:3.12-slim AS base

WORKDIR /app
COPY src/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
EXPOSE 8000
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS production
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

