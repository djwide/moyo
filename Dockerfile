# Cloud Run worker only: Firestore order → explore APIs → report PDFs → GCS.
# Does not ship torch / FAISS / GUI / ingest. Desktop corpus work stays local.
# PDF and Word text extraction is included for MoyoMap document import.

# Stage 1: install Python deps (no compilers leaked into the runtime image)
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY moyo ./moyo
COPY shared_utils ./shared_utils
COPY reports ./reports
COPY config ./config
COPY cloud_worker.py report_validation.py ./

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir ".[reports,cloud,documents]"

# Stage 2: WeasyPrint system libs + installed venv + worker source
FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/venv/bin:$PATH
ENV MOYO_CLOUD_WORK_DIR=/tmp/moyo
ENV MOYO_CLOUD_RUNTIME=1
ENV MOYO_UTILITY_PROVIDER=custom
ENV MOYO_UTILITY_MODEL=google/gemini-2.5-flash
ENV GOOGLE_CLOUD_PROJECT=senteguard-website
ENV MOYO_VERTEX_GEMINI=1
ENV MOYO_VERTEX_GEMINI_MODEL=google/gemini-2.5-pro
ENV MOYO_VERTEX_UTILITY_MODEL=google/gemini-2.5-flash

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libpangoft2-1.0-0 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        shared-mime-info \
        fonts-liberation \
        fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

ENTRYPOINT ["python", "cloud_worker.py"]
