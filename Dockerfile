FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MOYO_CLOUD_WORK_DIR=/tmp/moyo

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libmagic1 \
        libgomp1 \
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

COPY . .

RUN pip install --no-cache-dir ".[reports,cloud]"

ENTRYPOINT ["python", "cloud_worker.py"]
