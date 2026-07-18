# Snooker AI — CPU image (GPU: use Dockerfile.gpu)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY snooker_ai ./snooker_ai
COPY apps ./apps
COPY configs ./configs
COPY training ./training
COPY annotation ./annotation
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install -e .

RUN mkdir -p /app/data/uploads /app/data/jobs /app/data/outputs /app/models

EXPOSE 8000

VOLUME ["/app/data", "/app/models"]

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
