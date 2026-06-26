# Dockerfile — Minimal RunPod serverless image for Fish Speech
#
# Build:
#   docker build -t fish-speech-runpod .
#
# Run:
#   docker run --gpus all -p 8000:8000 fish-speech-runpod

# ── Args ────────────────────────────────────────────────────
ARG HF_TOKEN
ARG CUDA_VER=12.9.0
ARG UBUNTU_VER=24.04
ARG PY_VER=3.12
ARG UV_VERSION=0.8.15
ARG UV_EXTRA=cu129

# ── Stage 1: UV binary ─────────────────────────────────────
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

# ── Stage 2: Build & run ───────────────────────────────────
FROM nvidia/cuda:${CUDA_VER}-cudnn-runtime-ubuntu${UBUNTU_VER}

ARG HF_TOKEN
ARG PY_VER
ARG UV_EXTRA

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_TOKEN=${HF_TOKEN}

# System deps — single layer, minimal set for audio + networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git ca-certificates curl \
    libsox-dev ffmpeg portaudio19-dev libportaudio2 \
    build-essential cmake libasound-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# UV
COPY --from=uv-bin /uv /uvx /bin/

WORKDIR /app

# Download model weights into the image
RUN uvx hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro

RUN uv venv /app/.venv -p ${PY_VER}

ENV PATH="/app/.venv:$PATH"

# Deps first for layer caching
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --extra ${UV_EXTRA} --frozen --no-install-project

# App source
COPY . .
RUN uv sync --extra ${UV_EXTRA} --frozen
RUN uv pip install runpod

# ── Runtime config ──────────────────────────────────────────
ENV PORT=8000
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/ping || exit 1

CMD ["uv", "run", "python", "runpod.py"]
