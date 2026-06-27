FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ARG HF_TOKEN

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    UV_COMPILE_BYTECODE=1 \
    HF_TOKEN=${HF_TOKEN}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ca-certificates curl \
    libsox-dev ffmpeg portaudio19-dev libportaudio2 \
    build-essential cmake libasound-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /uvx /bin/

WORKDIR /app

RUN uvx hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro && \
    uv venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

COPY . .
RUN uv pip install --no-cache-dir . \
    --index https://download.pytorch.org/whl/cu129 --index https://pypi.nvidia.com
RUN uv pip install -U runpod

CMD ["python", "-u", "handler.py"]
