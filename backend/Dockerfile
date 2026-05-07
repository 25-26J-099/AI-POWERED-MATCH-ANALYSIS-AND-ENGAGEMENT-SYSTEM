# ── Backend Dockerfile ────────────────────────────────────────────────────────
# Base: CUDA 12.4.1 + cuDNN 9 on Ubuntu 22.04 (compatible with A100 GPU).
# Single worker process shares the GPU across async handlers to avoid OOM.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    sox \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/pip
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip     pip     /usr/bin/pip3      1

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

# Copy requirements first so this layer is cached independently of app code
COPY requirements-linux.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-linux.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Runtime environment ───────────────────────────────────────────────────────
# Binary paths — ffmpeg and sox are on the system PATH inside the container.
# Windows hardcoded .exe paths in .env are overridden by these ENV values.
ENV FFMPEG_BINARY=ffmpeg \
    SOX_BINARY=sox \
    OLLAMA_BINARY=ollama \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell PaddlePaddle not to show device warnings in logs
    FLAGS_use_cuda_managed_memory=0

EXPOSE 8000

# Single worker: GPU memory is shared across all async requests in one process.
# Increase --timeout-keep-alive for long video uploads.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "600"]
