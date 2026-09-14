# vidmakr-sheetsage — SheetSage2 music transcription sidecar (song audio ->
# melody-only ABC score for YuE2 cover mode). Its own image because SheetSage2
# pins torch==2.8.0 / transformers==4.45.2 / numpy==1.24.3 on python 3.10-3.11,
# which neither the ai image (torch 2.7.1) nor the music image (torch 2.10)
# can host. GPU-only, one transcription at a time, weights in a named HF cache
# volume, inputs/outputs in a tmpfs scratch — no host disk.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf-cache \
    SCRATCH_DIR=/scratch

# ffmpeg: SheetSage2 decodes via torchaudio's ffmpeg backend or the ffmpeg CLI
# fallback (any input format → 24 kHz). curl: healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# SheetSage2's own requirements.txt (pins verified 2026-09-13). The PyPI torch
# 2.8.0 wheel bundles CUDA 12.8 libs → sm_120 kernels for the RTX PRO 6000.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py ./

EXPOSE 3016
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3016"]
