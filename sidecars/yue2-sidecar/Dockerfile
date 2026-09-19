# vidmakr-music — YuE2 song generation sidecar (style + lyrics -> stereo song).
# Its own image because the yue2-infer runtime pins torch==2.10.0 and
# transformers==4.57.6, which the ai container's validated torch 2.7.1 stack
# (Whisper/SigLIP/InsightFace/RIFE) must not chase. GPU-only, one song at a
# time, weights in a named HF cache volume, outputs in a tmpfs scratch that is
# handed to vidmakr-ai over HTTP and then deleted — no host disk.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf-cache \
    SCRATCH_DIR=/scratch

# ffmpeg: optional mp3/m4a delivery of the FLAC master. curl: healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The yue2-infer runtime, pinned to the yue2-v0.1.6 tag (see PINNED.md).
# Its pyproject pins torch==2.10.0; the PyPI wheel bundles CUDA 12.8 libs, which
# carry sm_120 (Blackwell) kernels — no separate index needed.
ARG YUE2_REF=9c6c4b349be978b06a9d0d958471a07a6cdeff4d
RUN pip install --upgrade pip \
 && pip install "git+https://github.com/multimodal-art-projection/YuE.git@${YUE2_REF}"

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app.py ./

EXPOSE 3015
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3015"]
