# vidmakr-clipgrab — fetch publicly reachable web video (YouTube first) as
# H.264 mp4 reference clips for vidmakr. Standalone: no GPU, no model access,
# no host disk. yt-dlp is upgraded to the latest release EVERY container start
# (entrypoint.sh) because YouTube breaks extractors every few weeks — the
# version baked at image build time is only the offline fallback.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SCRATCH_DIR=/scratch

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# yt-dlp needs a JavaScript runtime for YouTube's signature/n-param challenges
# (2025+). Deno is the runtime yt-dlp recommends; copied from the official
# binary image so no install script runs at build time.
COPY --from=denoland/deno:bin-2.4.3 /deno /usr/local/bin/deno

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 3014
ENTRYPOINT ["./entrypoint.sh"]
