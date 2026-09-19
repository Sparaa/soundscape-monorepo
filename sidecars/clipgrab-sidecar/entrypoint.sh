#!/bin/sh
# Every start: pull the newest yt-dlp release before serving. Network hiccups
# must not keep the service down — fall through to the baked version.
set -u
echo "==> clipgrab: upgrading yt-dlp (baked: $(yt-dlp --version 2>/dev/null || echo none))"
if pip install -q --upgrade yt-dlp 2>&1 | tail -n 2; then
  echo "==> clipgrab: yt-dlp $(yt-dlp --version)"
else
  echo "!! clipgrab: yt-dlp upgrade failed — serving with the baked version $(yt-dlp --version)" >&2
fi
mkdir -p "${SCRATCH_DIR:-/scratch}"
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-3014}" --no-access-log
