"""vidmakr-clipgrab — a thin HTTP face over yt-dlp + ffmpeg.

Fetches publicly reachable web video (YouTube first; anything yt-dlp knows)
as an H.264 mp4 so vidmakr can use it as a MOTION reference (MiniMax H3 R2V)
and as editor media. Only the api container talks to this service; it never
touches the GPU, the model library, or the host disk — every download lands
in a per-request scratch dir (tmpfs) that is removed once the bytes are sent.

Routes
  GET  /healthz          → {ok, yt_dlp, js_runtime}
  GET  /info?url=        → title / duration / uploader / thumbnail
  POST /clip             → {url, start?, end?}  ⇒ video/mp4 body
  POST /update           → upgrade yt-dlp in place, return the version
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shutil
import subprocess
import tempfile
import base64
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch")
MAX_CLIP_SEC = float(os.environ.get("MAX_CLIP_SEC", "600"))  # refuse >10 min pulls
MAX_HEIGHT = int(os.environ.get("MAX_HEIGHT", "1080"))
INFO_TIMEOUT_SEC = float(os.environ.get("INFO_TIMEOUT_SEC", "90"))
FETCH_TIMEOUT_SEC = float(os.environ.get("FETCH_TIMEOUT_SEC", "900"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))

app = FastAPI(title="vidmakr-clipgrab", docs_url=None, redoc_url=None)
_gate = asyncio.Semaphore(MAX_CONCURRENT)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no network)
# --------------------------------------------------------------------------- #

class BadUrl(ValueError):
    pass


def validate_url(raw: str) -> str:
    """Accept only http(s) URLs to public hosts. The api proxies user input
    straight here, so this is the SSRF line: loopback, RFC1918, link-local and
    bare/dotless hostnames are refused before yt-dlp ever sees them."""
    url = (raw or "").strip()
    if not url:
        raise BadUrl("empty url")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise BadUrl("only http(s) URLs are supported")
    host = (parsed.hostname or "").lower()
    if not host:
        raise BadUrl("url has no host")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise BadUrl("local hosts are not allowed")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise BadUrl("private addresses are not allowed")
    elif "." not in host:
        raise BadUrl("hostname must be fully qualified")
    return url


def format_selector(max_height: int = MAX_HEIGHT) -> str:
    """Prefer H.264 + AAC so the result is a plain mp4 no browser or ffmpeg
    pipeline has to transcode; fall back to whatever exists."""
    h = int(max_height)
    return "/".join([
        f"bv*[vcodec^=avc1][height<={h}]+ba[acodec^=mp4a]",
        f"bv*[vcodec^=avc1][height<={h}]+ba",
        f"b[ext=mp4][height<={h}]",
        f"bv*[height<={h}]+ba",
        "b",
    ])


def section_arg(start: Optional[float], end: Optional[float]) -> Optional[str]:
    """yt-dlp --download-sections value, or None for the whole video."""
    if start is None and end is None:
        return None
    s = max(0.0, float(start or 0.0))
    e = "inf" if end is None else f"{float(end):.3f}"
    return f"*{s:.3f}-{e}"


def check_range(start: Optional[float], end: Optional[float], duration: Optional[float]) -> None:
    """Raise ValueError when the requested window is invalid or too long."""
    if start is not None and start < 0:
        raise ValueError("start must be >= 0")
    if end is not None and start is not None and end <= start:
        raise ValueError("end must be after start")
    if duration is not None and start is not None and start >= duration:
        raise ValueError(f"start is past the end of the video ({duration:.1f}s)")
    span_end = end if end is not None else duration
    if span_end is None:
        return  # live/unknown-length: yt-dlp decides, FETCH_TIMEOUT caps it
    span = span_end - (start or 0.0)
    if span > MAX_CLIP_SEC:
        raise ValueError(
            f"clip would be {span:.0f}s — the limit is {MAX_CLIP_SEC:.0f}s; give a start/end window"
        )


def build_clip_cmd(url: str, out_dir: str, start: Optional[float], end: Optional[float],
                   max_height: int = MAX_HEIGHT) -> list[str]:
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--no-progress",
        "--no-cache-dir",
        "--socket-timeout", "30",
        "--retries", "3",
        "-f", format_selector(max_height),
        "--merge-output-format", "mp4",
        "--remux-video", "mp4",
        "-o", os.path.join(out_dir, "clip.%(ext)s"),
    ]
    sec = section_arg(start, end)
    if sec:
        # force-keyframes re-encodes the window so the cut lands exactly on the
        # requested times — motion references need the timing, not the bitrate.
        cmd += ["--download-sections", sec, "--force-keyframes-at-cuts"]
    cmd.append(url)
    return cmd


def build_info_cmd(url: str) -> list[str]:
    return ["yt-dlp", "--no-playlist", "--no-warnings", "--no-cache-dir",
            "--socket-timeout", "30", "-J", url]


def parse_info(raw: dict[str, Any]) -> dict[str, Any]:
    """Trim yt-dlp's -J dump to what the UI shows."""
    if raw.get("_type") == "playlist":
        entries = raw.get("entries") or []
        raw = entries[0] if entries else raw
    return {
        "title": raw.get("title") or "",
        "durationSec": raw.get("duration"),
        "uploader": raw.get("uploader") or raw.get("channel") or "",
        "thumbnail": raw.get("thumbnail") or "",
        "extractor": raw.get("extractor_key") or raw.get("extractor") or "",
        "webpageUrl": raw.get("webpage_url") or raw.get("original_url") or "",
        "width": raw.get("width"),
        "height": raw.get("height"),
        "isLive": bool(raw.get("is_live")),
        "thumbnailDataUrl": None,
    }


THUMB_MAX_BYTES = 2 * 1024 * 1024


def thumbnail_data_url(url: str, timeout: float = 15.0) -> Optional[str]:
    """Fetch a (public) thumbnail and return it as a data: URL. The web app's
    CSP allows img-src data: but not arbitrary CDNs, and this container is the
    one place outside content is allowed to enter — so the picture rides along
    inline instead of the browser loading it. None on any failure or oddity."""
    try:
        clean = validate_url(url)
    except BadUrl:
        return None
    try:
        req = urllib.request.Request(clean, headers={"User-Agent": "vidmakr-clipgrab/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — validated public http(s)
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if not ctype.startswith("image/"):
                return None
            data = resp.read(THUMB_MAX_BYTES + 1)
    except Exception:  # noqa: BLE001
        return None
    if not data or len(data) > THUMB_MAX_BYTES:
        return None
    return f"data:{ctype};base64,{base64.b64encode(data).decode('ascii')}"


def find_output(out_dir: str) -> Optional[Path]:
    """yt-dlp names the merged file clip.mp4; a remux fallback may leave
    another extension. Prefer mp4, else the single largest media file."""
    p = Path(out_dir)
    mp4 = p / "clip.mp4"
    if mp4.exists():
        return mp4
    cands = [f for f in p.iterdir() if f.is_file() and not f.name.endswith((".part", ".ytdl", ".json"))]
    if not cands:
        return None
    return max(cands, key=lambda f: f.stat().st_size)


# --------------------------------------------------------------------------- #
# Subprocess plumbing
# --------------------------------------------------------------------------- #

async def _run(cmd: list[str], timeout: float) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise HTTPException(status_code=504, detail=f"timed out after {timeout:.0f}s")
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _ytdlp_error(stderr: str) -> str:
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    for l in reversed(lines):
        if l.startswith("ERROR:"):
            return l[len("ERROR:"):].strip()
    return lines[-1] if lines else "yt-dlp failed"


def _version(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "unavailable"


async def fetch_info(url: str, *, with_thumbnail: bool = False) -> dict[str, Any]:
    rc, out, err = await _run(build_info_cmd(url), INFO_TIMEOUT_SEC)
    if rc != 0:
        raise HTTPException(status_code=422, detail=_ytdlp_error(err))
    try:
        meta = parse_info(json.loads(out))
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="yt-dlp returned unreadable metadata")
    if with_thumbnail and meta.get("thumbnail"):
        meta["thumbnailDataUrl"] = await asyncio.to_thread(thumbnail_data_url, meta["thumbnail"])
    return meta


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "vidmakr-clipgrab",
        "yt_dlp": _version(["yt-dlp", "--version"]),
        "js_runtime": _version(["deno", "--version"]),
    }


@app.get("/info")
async def info(url: str = Query(...)) -> dict[str, Any]:
    try:
        clean = validate_url(url)
    except BadUrl as e:
        raise HTTPException(status_code=400, detail=str(e))
    return await fetch_info(clean, with_thumbnail=True)


class ClipRequest(BaseModel):
    url: str
    start: Optional[float] = Field(default=None, ge=0)
    end: Optional[float] = Field(default=None, ge=0)


@app.post("/clip")
async def clip(req: ClipRequest) -> FileResponse:
    try:
        url = validate_url(req.url)
    except BadUrl as e:
        raise HTTPException(status_code=400, detail=str(e))

    meta = await fetch_info(url)
    if meta.get("isLive"):
        raise HTTPException(status_code=422, detail="live streams are not supported")
    try:
        check_range(req.start, req.end, meta.get("durationSec"))
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))

    os.makedirs(SCRATCH_DIR, exist_ok=True)
    out_dir = tempfile.mkdtemp(prefix="clip-", dir=SCRATCH_DIR)
    try:
        async with _gate:
            rc, _out, err = await _run(build_clip_cmd(url, out_dir, req.start, req.end), FETCH_TIMEOUT_SEC)
        if rc != 0:
            raise HTTPException(status_code=422, detail=_ytdlp_error(err))
        path = find_output(out_dir)
        if path is None:
            raise HTTPException(status_code=502, detail="download produced no file")
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    title = meta.get("title") or "clip"
    headers = {
        # Percent-encoded so any unicode title survives an ASCII header.
        "X-Clipgrab-Title": urllib.parse.quote(title, safe=""),
        "X-Clipgrab-Source": urllib.parse.quote(meta.get("webpageUrl") or url, safe=":/?=&%"),
        "X-Clipgrab-Duration": str(meta.get("durationSec") or ""),
        "Cache-Control": "no-store",
    }
    return FileResponse(
        str(path),
        media_type="video/mp4",
        filename="clip.mp4",
        headers=headers,
        background=BackgroundTask(shutil.rmtree, out_dir, ignore_errors=True),
    )


@app.post("/update")
async def update() -> dict[str, Any]:
    before = _version(["yt-dlp", "--version"])
    rc, out, err = await _run(["pip", "install", "-q", "--upgrade", "yt-dlp"], 300)
    if rc != 0:
        raise HTTPException(status_code=502, detail=(err or out).strip()[-500:] or "pip failed")
    return {"ok": True, "before": before, "after": _version(["yt-dlp", "--version"])}
