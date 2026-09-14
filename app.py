"""vidmakr-music — YuE2 song generation as a small HTTP worker.

style + lyrics -> one stereo song (48 kHz FLAC master, mp3/m4a on request).
Only vidmakr-ai talks to this service. GPU-only, one song at a time, the
model loads lazily on the first job and is dropped after an idle period so
the card's VRAM goes back to the LLM / ComfyUI instance sharing it. Outputs
land in a per-job scratch dir (tmpfs) that vidmakr-ai downloads and then
deletes — nothing touches the host disk.

Routes
  GET    /healthz             → {ok, loaded, busy, queue, gpu, model, vae, runtime}
  POST   /generate            → {style, lyrics, cot?, seed?, cfg_scale?, id?} ⇒ {job_id, state, position}
  GET    /jobs                → in-memory job list (state only)
  GET    /jobs/{id}           → {state, stage, progress, tokens, seconds, audio_seconds, truncated, error, ...}
  GET    /jobs/{id}/audio     → audio bytes (?format=flac|mp3|m4a; flac is the master)
  GET    /jobs/{id}/score     → the ABC plan as text/plain (cot=full|melody only)
  DELETE /jobs/{id}           → queued: dropped at once; running: cancelled (status stays
                                readable until retention); done: scratch dir removed
  POST   /unload              → free the model now (debug / make room on the card)

Progress is stage-based: the pipeline reports each generated token for the
plan and semantic stages (counted against their token caps), while synthesis
and decoding are timed against the previous run's durations.
"""

from __future__ import annotations

import logging
import os
import re
import queue
import random
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

log = logging.getLogger("vidmakr-music")
if not log.handlers:  # uvicorn's dictConfig leaves the root logger handler-less
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch")
MODEL_ID = os.environ.get("MUSIC_MODEL", "m-a-p/YuE2-3B")
VAE_ID = os.environ.get("MUSIC_VAE", "m-a-p/YuE2-Vae")
LOCAL_FILES_ONLY = os.environ.get("MUSIC_LOCAL_FILES_ONLY", "1") == "1"
BUDGET_GIB = float(os.environ.get("MUSIC_BUDGET_GIB", "24"))
BACKEND = os.environ.get("MUSIC_BACKEND", "torch")            # torch | torch-eager | vllm
QUANTIZATION = os.environ.get("MUSIC_QUANTIZATION", "none")   # none | fp8
VERIFY_HASHES = os.environ.get("MUSIC_VERIFY_HASHES", "first")  # first | always | never
IDLE_UNLOAD_MIN = float(os.environ.get("MUSIC_IDLE_UNLOAD_MIN", "10"))
JOB_RETAIN_MIN = float(os.environ.get("MUSIC_JOB_RETAIN_MIN", "30"))
MAX_QUEUE = int(os.environ.get("MUSIC_MAX_QUEUE", "8"))
MAX_STYLE_CHARS = int(os.environ.get("MUSIC_MAX_STYLE_CHARS", "600"))
MAX_LYRICS_CHARS = int(os.environ.get("MUSIC_MAX_LYRICS_CHARS", "6000"))
SWEEP_SEC = float(os.environ.get("MUSIC_SWEEP_SEC", "30"))

# Token caps from yue2.protocol.GenerationConfig: the plan (ABC) stage samples
# at most 4096 tokens, the semantic stage at most 9000. Semantic tokens run at
# the codec frame rate — measured 25.0 tokens per second of audio on both spike
# songs (1484/59.4 s, 4997/199.9 s) — so the planned score's length predicts
# the semantic stage's token count, and 9000 tokens is a hard ~6 min ceiling.
PLAN_MAX_TOKENS = 4096
SEMANTIC_MAX_TOKENS = 9000
SEMANTIC_TOKENS_PER_SEC = 25.0
TOKENS_PER_LYRIC_LINE = 140  # fallback when there is no score (cot=off)
COT_MODES = ("full", "melody", "off")
AUDIO_FORMATS = {"flac": "audio/flac", "mp3": "audio/mpeg", "m4a": "audio/mp4"}
# Long songs (2026-09-13, ported from ComfyUI-YuE2-LongSong): the lyrics are
# split on [Section] blocks into chunks of at most LONG_MAX_LINES lyric lines
# (~3 min each), every chunk renders as its own song with the same style and
# seed+k (chunks after the first also name the first plan's key and BPM), then
# hard edges are trimmed and the chunks are joined with an equal-power
# crossfade. The pack's example: 2.5 s off the end of chunk 0, 1.5 s off the
# start of chunk 1, 4 s crossfade.
LONG_MAX_LINES = int(os.environ.get("MUSIC_LONG_MAX_LINES", "24"))
LONG_MAX_CHUNKS = int(os.environ.get("MUSIC_LONG_MAX_CHUNKS", "4"))
LONG_CROSSFADE_MS = 4000.0
LONG_TRIM_START_S = 1.5
LONG_TRIM_END_S = 2.5
SAMPLE_RATE = 48000

# Stage -> (start, end) share of the overall progress bar.
STAGE_SPAN = {
    "queued": (0.0, 0.0),
    "loading": (0.0, 0.05),
    "plan": (0.05, 0.20),
    "semantic": (0.20, 0.70),
    "synthesize": (0.70, 0.92),
    "decode": (0.92, 0.99),
    "saving": (0.99, 1.0),
    "done": (1.0, 1.0),
}
# Learned wall-clock estimates for the timed stages (EMA over completed jobs).
_STAGE_SECONDS = {"loading": 40.0, "synthesize": 25.0, "decode": 10.0}

app = FastAPI(title="vidmakr-music", docs_url=None, redoc_url=None)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no torch)
# --------------------------------------------------------------------------- #

class BadRequest(ValueError):
    pass


def normalize_lyrics(raw: str) -> str:
    """Trim, normalise newlines, collapse runs of blank lines, and bracket bare
    section headers ("Verse 2:" -> "[Verse 2]") so the model sees the tags it
    was trained on. Lyrics without any section tag get a leading [Verse]."""
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line[:1] == "[" and line[-1:] == "]":
            line = "[" + line[1:-1].strip().title() + "]"
        elif line.endswith(":") and len(line) <= 24 and line[:-1].replace(" ", "").isalnum():
            line = "[" + line[:-1].strip().title() + "]"
        lines.append(line)
    out: list[str] = []
    for line in lines:
        if line == "" and (not out or out[-1] == ""):
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    text = "\n".join(out)
    if not text:
        raise BadRequest("lyrics are empty")
    if not any(l.startswith("[") and l.endswith("]") for l in out):
        text = "[Verse]\n" + text
    if len(text) > MAX_LYRICS_CHARS:
        raise BadRequest(f"lyrics exceed {MAX_LYRICS_CHARS} characters")
    return text


def normalize_style(raw: str) -> str:
    style = " ".join((raw or "").split())
    if not style:
        raise BadRequest("style is empty")
    if len(style) > MAX_STYLE_CHARS:
        raise BadRequest(f"style exceeds {MAX_STYLE_CHARS} characters")
    return style


MAX_ABC_CHARS = int(os.environ.get("MUSIC_MAX_ABC_CHARS", "60000"))


def normalize_abc(raw: Optional[str]) -> Optional[str]:
    """An external ABC score (cover mode, from SheetSage2 or the user): must
    carry a K: header and at least one music line; CRLF → LF; size-capped."""
    if raw is None:
        return None
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return None
    if len(text) > MAX_ABC_CHARS:
        raise BadRequest(f"abc too long (> {MAX_ABC_CHARS} chars)")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not any(ln.startswith("K:") for ln in lines):
        raise BadRequest("abc needs a K: (key) header")
    if not any(not re.match(r"^[A-Za-z]:|^%", ln) for ln in lines):
        raise BadRequest("abc has headers but no music lines")
    return text + "\n"


def build_request(style: str, lyrics: str, cot: Optional[str] = None, seed: Optional[int] = None,
                  cfg_scale: Optional[float] = None, song_id: Optional[str] = None,
                  abc: Optional[str] = None, long: Any = None) -> dict[str, Any]:
    """Validate and shape a request into yue2.protocol.SongRequest kwargs (+ a
    `long` options dict the worker strips before building the SongRequest).
    abc = an external score (cover mode): YuE2 tokenizes it instead of planning
    one, so cot must be melody or full. long = chunked rendering + stitching
    for songs past one plan's budget; not combinable with a cover score (its
    ABC would have to be sliced per chunk — v2)."""
    cot = (cot or "full").lower()
    if cot not in COT_MODES:
        raise BadRequest("cot must be full, melody or off")
    abc = normalize_abc(abc)
    if abc is not None and cot == "off":
        raise BadRequest("an external abc score needs cot=melody or full")
    long_opts = normalize_long(long)
    if abc is not None and long_opts is not None:
        raise BadRequest("a cover score cannot be combined with long mode")
    if seed is None:
        seed = random.randrange(0, 2**31)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise BadRequest("seed must be an integer in [0, 2**63)")
    if cfg_scale is not None and not 0 <= float(cfg_scale) <= 20:
        raise BadRequest("cfg_scale must be within [0, 20]")
    req: dict[str, Any] = {
        "style": normalize_style(style),
        "lyrics": normalize_lyrics(lyrics),
        "cot": cot,
        "seed": int(seed),
        "id": song_id or "song",
    }
    if cfg_scale is not None:
        req["cfg_scale"] = float(cfg_scale)
    if abc is not None:
        req["abc"] = abc
    if long_opts is not None:
        req["long"] = long_opts
    return req


# ---- long songs: lyric chunking + audio stitching ---------------------------

_SECTION_TAG_RE = re.compile(r"^\[[^\]]+\]$")


def split_sections(lyrics: str) -> list[str]:
    """Normalized lyrics → tagged section blocks (tag line + its lines). An
    untagged block after a tagged one belongs to that section."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", (lyrics or "").strip()) if b.strip()]
    out: list[str] = []
    for block in blocks:
        first = block.splitlines()[0].strip()
        if _SECTION_TAG_RE.match(first) or not out:
            out.append(block)
        else:
            out[-1] = out[-1] + "\n" + block
    return out


def section_lines(block: str) -> int:
    return sum(1 for ln in block.splitlines() if ln.strip() and not _SECTION_TAG_RE.match(ln.strip()))


def is_chorus(block: str) -> bool:
    tag = block.splitlines()[0].strip().lower() if block.strip() else ""
    return "chorus" in tag and "pre" not in tag


def plan_chunks(sections: list[str], max_lines: int, overlap_chorus: bool = False) -> list[str]:
    """Greedy: sections fill a chunk up to max_lines lyric lines, then a new
    chunk starts (a lone oversized section is its own chunk). overlap_chorus
    repeats the previous chunk's latest [Chorus] at the head of the new chunk
    (the LongSong pack's "shared chorus at the seam") when it fits."""
    chunks: list[list[str]] = [[]]
    count = 0
    for sec in sections:
        n = section_lines(sec)
        if chunks[-1] and count + n > max_lines:
            head: list[str] = []
            c = 0
            if overlap_chorus:
                chorus = next((s for s in reversed(chunks[-1]) if is_chorus(s)), None)
                if chorus is not None and section_lines(chorus) + n <= max_lines:
                    head, c = [chorus], section_lines(chorus)
            chunks.append(head)
            count = c
        chunks[-1].append(sec)
        count += n
    return ["\n\n".join(c) for c in chunks if c]


_KEY_RE = re.compile(r"^K:\s*([A-Ga-g][#b]?)\s*([A-Za-z]*)")
_Q_RE = re.compile(r"^Q:\s*(?:\d+\s*/\s*\d+\s*=\s*)?(\d+)")


def abc_key_bpm(abc: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """(key, bpm) from K:/Q: headers — "K:Dm" → "D minor", "Q:1/4=92" → 92."""
    key = bpm = None
    for line in (abc or "").splitlines():
        line = line.strip()
        if key is None and line.startswith("K:"):
            m = _KEY_RE.match(line)
            if m:
                tonic, mode = m.group(1), m.group(2).lower()
                mode_word = ("minor" if mode in ("m", "min", "minor") else
                             "major" if mode in ("", "maj", "major") else mode)
                key = f"{tonic[0].upper()}{tonic[1:]} {mode_word}"
        elif bpm is None and line.startswith("Q:"):
            m = _Q_RE.match(line)
            if m:
                bpm = int(m.group(1))
    return key, bpm


def style_with_key(style: str, abc: Optional[str]) -> str:
    """Later chunks name the first chunk's key and tempo so the halves agree."""
    key, bpm = abc_key_bpm(abc)
    out = style.rstrip(" ,.")
    low = out.lower()
    if key and " key" not in low and f" {key.lower()}" not in low:
        out += f", in the key of {key}"
    if bpm and "bpm" not in low:
        out += f", {bpm} BPM"
    return out


def equal_power_crossfade(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
    """Join a then b (samples × channels), blending `overlap` samples with
    cos/sin quarter-wave gains (constant power across the seam)."""
    overlap = int(max(0, min(overlap, len(a), len(b))))
    if overlap == 0:
        return np.concatenate([a, b])
    t = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    shape = (overlap,) + (1,) * (a.ndim - 1)
    fade_out, fade_in = np.cos(t * np.pi / 2).reshape(shape), np.sin(t * np.pi / 2).reshape(shape)
    mid = a[-overlap:] * fade_out + b[:overlap] * fade_in
    return np.concatenate([a[:-overlap], mid.astype(a.dtype, copy=False), b[overlap:]])


def trim_edges(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Drop the hard intro/outro seconds of a chunk; never below one second."""
    s = int(round(max(0.0, start_s) * sr))
    e = len(audio) - int(round(max(0.0, end_s) * sr))
    if e - s < sr:
        return audio
    return audio[s:e]


def stitch_chunks(chunks: list[np.ndarray], sr: int, crossfade_ms: float,
                  trim_start_s: float, trim_end_s: float) -> np.ndarray:
    """Trim inner edges (start of every chunk but the first, end of every
    chunk but the last) and crossfade-concatenate."""
    out: Optional[np.ndarray] = None
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk = trim_edges(chunk, sr, trim_start_s if i > 0 else 0.0, trim_end_s if i < n - 1 else 0.0)
        out = chunk if out is None else equal_power_crossfade(out, chunk, int(round(crossfade_ms / 1000.0 * sr)))
    return out if out is not None else np.zeros((0, 2), dtype=np.float32)


def normalize_long(long: Any) -> Optional[dict[str, Any]]:
    """Long-song options → validated dict, or None when off/absent."""
    if not long:
        return None
    if long is True:
        long = {}
    if not isinstance(long, dict):
        raise BadRequest("long must be true or an options object")
    if long.get("enabled") is False:
        return None
    out = {
        "max_lines_per_chunk": int(long.get("max_lines_per_chunk") or LONG_MAX_LINES),
        "overlap_chorus": bool(long.get("overlap_chorus", False)),
        "crossfade_ms": float(long.get("crossfade_ms", LONG_CROSSFADE_MS)),
        "trim_start_s": float(long.get("trim_start_s", LONG_TRIM_START_S)),
        "trim_end_s": float(long.get("trim_end_s", LONG_TRIM_END_S)),
    }
    if not 6 <= out["max_lines_per_chunk"] <= 60:
        raise BadRequest("max_lines_per_chunk must be within [6, 60]")
    if not 0 <= out["crossfade_ms"] <= 15000:
        raise BadRequest("crossfade_ms must be within [0, 15000]")
    if not 0 <= out["trim_start_s"] <= 30 or not 0 <= out["trim_end_s"] <= 30:
        raise BadRequest("trims must be within [0, 30] seconds")
    return out


_ABC_FRACTION = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def abc_seconds(abc: Optional[str]) -> Optional[float]:
    """Planned song length from a YuE2 ABC score: bars in the Vocal voice ×
    seconds per bar from the M: meter and Q: tempo headers. None when the
    headers are missing or unparsable."""
    if not abc:
        return None
    meter = tempo = None
    for line in abc.splitlines():
        if line.startswith("M:") and meter is None:
            m = _ABC_FRACTION.match(line[2:])
            if m:
                meter = int(m.group(1)) / int(m.group(2))
        elif line.startswith("Q:") and tempo is None:
            body = line[2:].strip()
            if "=" in body:
                unit, bpm = body.split("=", 1)
                m = _ABC_FRACTION.match(unit)
                try:
                    tempo = (int(m.group(1)) / int(m.group(2)) if m else 0.25, float(bpm))
                except ValueError:
                    tempo = None
            else:
                try:
                    tempo = (0.25, float(body))
                except ValueError:
                    tempo = None
    if not meter or not tempo or tempo[1] <= 0:
        return None
    unit, bpm = tempo
    bars = {}
    voice = None
    for line in abc.splitlines():
        if line.startswith("V:"):
            voice = line[2:].strip().split()[0] if line[2:].strip() else None
            continue
        if voice and not line.startswith(("%", "K:", "L:", "M:", "Q:", "T:", "X:")):
            bars[voice] = bars.get(voice, 0) + line.count("|")
    count = bars.get("Vocal") or (max(bars.values()) if bars else 0)
    if count <= 0:
        return None
    return round(count * (meter / unit) * 60.0 / bpm, 1)


def expected_semantic_tokens(abc: Optional[str], lyrics: str) -> int:
    """Token target for the semantic stage's progress bar (never the model's
    limit): the planned score length when there is one, else a per-line guess."""
    seconds = abc_seconds(abc)
    if seconds:
        guess = seconds * SEMANTIC_TOKENS_PER_SEC * 1.05
    else:
        lines = sum(1 for l in (lyrics or "").splitlines() if l.strip() and not l.strip().startswith("["))
        guess = max(lines, 4) * TOKENS_PER_LYRIC_LINE
    return int(min(max(guess, 400), SEMANTIC_MAX_TOKENS))


def stage_progress(stage: str, frac: float) -> float:
    """Overall 0..1 progress for being `frac` of the way through `stage`."""
    lo, hi = STAGE_SPAN.get(stage, (0.0, 0.0))
    frac = min(max(float(frac), 0.0), 1.0)
    return round(lo + (hi - lo) * frac, 4)


def timed_fraction(elapsed: float, expected: float) -> float:
    """Progress guess for a stage without callbacks: asymptotic so it never
    claims done before the stage returns."""
    if expected <= 0:
        return 0.0
    x = elapsed / expected
    return min(0.95, x if x < 0.8 else 0.8 + (x - 0.8) * 0.25)


def learn_seconds(stage: str, seconds: float, alpha: float = 0.5) -> float:
    prev = _STAGE_SECONDS.get(stage, seconds)
    _STAGE_SECONDS[stage] = prev * (1 - alpha) + seconds * alpha
    return _STAGE_SECONDS[stage]


def ffmpeg_cmd(src: Path, dst: Path, fmt: str) -> list[str]:
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if fmt == "mp3":
        return base + ["-codec:a", "libmp3lame", "-q:a", "2", str(dst)]
    if fmt == "m4a":
        return base + ["-codec:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)]
    raise BadRequest("format must be flac, mp3 or m4a")


# --------------------------------------------------------------------------- #
# Job store + worker
# --------------------------------------------------------------------------- #

@dataclass
class Job:
    id: str
    request: dict[str, Any]
    dir: Path
    state: str = "queued"          # queued | running | done | failed | cancelled
    stage: str = "queued"
    progress: float = 0.0
    tokens: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    audio_seconds: Optional[float] = None
    planned_seconds: Optional[float] = None
    truncated: dict[str, bool] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    abc: Optional[str] = None
    cancel: bool = False
    last_access: float = field(default_factory=time.time)
    # Long mode: the slice of the progress bar the current chunk owns, and
    # per-chunk facts once rendered.
    chunk_span: tuple[float, float] = (0.0, 1.0)
    chunks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id, "state": self.state, "stage": self.stage,
            "progress": self.progress, "tokens": self.tokens, "error": self.error,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
            "seconds": (None if self.started_at is None else
                        round((self.finished_at or time.time()) - self.started_at, 1)),
            "audio_seconds": self.audio_seconds, "planned_seconds": self.planned_seconds,
            "truncated": self.truncated,
            "timing": self.timing, "has_score": self.abc is not None,
            "request": {k: v for k, v in self.request.items() if k not in ("lyrics", "abc")},
            "cover": self.request.get("abc") is not None,
            "long": self.request.get("long") is not None, "chunks": self.chunks,
        }


class Runner:
    """Owns the pipeline and the single worker thread. `factory` builds the
    pipeline (swapped for a fake in tests); `pipe` is None while unloaded."""

    def __init__(self, factory: Optional[Callable[[], Any]] = None):
        self.factory = factory or self._load_pipeline
        self.pipe: Any = None
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.q: "queue.Queue[str]" = queue.Queue()
        self.current: Optional[str] = None
        self.last_activity = time.time()
        self.verified = False
        self._thread: Optional[threading.Thread] = None
        self._sweeper: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._work, name="music-worker", daemon=True)
            self._thread.start()
        if self._sweeper is None:
            self._sweeper = threading.Thread(target=self._sweep, name="music-sweeper", daemon=True)
            self._sweeper.start()

    def stop(self) -> None:
        self._stop.set()

    def _load_pipeline(self) -> Any:
        from yue2 import YuE2Pipeline  # heavy import, only on first job
        verify = VERIFY_HASHES == "always" or (VERIFY_HASHES == "first" and not self.verified)
        pipe = YuE2Pipeline.from_pretrained(
            MODEL_ID, vae=VAE_ID, local_files_only=LOCAL_FILES_ONLY, progress=False,
            memory_budget_gib=BUDGET_GIB, backend=BACKEND, quantization=QUANTIZATION,
            verify_hashes=verify,
        )
        self.verified = True
        return pipe

    def ensure_loaded(self, job: Optional[Job] = None) -> Any:
        if self.pipe is not None:
            return self.pipe
        t0 = time.time()
        if job is not None:
            self._set(job, stage="loading", progress=stage_progress("loading", 0))
        log.info("loading %s + %s (backend=%s quant=%s budget=%sGiB)", MODEL_ID, VAE_ID, BACKEND, QUANTIZATION, BUDGET_GIB)
        self.pipe = self.factory()
        learn_seconds("loading", time.time() - t0)
        log.info("model loaded in %.1fs", time.time() - t0)
        return self.pipe

    def unload(self) -> bool:
        with self.lock:
            if self.pipe is None or self.current is not None:
                return False
            pipe, self.pipe = self.pipe, None
        try:
            pipe.close()
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("close failed: %s", exc)
        log.info("model unloaded (idle)")
        return True

    # ---- jobs --------------------------------------------------------------
    def submit(self, request: dict[str, Any]) -> tuple[Job, int]:
        with self.lock:
            pending = sum(1 for j in self.jobs.values() if j.state == "queued")
            if pending >= MAX_QUEUE:
                raise BadRequest(f"queue is full ({MAX_QUEUE} pending)")
            job_id = uuid.uuid4().hex[:12]
            job = Job(id=job_id, request=request, dir=Path(SCRATCH_DIR) / job_id)
            # jobs ahead of this one: the running song plus everything queued
            position = pending + (1 if self.current is not None else 0)
            self.jobs[job_id] = job
            self.last_activity = time.time()
        self.q.put(job_id)
        return job, position

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.last_access = time.time()
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self.lock:
            job.cancel = True
            if job.state == "queued":
                job.state, job.stage, job.finished_at = "cancelled", "done", time.time()
        return job

    def delete(self, job_id: str) -> None:
        job = self.cancel(job_id)
        if job.state in ("queued", "running"):
            return  # the worker drops the dir once it notices the cancel
        self._drop(job)

    def _drop(self, job: Job) -> None:
        with self.lock:
            self.jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    def _set(self, job: Job, **fields: Any) -> None:
        with self.lock:
            for k, v in fields.items():
                setattr(job, k, v)

    # ---- worker ------------------------------------------------------------
    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            job = self.jobs.get(job_id)
            if job is None or job.state != "queued":
                continue
            with self.lock:
                self.current = job.id
                job.state, job.started_at = "running", time.time()
                self.last_activity = job.started_at
            try:
                self._run(job)
                self._set(job, state="done", stage="done", progress=1.0, finished_at=time.time())
                log.info("job %s done: %.1fs audio in %.1fs", job.id, job.audio_seconds or 0,
                         job.finished_at - job.started_at)
            except InterruptedError:
                self._set(job, state="cancelled", stage="done", finished_at=time.time())
                shutil.rmtree(job.dir, ignore_errors=True)
                log.info("job %s cancelled", job.id)
            except Exception as exc:
                log.exception("job %s failed", job.id)
                self._set(job, state="failed", stage="done", error=f"{type(exc).__name__}: {exc}",
                          finished_at=time.time())
                if job.cancel:
                    self._drop(job)
            finally:
                with self.lock:
                    self.current = None
                    self.last_activity = time.time()

    def _prog(self, job: Job, stage: str, frac: float) -> float:
        """Overall progress: the stage's share, scaled into the current chunk's span."""
        lo, hi = job.chunk_span
        return round(lo + (hi - lo) * stage_progress(stage, frac), 4)

    def _run(self, job: Job) -> None:
        from yue2.protocol import SongRequest
        pipe = self.ensure_loaded(job)
        base = {k: v for k, v in job.request.items() if k != "long"}
        long = job.request.get("long")
        lyric_chunks = (plan_chunks(split_sections(base["lyrics"]), long["max_lines_per_chunk"], long["overlap_chorus"])
                        if long else [base["lyrics"]])
        if len(lyric_chunks) > LONG_MAX_CHUNKS:
            raise RuntimeError(f"long song needs {len(lyric_chunks)} chunks (max {LONG_MAX_CHUNKS}) — shorten the lyrics")
        n = len(lyric_chunks)
        audios: list[np.ndarray] = []
        abcs: list[Optional[str]] = []
        truncated = {"abc": False, "semantic": False}
        for i, lyrics in enumerate(lyric_chunks):
            req_d = dict(base, lyrics=lyrics)
            if n > 1:
                req_d["seed"] = int(base["seed"]) + i
                req_d["id"] = f"{base['id']}_{i + 1}"
                if i > 0 and abcs and abcs[0]:
                    req_d["style"] = style_with_key(base["style"], abcs[0])
            job.chunk_span = (i / n, (i + 1) / n)
            audio, abc, trunc = self._render(job, pipe, SongRequest(**req_d))
            audios.append(audio)
            abcs.append(abc)
            truncated = {k: truncated[k] or trunc[k] for k in truncated}
            job.chunks.append({"index": i + 1, "lines": sum(section_lines(sec) for sec in split_sections(lyrics)),
                               "seed": req_d.get("seed"), "seconds": round(len(audio) / SAMPLE_RATE, 2),
                               "planned_seconds": abc_seconds(abc), "style": req_d["style"]})
            if job.cancel:
                raise InterruptedError("cancelled between chunks")
        job.chunk_span = (0.0, 1.0)
        if n > 1:
            audio = stitch_chunks(audios, SAMPLE_RATE, long["crossfade_ms"], long["trim_start_s"], long["trim_end_s"])
            abc = "\n".join(f"% chunk {i + 1}\n{a.rstrip()}" for i, a in enumerate(abcs) if a) or None
            planned = sum(c["planned_seconds"] or 0 for c in job.chunks) or None
        else:
            audio, abc = audios[0], abcs[0]
            planned = abc_seconds(abc)
        self._set(job, stage="saving", progress=stage_progress("saving", 0), abc=abc, planned_seconds=planned)
        job.dir.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sf.write(job.dir / "audio.flac", audio, SAMPLE_RATE, subtype="PCM_24")
        if abc:
            (job.dir / "score.abc").write_text(abc, encoding="utf-8")
        self._set(job, audio_seconds=round(len(audio) / SAMPLE_RATE, 2), truncated=truncated)

    def _render(self, job: Job, pipe: Any, request: Any) -> tuple[np.ndarray, Optional[str], dict[str, bool]]:
        """One YuE2 song (plan → semantic → synthesize → decode) for one
        request; progress lands inside job.chunk_span."""
        cancelled = lambda: job.cancel  # noqa: E731

        def on_token(cap: int, stage: str):
            def cb(_phase: str, _token: int) -> None:
                with self.lock:
                    job.tokens += 1
                    job.progress = self._prog(job, stage, min(job.tokens / cap, 0.98))
            return cb

        t = time.time()
        self._set(job, stage="plan", tokens=0, progress=self._prog(job, "plan", 0))
        plan = pipe.plan(request=request, cancelled=cancelled,
                         on_token=on_token(PLAN_MAX_TOKENS, "plan") if request.cot != "off" else None)
        if cancelled():
            raise InterruptedError("cancelled after plan")
        job.timing["plan_seconds"] = round(job.timing.get("plan_seconds", 0.0) + time.time() - t, 2)

        t = time.time()
        abc = getattr(plan, "abc", None)
        target = expected_semantic_tokens(abc, request.lyrics)
        self._set(job, stage="semantic", tokens=0, progress=self._prog(job, "semantic", 0),
                  abc=abc, planned_seconds=abc_seconds(abc))
        job.timing["semantic_target_tokens"] = target
        semantic = pipe.generate_semantic(plan, cancelled=cancelled,
                                          on_token=on_token(target, "semantic"))
        job.timing["semantic_seconds"] = round(job.timing.get("semantic_seconds", 0.0) + time.time() - t, 2)
        job.timing["semantic_tokens"] = job.timing.get("semantic_tokens", 0) + len(getattr(semantic, "tokens", []) or [])

        t = time.time()
        self._set(job, stage="synthesize", progress=self._prog(job, "synthesize", 0))
        ticker = self._ticker(job, "synthesize", t)
        try:
            latents = pipe.synthesize(semantic, cancelled=cancelled)
        finally:
            ticker.set()
        learn_seconds("synthesize", time.time() - t)
        job.timing["synthesize_seconds"] = round(job.timing.get("synthesize_seconds", 0.0) + time.time() - t, 2)
        if cancelled():
            raise InterruptedError("cancelled before decode")

        t = time.time()
        self._set(job, stage="decode", progress=self._prog(job, "decode", 0))
        ticker = self._ticker(job, "decode", t)
        try:
            audio = pipe.decode(latents)
        finally:
            ticker.set()
        learn_seconds("decode", time.time() - t)
        job.timing["decode_seconds"] = round(job.timing.get("decode_seconds", 0.0) + time.time() - t, 2)
        trunc = {"abc": bool(getattr(plan, "truncated", False)), "semantic": bool(getattr(semantic, "truncated", False))}
        return np.asarray(audio), abc, trunc

    def _ticker(self, job: Job, stage: str, t0: float) -> threading.Event:
        """Advance a callback-less stage's progress on a timer until `stop` is set."""
        stop = threading.Event()

        def tick() -> None:
            while not stop.wait(1.0):
                frac = timed_fraction(time.time() - t0, _STAGE_SECONDS.get(stage, 20.0))
                self._set(job, progress=self._prog(job, stage, frac))

        threading.Thread(target=tick, daemon=True).start()
        return stop

    # ---- sweeper -----------------------------------------------------------
    def _sweep(self) -> None:
        while not self._stop.wait(SWEEP_SEC):
            self.sweep_once()

    def sweep_once(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        stale = [j for j in list(self.jobs.values())
                 if j.state in ("done", "failed", "cancelled") and now - j.last_access > JOB_RETAIN_MIN * 60]
        for job in stale:
            log.info("job %s expired", job.id)
            self._drop(job)
        idle = self.current is None and self.q.empty() and now - self.last_activity > IDLE_UNLOAD_MIN * 60
        if idle and self.pipe is not None:
            self.unload()


runner = Runner()


@app.on_event("startup")
async def _startup() -> None:
    Path(SCRATCH_DIR).mkdir(parents=True, exist_ok=True)
    runner.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    runner.stop()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

class GenerateRequest(BaseModel):
    style: str = Field(..., description="genre, instruments, vocal character, language, tempo")
    lyrics: str = Field(..., description="[Verse]/[Chorus]-tagged lyric lines")
    cot: Optional[str] = Field(None, description="full | melody | off")
    seed: Optional[int] = None
    cfg_scale: Optional[float] = None
    id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$")
    # Cover mode: an external ABC score (SheetSage2 melody-only transcription
    # of the source song, or hand-edited). YuE2 sings the lyrics to it; the
    # style line steers the accompaniment. Requires cot=melody|full.
    abc: Optional[str] = None
    # Long mode: true, or {max_lines_per_chunk, overlap_chorus, crossfade_ms,
    # trim_start_s, trim_end_s}. Lyrics are chunked on [Section] blocks and
    # the chunk songs are crossfaded together (see LONG_* constants).
    long: Optional[Any] = None


def _gpu_info() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        i = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(i)
        return {"available": True, "name": torch.cuda.get_device_name(i),
                "free_gib": round(free / 2**30, 1), "total_gib": round(total / 2**30, 1)}
    except Exception as exc:  # torch missing or driver trouble: report, don't fail health
        return {"available": False, "error": str(exc)[:200]}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    try:
        from yue2 import __version__ as runtime
    except Exception:
        runtime = None
    return {
        "ok": True, "loaded": runner.pipe is not None, "busy": runner.current,
        "queue": sum(1 for j in runner.jobs.values() if j.state == "queued"),
        "model": MODEL_ID, "vae": VAE_ID, "backend": BACKEND, "quantization": QUANTIZATION,
        "runtime": runtime, "gpu": _gpu_info(), "estimates": dict(_STAGE_SECONDS),
    }


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, Any]:
    try:
        request = build_request(req.style, req.lyrics, req.cot, req.seed, req.cfg_scale, req.id,
                                abc=req.abc, long=req.long)
        job, position = runner.submit(request)
    except BadRequest as exc:
        raise HTTPException(400, str(exc))
    return {"job_id": job.id, "state": job.state, "position": position, "seed": request["seed"]}


@app.get("/jobs")
async def jobs() -> dict[str, Any]:
    return {"busy": runner.current, "loaded": runner.pipe is not None,
            "jobs": [j.to_dict() for j in runner.jobs.values()]}


def _job(job_id: str) -> Job:
    try:
        return runner.get(job_id)
    except KeyError:
        raise HTTPException(404, "no such job")


@app.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    return _job(job_id).to_dict()


@app.get("/jobs/{job_id}/audio")
async def job_audio(job_id: str, format: str = Query("flac")) -> FileResponse:
    job = _job(job_id)
    if job.state != "done":
        raise HTTPException(409, f"job is {job.state}")
    fmt = format.lower()
    if fmt not in AUDIO_FORMATS:
        raise HTTPException(400, "format must be flac, mp3 or m4a")
    master = job.dir / "audio.flac"
    if not master.is_file():
        raise HTTPException(410, "audio already released")
    out = master if fmt == "flac" else job.dir / f"audio.{fmt}"
    if not out.is_file():
        proc = subprocess.run(ffmpeg_cmd(master, out, fmt), capture_output=True, text=True)
        if proc.returncode != 0:
            raise HTTPException(500, f"ffmpeg failed: {proc.stderr[-300:]}")
    name = f"{job.request.get('id', 'song')}-{job.id}.{fmt}"
    return FileResponse(out, media_type=AUDIO_FORMATS[fmt], filename=name,
                        headers={"Cache-Control": "no-store"})


@app.get("/jobs/{job_id}/score")
async def job_score(job_id: str) -> PlainTextResponse:
    job = _job(job_id)
    if job.abc is None:
        raise HTTPException(404, "no score for this job (cot=off or not planned yet)")
    return PlainTextResponse(job.abc, headers={"Cache-Control": "no-store"})


@app.delete("/jobs/{job_id}")
async def job_delete(job_id: str) -> dict[str, Any]:
    _job(job_id)
    runner.delete(job_id)
    return {"ok": True}


@app.post("/unload")
async def unload() -> dict[str, Any]:
    return {"unloaded": runner.unload(), "loaded": runner.pipe is not None}
