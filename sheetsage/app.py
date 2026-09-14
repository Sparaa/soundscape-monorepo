"""vidmakr-sheetsage — SheetSage2 transcription sidecar.

Song audio in (base64, any ffmpeg-readable format) → melody-only ABC score out,
for YuE2 cover mode ("sing OUR lyrics to THAT melody", or the same melody in a
new style). One transcription at a time on the GPU, lazy model load with idle
unload, inputs in a tmpfs scratch that is deleted with the job — no host disk.
Same shape as the music sidecar (music/app.py) so the ai side drives both the
same way. Pins: python 3.11, torch 2.8.0, transformers 4.45.2 (SheetSage2's
own requirements.txt) — which is why this is not inside vidmakr-music.
"""
from __future__ import annotations

import base64
import binascii
import gc
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

log = logging.getLogger("sheetsage")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL = os.environ.get("SHEETSAGE_MODEL", "m-a-p/SheetSage2")
LOCAL_FILES_ONLY = os.environ.get("SHEETSAGE_LOCAL_FILES_ONLY", "1") == "1"
DTYPE = os.environ.get("SHEETSAGE_DTYPE", "bf16")            # bf16 | fp32
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch")
IDLE_UNLOAD_MIN = float(os.environ.get("SHEETSAGE_IDLE_UNLOAD_MIN", "10"))
JOB_RETAIN_MIN = float(os.environ.get("SHEETSAGE_JOB_RETAIN_MIN", "30"))
MAX_QUEUE = int(os.environ.get("SHEETSAGE_MAX_QUEUE", "8"))
MAX_AUDIO_MB = float(os.environ.get("SHEETSAGE_MAX_AUDIO_MB", "80"))
# Hard cap on the audio analysed (SheetSage2 slides 300 s windows over any
# length; a 15-minute upload is a mistake, not a song).
MAX_SECONDS = float(os.environ.get("SHEETSAGE_MAX_SECONDS", "900"))
SWEEP_SEC = float(os.environ.get("SHEETSAGE_SWEEP_SEC", "30"))

# Stage -> (start, end) share of the progress bar. "encoding" advances per
# 300 s window; decoding/ABC assembly is short and untimed.
STAGE_SPAN = {
    "queued": (0.0, 0.0), "loading": (0.0, 0.10), "decoding": (0.10, 0.15),
    "encoding": (0.15, 0.90), "scoring": (0.90, 0.99), "done": (1.0, 1.0),
}

app = FastAPI(title="vidmakr-sheetsage", docs_url=None, redoc_url=None)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no torch)
# --------------------------------------------------------------------------- #

class BadRequest(ValueError):
    pass


def decode_audio_b64(b64: str) -> bytes:
    """Base64 → bytes with a size cap (before decoding: 4/3 overhead)."""
    if not b64 or not b64.strip():
        raise BadRequest("audio_b64 is required")
    if len(b64) > MAX_AUDIO_MB * 1024 * 1024 * 4 / 3:
        raise BadRequest(f"audio larger than {MAX_AUDIO_MB:g} MB")
    try:
        data = base64.b64decode(b64, validate=False)
    except (ValueError, binascii.Error) as exc:
        raise BadRequest(f"audio_b64 is not valid base64: {exc}") from exc
    if len(data) < 64:
        raise BadRequest("audio is empty")
    return data


_KEY_RE = re.compile(r"^K:\s*([A-Ga-g][#b]?)\s*([A-Za-z]*)")
_Q_RE = re.compile(r"^Q:\s*(?:\d+\s*/\s*\d+\s*=\s*)?(\d+)")


def abc_key_bpm(abc: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """(key, bpm) from an ABC score's K: and Q: headers — e.g. "K:Dm" → "D minor",
    "Q:1/4=120" → 120. None when a header is missing or unreadable."""
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


_HEADER_RE = re.compile(r"^[A-Za-z]:")


def abc_summary(abc: Optional[str]) -> dict[str, Any]:
    """Shape facts the lyric writer / UI want: bar count (barlines in music
    lines), voice count (V: headers) and the section labels YuE2/SheetSage2
    leave as % comments or P: parts, plus key/bpm."""
    bars = 0
    voices: set[str] = set()
    sections: list[str] = []
    for line in (abc or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("%"):
            label = s.lstrip("%").strip()
            if label and re.search(r"\b(intro|verse|pre-?chorus|chorus|bridge|outro|solo|section)\b", label, re.I):
                sections.append(label)
            continue
        if s.startswith("V:"):
            voices.add(s[2:].strip().split()[0] if s[2:].strip() else "1")
            continue
        if s.startswith("P:"):
            sections.append(s[2:].strip())
            continue
        if _HEADER_RE.match(s):
            continue
        bars += s.count("|")
    key, bpm = abc_key_bpm(abc)
    return {"bars": bars, "voices": max(1, len(voices)), "sections": sections, "key": key, "bpm": bpm}


def stage_progress(stage: str, frac: float) -> float:
    lo, hi = STAGE_SPAN.get(stage, (0.0, 1.0))
    return round(lo + (hi - lo) * max(0.0, min(1.0, frac)), 4)


def probe_seconds(path: Path) -> Optional[float]:
    """Input duration via ffprobe; None when unavailable (tests, odd files)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60, check=False,
        ).stdout.strip()
        return round(float(out), 2) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- #
# Jobs + runner
# --------------------------------------------------------------------------- #

@dataclass
class Job:
    id: str
    name: str
    dir: Path
    melody_only: bool = True
    max_seconds: Optional[float] = None
    state: str = "queued"          # queued | running | done | failed | cancelled
    stage: str = "queued"
    progress: float = 0.0
    window: int = 0
    windows: int = 0
    abc: Optional[str] = None
    abc_error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    midi: Optional[bytes] = None
    seconds: Optional[float] = None
    error: Optional[str] = None
    timing: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cancel: bool = False
    last_access: float = field(default_factory=time.time)

    @property
    def audio_path(self) -> Path:
        return self.dir / "input.bin"

    def to_dict(self) -> dict[str, Any]:
        summary = abc_summary(self.abc) if self.abc else {}
        return {
            "job_id": self.id, "name": self.name, "state": self.state, "stage": self.stage,
            "progress": self.progress, "window": self.window, "windows": self.windows,
            "melody_only": self.melody_only, "seconds": self.seconds,
            "abc": self.abc, "abc_error": self.abc_error, "warnings": self.warnings,
            "has_midi": self.midi is not None, "error": self.error, "timing": self.timing,
            "key": summary.get("key"), "bpm": summary.get("bpm"), "bars": summary.get("bars"),
            "sections": summary.get("sections", []),
            "created_at": self.created_at, "started_at": self.started_at, "finished_at": self.finished_at,
        }


class Runner:
    """Owns the model and the single worker thread. `factory` builds the
    model (swapped for a fake in tests); `model` is None while unloaded."""

    def __init__(self, factory: Optional[Callable[[], Any]] = None):
        self.factory = factory or self._load_model
        self.model: Any = None
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.q: "queue.Queue[str]" = queue.Queue()
        self.current: Optional[str] = None
        self.last_activity = time.time()
        self._thread: Optional[threading.Thread] = None
        self._sweeper: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._work, name="sheetsage-worker", daemon=True)
            self._thread.start()
        if self._sweeper is None:
            self._sweeper = threading.Thread(target=self._sweep, name="sheetsage-sweeper", daemon=True)
            self._sweeper.start()

    def stop(self) -> None:
        self._stop.set()

    def _load_model(self) -> Any:
        import torch  # heavy import, only on first job
        from transformers import AutoModel
        torch.set_num_threads(min(4, torch.get_num_threads()))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = time.time()
        model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, local_files_only=LOCAL_FILES_ONLY)
        model = model.eval().to(device)
        log.info("SheetSage2 loaded on %s in %.1fs", device, time.time() - t)
        return model

    def ensure_loaded(self, job: Optional[Job] = None) -> Any:
        if self.model is None:
            if job is not None:
                self._set(job, stage="loading", progress=stage_progress("loading", 0.1))
            self.model = self.factory()
        return self.model

    def unload(self) -> bool:
        with self.lock:
            if self.model is None or self.current is not None:
                return False
            self.model = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — no torch in tests
            pass
        log.info("SheetSage2 unloaded (idle)")
        return True

    def submit(self, audio: bytes, name: str, melody_only: bool, max_seconds: Optional[float]) -> tuple[Job, int]:
        with self.lock:
            pending = sum(1 for j in self.jobs.values() if j.state == "queued")
            if pending >= MAX_QUEUE:
                raise BadRequest(f"queue is full ({MAX_QUEUE} pending)")
            job_id = uuid.uuid4().hex[:12]
            job = Job(id=job_id, name=name, dir=Path(SCRATCH_DIR) / job_id, melody_only=melody_only,
                      max_seconds=max_seconds)
            position = pending + (1 if self.current is not None else 0)
            self.jobs[job_id] = job
            self.last_activity = time.time()
        job.dir.mkdir(parents=True, exist_ok=True)
        job.audio_path.write_bytes(audio)
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
            return
        self._drop(job)

    def _drop(self, job: Job) -> None:
        with self.lock:
            self.jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    def _set(self, job: Job, **fields: Any) -> None:
        with self.lock:
            for k, v in fields.items():
                setattr(job, k, v)

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
                log.info("job %s done: %s bars in %.1fs", job.id, abc_summary(job.abc).get("bars"),
                         job.finished_at - job.started_at)
            except InterruptedError:
                self._set(job, state="cancelled", stage="done", finished_at=time.time())
                log.info("job %s cancelled", job.id)
            except Exception as exc:  # noqa: BLE001 — surfaced on the job
                log.exception("job %s failed", job.id)
                self._set(job, state="failed", stage="done", error=f"{type(exc).__name__}: {exc}",
                          finished_at=time.time())
            finally:
                # The input audio is only needed for the run itself.
                try:
                    job.audio_path.unlink(missing_ok=True)
                except OSError:
                    pass
                with self.lock:
                    self.current = None
                    self.last_activity = time.time()

    def _run(self, job: Job) -> None:
        model = self.ensure_loaded(job)
        if job.cancel:
            raise InterruptedError("cancelled before start")
        t = time.time()
        self._set(job, stage="decoding", progress=stage_progress("decoding", 0.5),
                  seconds=probe_seconds(job.audio_path))
        audio = job.audio_path.read_bytes()

        def progress(value: dict[str, Any]) -> None:
            if job.cancel:
                raise InterruptedError("cancelled")
            if value.get("stage") == "encoding":
                w, n = int(value.get("window") or 0), int(value.get("windows") or 1)
                self._set(job, stage="encoding", window=w, windows=n,
                          progress=stage_progress("encoding", (w - 1) / max(1, n)))

        options: dict[str, Any] = dict(output_dir=None, melody_only=job.melody_only, progress=progress, dtype=DTYPE)
        if job.max_seconds:
            options["max_seconds"] = float(job.max_seconds)
        try:
            result = model.transcribe(audio, **options)
        except RuntimeError as exc:
            # melody_only ABC could not be built: the transcription itself
            # completed (error.result) but a cover needs the score → fail loudly.
            partial = getattr(exc, "result", None)
            if isinstance(partial, dict):
                self._set(job, warnings=list(partial.get("warnings") or []),
                          midi=partial.get("midi") if isinstance(partial.get("midi"), (bytes, bytearray)) else None)
            raise
        self._set(job, stage="scoring", progress=stage_progress("scoring", 0.5))
        abc = result.get("abc")
        midi = result.get("midi")
        self._set(job, abc=abc if isinstance(abc, str) and abc.strip() else None,
                  abc_error=result.get("abc_error"), warnings=list(result.get("warnings") or []),
                  midi=bytes(midi) if isinstance(midi, (bytes, bytearray)) else None)
        job.timing["transcribe_seconds"] = round(time.time() - t, 2)
        if job.abc is None:
            raise RuntimeError(job.abc_error or "no ABC score was produced")

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
        if idle and self.model is not None:
            self.unload()


runner = Runner()


@app.on_event("startup")
async def _startup() -> None:
    Path(SCRATCH_DIR).mkdir(parents=True, exist_ok=True)
    runner.start()


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

class TranscribeRequest(BaseModel):
    audio_b64: str = Field(..., description="the song, any ffmpeg-readable format")
    name: str = ""
    melody_only: bool = True          # chord-free score, what YuE2 cover mode wants
    max_seconds: Optional[float] = None


def _gpu_info() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        free, total = torch.cuda.mem_get_info()
        return {"available": True, "name": torch.cuda.get_device_name(0),
                "free_gib": round(free / 2**30, 1), "total_gib": round(total / 2**30, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "loaded": runner.model is not None, "busy": runner.current,
            "queue": runner.q.qsize(), "model": MODEL, "dtype": DTYPE, "gpu": _gpu_info()}


@app.post("/transcribe")
async def transcribe(req: TranscribeRequest) -> dict[str, Any]:
    try:
        audio = decode_audio_b64(req.audio_b64)
        max_seconds = req.max_seconds
        if max_seconds is not None and not 0 < max_seconds <= MAX_SECONDS:
            raise BadRequest(f"max_seconds must be within (0, {MAX_SECONDS:g}]")
        job, position = runner.submit(audio, (req.name or "song")[:120], bool(req.melody_only),
                                      max_seconds if max_seconds is not None else MAX_SECONDS)
    except BadRequest as exc:
        raise HTTPException(400, str(exc))
    return {"job_id": job.id, "state": job.state, "position": position}


@app.get("/jobs")
async def jobs() -> dict[str, Any]:
    return {"busy": runner.current, "loaded": runner.model is not None,
            "jobs": [j.to_dict() for j in runner.jobs.values()]}


def _job(job_id: str) -> Job:
    try:
        return runner.get(job_id)
    except KeyError:
        raise HTTPException(404, "no such job")


@app.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    return _job(job_id).to_dict()


@app.get("/jobs/{job_id}/midi")
async def job_midi(job_id: str) -> Response:
    job = _job(job_id)
    if job.midi is None:
        raise HTTPException(404, "no MIDI for this job")
    return Response(content=job.midi, media_type="audio/midi")


@app.delete("/jobs/{job_id}")
async def job_delete(job_id: str) -> dict[str, Any]:
    _job(job_id)
    runner.delete(job_id)
    return {"ok": True}


@app.post("/unload")
async def unload() -> dict[str, Any]:
    return {"unloaded": runner.unload()}
