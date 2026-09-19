"""The radio: one asyncio loop per station keeps the buffer of ready songs full while it plays.

States: stopped → warming (first render) → playing → stopping (finish the render in flight, keep ONE spare) → stopped.
Buffer target: 2 ready while playing/warming; 1 while stopping (the spare); 0 when stopped."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from . import abc as abclib, agent, gate, library, llm as llmmod, steer

log = logging.getLogger("soundscape.radio")
TARGET = {"stopped": 0, "warming": 2, "playing": 2, "stopping": 1}
MAX_RETRIES = 2


@dataclass
class Rendering:
    plan: dict[str, Any]
    stage: str = "planning"
    progress: float = 0.0
    started: float = field(default_factory=time.time)
    attempt: int = 1


class Radio:
    """Per-station controller. `render_track` is injectable so tests run without sidecars."""

    def __init__(self, station_id: str, *, store: "Store", renderer: "Renderer", loop_s: float = 1.0):
        self.station_id, self.store, self.renderer, self.loop_s = station_id, store, renderer, loop_s
        self.state = "stopped"
        self.rendering: Optional[Rendering] = None
        self.now_playing: Optional[dict[str, Any]] = None
        self.events: list[dict[str, Any]] = []
        self._task: Optional[asyncio.Task] = None
        self._render_task: Optional[asyncio.Task] = None

    # ---- controls
    def play(self) -> None:
        if self.state in ("stopped", "stopping"):
            self.state = "playing" if self.store.ready(self.station_id) else "warming"
            self._event("play")
        self._ensure_loop()

    def stop(self) -> None:
        if self.state in ("playing", "warming"):
            self.state = "stopping"
            self._event("stop")
        self.now_playing = None

    def next(self, song_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Pop the next ready song (the client starts playing it), or a specific cued one the listener picked.
        None → nothing cued yet (or that song is no longer cued)."""
        song = self.store.pop_ready(self.station_id, song_id)
        if song:
            self.now_playing = song
            if self.state == "warming":
                self.state = "playing"
            self._event("next", song_id=song["id"])
        return song

    def status(self) -> dict[str, Any]:
        return {"state": self.state, "now_playing": self.now_playing, "ready": self.store.ready(self.station_id),
                "rendering": ({"plan": self.rendering.plan, "stage": self.rendering.stage, "progress": self.rendering.progress,
                               "seconds": round(time.time() - self.rendering.started), "attempt": self.rendering.attempt} if self.rendering else None),
                "buffer_target": TARGET[self.state], "recent": self.store.recent(self.station_id), "events": self.events[-12:]}

    # ---- loop
    def _ensure_loop(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                target = TARGET[self.state]
                ready = len(self.store.ready(self.station_id))
                busy = self._render_task is not None and not self._render_task.done()
                if ready < target and not busy:
                    self._render_task = asyncio.create_task(self._render_one())
                    busy = True
                if self.state == "stopping" and ready >= 1 and not busy:
                    self.state = "stopped"
                    self._event("stopped", spare=ready)
                if self.state == "stopped" and not busy:
                    return
                await asyncio.sleep(self.loop_s)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # keep the loop alive on unexpected errors
            log.exception("radio loop crashed: %s", e)
            self._event("error", error=str(e)[:200])

    async def _render_one(self) -> None:
        try:
            await self._render_attempts()
        except asyncio.CancelledError:
            raise
        except Exception as e:   # LLM/sidecar unreachable etc.: log it, clear the stale "rendering" status, let the loop retry
            log.warning("render crashed: %s", e)
            self._event("render_failed", error=f"{type(e).__name__}: {e}"[:200])
            await asyncio.sleep(5)   # not a tight loop while a backend is down
        finally:
            self.rendering = None

    async def _render_attempts(self) -> None:
        station = self.store.station(self.station_id)
        seeds = self.store.seeds(self.station_id)
        history = self.store.history(self.station_id)
        for attempt in range(1, MAX_RETRIES + 2):
            plan = agent.plan_track(station=station, seeds=seeds, history=history)
            self.rendering = Rendering(plan=plan.to_dict(), attempt=attempt)
            try:
                song = await self.renderer(station, seeds, plan, self._progress)
            except abclib.MusicError as e:
                self._event("render_failed", error=str(e)[:200], mode=plan.mode)
                log.warning("render failed (%s): %s", plan.mode, e)
                continue
            if song.get("gate", {}).get("ok", True):
                self.store.add_song(self.station_id, song, status="ready")
                self._event("cued", song_id=song["id"], mode=plan.mode, explain=plan.explain)
                break
            self.store.add_song(self.station_id, song, status="rejected")
            self._event("rejected", song_id=song["id"], reasons=song["gate"]["reasons"])
            history = self.store.history(self.station_id)

    def _progress(self, stage: str, progress: float) -> None:
        if self.rendering:
            self.rendering.stage, self.rendering.progress = stage, round(progress, 3)

    def _event(self, kind: str, **data: Any) -> None:
        self.events.append({"t": time.time(), "kind": kind, **data})
        self.events = self.events[-50:]


Renderer = Callable[[dict[str, Any], list[dict[str, Any]], agent.Plan, Callable[[str, float], None]], Awaitable[dict[str, Any]]]


class Store:
    """The songs/plans view the radio needs, over the SQLite connection."""

    def __init__(self, con, library: Path):
        """con = a connection (tests) or a zero-argument getter returning THIS THREAD's connection (the app)."""
        self._con, self.library = con, library

    @property
    def con(self):
        return self._con if isinstance(self._con, sqlite3.Connection) else self._con()   # NB: Connection objects are callable

    def station(self, sid: str) -> dict[str, Any]:
        """The station with its profile STEERED by the votes on its songs (likes pull, dislikes push)."""
        r = dict(self.con.execute("SELECT * FROM stations WHERE id=?", (sid,)).fetchone())
        for k in ("profile", "settings"):
            r[k] = json.loads(r[k]) if r.get(k) else {}
        liked, disliked = [], []
        for row in self.con.execute("SELECT tags, vote FROM songs WHERE station_id=? AND vote != 0 AND tags IS NOT NULL ORDER BY created", (sid,)):
            (liked if row["vote"] > 0 else disliked).append(json.loads(row["tags"]))
        if r["profile"]:
            r["profile"] = steer.steer(r["profile"], liked, disliked)
        return r

    def seeds(self, sid: str) -> list[dict[str, Any]]:
        out = []
        for r in self.con.execute("SELECT id, title, analysis FROM seeds WHERE station_id=? ORDER BY created", (sid,)):
            out.append({"id": r["id"], "title": r["title"], "analysis": json.loads(r["analysis"]) if r["analysis"] else {}})
        return out

    def history(self, sid: str, n: int = 40) -> list[dict[str, Any]]:
        rows = self.con.execute("SELECT plan FROM songs WHERE station_id=? AND plan IS NOT NULL ORDER BY created DESC LIMIT ?", (sid, n)).fetchall()
        return [json.loads(r["plan"]) for r in reversed(rows)]

    def _song(self, r) -> dict[str, Any]:
        d = dict(r)
        for k in ("plan", "gate", "tags"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        d["liked"], d["saved"], d["vote"] = bool(d.get("liked")), bool(d.get("saved")), int(d.get("vote") or 0)
        return d

    def ready(self, sid: str) -> list[dict[str, Any]]:
        return [self._song(r) for r in self.con.execute("SELECT * FROM songs WHERE station_id=? AND status='ready' ORDER BY created", (sid,))]

    def pop_ready(self, sid: str, song_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        if song_id:
            r = self.con.execute("SELECT * FROM songs WHERE station_id=? AND id=? AND status='ready'", (sid, song_id)).fetchone()
        else:
            r = self.con.execute("SELECT * FROM songs WHERE station_id=? AND status='ready' ORDER BY created LIMIT 1", (sid,)).fetchone()
        if not r:
            return None
        self.con.execute("UPDATE songs SET status='played', played=? WHERE id=?", (time.time(), r["id"]))
        self.con.commit()
        return self._song(self.con.execute("SELECT * FROM songs WHERE id=?", (r["id"],)).fetchone())

    def recent(self, sid: str, n: int = 12) -> list[dict[str, Any]]:
        return [self._song(r) for r in self.con.execute(
            "SELECT * FROM songs WHERE station_id=? AND status IN ('played','rejected') ORDER BY created DESC LIMIT ?", (sid, n))]

    def add_song(self, sid: str, song: dict[str, Any], *, status: str) -> None:
        self.con.execute("INSERT INTO songs (id, station_id, title, style, lyrics, abc, plan, seconds, path, created, status, explain, gate, tags) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (song["id"], sid, song.get("title"), song.get("style"), song.get("lyrics"), song.get("abc"),
                          json.dumps(song.get("plan")), song.get("seconds"), song["path"], time.time(), status, song.get("explain"),
                          json.dumps(song.get("gate")), json.dumps(song.get("tags")) if song.get("tags") else None))
        self.con.commit()
        if status == "ready":
            self.auto_playlist_add(sid, song["id"])

    def auto_playlist(self, sid: str) -> str:
        """The station's own playlist ("<station> — radio"): id kept in the station settings, (re)created on demand."""
        row = self.con.execute("SELECT name, settings FROM stations WHERE id=?", (sid,)).fetchone()
        settings = json.loads(row["settings"]) if row and row["settings"] else {}
        pid = settings.get("playlist_id")
        if not pid or not self.con.execute("SELECT 1 FROM playlists WHERE id=?", (pid,)).fetchone():
            pid = library.create_playlist(self.con, f"{row['name'] if row else 'Station'} — radio")
            settings["playlist_id"] = pid
            self.con.execute("UPDATE stations SET settings=? WHERE id=?", (json.dumps(settings), sid))
            self.con.commit()
        return pid

    def auto_playlist_add(self, sid: str, song_id: str) -> str:
        """Every song that passes the gate joins the station's playlist, which also marks it saved (never pruned)."""
        pid = self.auto_playlist(sid)
        if song_id not in library.playlist_items(self.con, pid):
            library.add_item(self.con, pid, song_id)
        return pid

    def backfill_auto_playlists(self) -> int:
        """Startup: songs cued before auto-save existed join their station's playlist (oldest first, idempotent)."""
        n = 0
        for r in self.con.execute("SELECT id, station_id FROM songs WHERE station_id IS NOT NULL AND status != 'rejected' ORDER BY created").fetchall():
            pid_row = self.con.execute("SELECT settings FROM stations WHERE id=?", (r["station_id"],)).fetchone()
            if not pid_row:
                continue
            pid = (json.loads(pid_row["settings"]) if pid_row["settings"] else {}).get("playlist_id")
            if pid and r["id"] in library.playlist_items(self.con, pid):
                continue
            self.auto_playlist_add(r["station_id"], r["id"])
            n += 1
        return n


def make_renderer(*, llm: llmmod.LLM, yue2, analyze_tags: Optional[Callable[[bytes], Awaitable[dict[str, Any] | None]]],
                  library: Path) -> Renderer:
    """The real pipeline: plan → write (LLM) → render (YuE2) → gate (ffmpeg + CLAP tags) → file + sidecar JSON."""

    async def render(station: dict[str, Any], seeds: list[dict[str, Any]], plan: agent.Plan, progress: Callable[[str, float], None]) -> dict[str, Any]:
        profile = station.get("profile") or {}
        seed = next((s for s in seeds if s["id"] == plan.seed_id), None)
        seed_analysis = (seed or {}).get("analysis")
        style = agent.styled(profile.get("style") or "English, pop, 110 BPM", plan)
        progress("writing", 0.05)
        if profile.get("instrumental"):
            song = {"title": plan.theme.title(), "style": style, "lyrics": "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Verse 2]\n\n[Chorus]\n\n[Outro]"}
        else:
            song = await llmmod.write_song(llm, station=station["name"], theme=plan.theme, style=style, duration_s=plan.duration_s,
                                           language=profile.get("language", "English"), mode=plan.mode,
                                           abc=(seed_analysis or {}).get("abc") if plan.mode != "inspired" else None,
                                           source_style=(seed_analysis or {}).get("style_guess"))
        request = agent.song_request(style=style, lyrics=song["lyrics"], plan=plan, seed_analysis=seed_analysis,
                                     seed=random.randrange(0, 2**31))
        progress("rendering", 0.1)

        def on_job(s: dict[str, Any]) -> None:
            progress(s.get("stage") or "rendering", 0.1 + 0.8 * float(s.get("progress") or 0.0))

        audio, job = await yue2.render(request, on_progress=on_job)
        song_id = uuid.uuid4().hex[:12]
        path = library / "songs" / f"{song_id}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, audio)            # 40 MB write + ffmpeg below: keep the event loop free
        progress("checking", 0.92)
        tags = await analyze_tags(audio) if analyze_tags else None
        verdict = await asyncio.to_thread(gate.check, path, profile_tags=profile.get("tags"), render_tags=tags, seconds=job.get("audio_seconds"))
        out = {"id": song_id, "title": song.get("title") or plan.theme.title(), "style": request["style"], "lyrics": request["lyrics"],
               "abc": job.get("abc"), "plan": plan.to_dict(), "seconds": job.get("audio_seconds") or verdict["seconds"],
               "path": str(path), "explain": plan.explain, "gate": verdict, "tags": tags, "request": {k: v for k, v in request.items() if k not in ("abc", "hook_abc")},
               "yue2": {k: job.get(k) for k in ("planned_seconds", "timing", "truncated", "hook")}}
        (library / "songs" / f"{song_id}.json").write_text(json.dumps(out, indent=1))
        progress("done", 1.0)
        return out

    return render
