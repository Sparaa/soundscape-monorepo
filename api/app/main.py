"""Soundscape API. Phase 1: stations + seeds (ingest → analysis → profile). Health from Phase 0."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import logging

import asyncio
import threading

from fastapi.responses import Response

from . import abc as abclib, analyze, composer, config, db, ingest, library, llm as llmmod, profile, radio, sidecars

log = logging.getLogger("soundscape")
app = FastAPI(title="Soundscape", version="0.2.0")


@app.middleware("http")
async def _errors_as_json(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:  # registered BEFORE CORS so CORS wraps it: the browser sees a readable 500, not a NetworkError
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse({"detail": f"internal error: {type(e).__name__}: {str(e)[:200]}"}, status_code=500)


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
state: dict[str, Any] = {}


_local = threading.local()


def con():
    """This thread's SQLite connection to the current library (request threads, the event loop and to_thread workers
    each get their own; WAL keeps them from blocking one another)."""
    path = str(config.LIBRARY_DIR)
    if getattr(_local, "path", None) != path or getattr(_local, "con", None) is None:
        _local.con, _local.path = db.connect(config.LIBRARY_DIR), path
    return _local.con


def reset_db() -> None:
    """Tests switch LIBRARY_DIR: drop this thread's connection and the cached store/radios."""
    _local.con = None
    for k in ("store", "radios", "renderer"):
        state.pop(k, None)


def store() -> radio.Store:
    if "store" not in state:
        state["store"] = radio.Store(con, config.LIBRARY_DIR)      # the getter, not a connection: per-thread
    return state["store"]


async def _render_tags(audio: bytes):
    """CLAP tags of a finished render for the gate (SheetSage2 sidecar `describe`); None when the sidecar is busy/down."""
    try:
        return (await analyze.transcribe(audio, name="render-check", timeout_s=240))["tags"]
    except Exception as e:  # the gate treats missing tags as "no similarity data"
        log.warning("render tag check skipped: %s", e)
        return None


def renderer():
    if "renderer" not in state:
        state["renderer"] = radio.make_renderer(llm=llmmod.LLM(), yue2=composer.Yue2(), analyze_tags=_render_tags, library=config.LIBRARY_DIR)
    return state["renderer"]


def radio_for(sid: str) -> radio.Radio:
    radios: dict[str, radio.Radio] = state.setdefault("radios", {})
    if sid not in radios:
        radios[sid] = radio.Radio(sid, store=store(), renderer=renderer())
    return radios[sid]


@app.on_event("startup")
async def _startup() -> None:
    con()
    added = store().backfill_auto_playlists()
    if added:
        log.info("auto playlists: %d earlier songs added", added)
    state["prune_task"] = asyncio.create_task(_prune_loop())


async def _prune_loop() -> None:
    while True:
        try:
            gone = library.prune(con(), config.LIBRARY_DIR)
            if gone:
                log.info("pruned %d unsaved songs", len(gone))
        except Exception as e:
            log.warning("prune: %s", e)
        await asyncio.sleep(3600)


@app.get("/healthz")
async def healthz() -> dict:
    side = await sidecars.all_health()
    return {"ok": all(s["ok"] for s in side.values()), "sidecars": side,
            "llm": {"base_url": config.LLM_BASE_URL, "model": config.LLM_MODEL}, "library": str(config.LIBRARY_DIR)}


# ---- stations ---------------------------------------------------------------

class StationCreate(BaseModel):
    name: str
    language: str = "English"


def _row(r) -> dict[str, Any]:
    d = dict(r)
    for k in ("profile", "settings", "analysis"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except ValueError:
                pass
    return d


def _station(sid: str) -> dict[str, Any]:
    r = con().execute("SELECT * FROM stations WHERE id=?", (sid,)).fetchone()
    if not r:
        raise HTTPException(404, "station not found")
    s = _row(r)
    s["seeds"] = [_seed_summary(_row(x)) for x in con().execute("SELECT * FROM seeds WHERE station_id=? ORDER BY created", (sid,))]
    return s


def _seed_summary(seed: dict[str, Any]) -> dict[str, Any]:
    a = seed.get("analysis") or {}
    return {"id": seed["id"], "title": seed["title"], "source": seed["source"], "seconds": seed["seconds"], "created": seed["created"],
            "key": a.get("key"), "bpm": a.get("bpm"), "sections": a.get("sections"), "style_guess": a.get("style_guess"),
            "promoted_sections": a.get("promoted_sections"), "warnings": a.get("warnings"), "has_score": bool(a.get("abc"))}


def _rebuild_profile(sid: str) -> dict[str, Any]:
    st = _row(con().execute("SELECT * FROM stations WHERE id=?", (sid,)).fetchone())
    analyses = [_row(x)["analysis"] for x in con().execute("SELECT analysis FROM seeds WHERE station_id=?", (sid,))]
    analyses = [a for a in analyses if isinstance(a, dict)]
    prof = profile.build_profile(analyses, language=(st.get("settings") or {}).get("language", "English")) if analyses else None
    con().execute("UPDATE stations SET profile=? WHERE id=?", (json.dumps(prof) if prof else None, sid))
    con().commit()
    return prof


@app.post("/stations")
def create_station(req: StationCreate) -> dict:
    sid = uuid.uuid4().hex[:12]
    name = " ".join(req.name.split())[:80] or "Untitled station"
    con().execute("INSERT INTO stations (id, name, created, profile, settings) VALUES (?,?,?,?,?)",
                  (sid, name, time.time(), None, json.dumps({"language": req.language})))
    con().commit()
    return _station(sid)


@app.get("/stations")
def list_stations() -> list[dict]:
    return [_station(_row(r)["id"]) for r in con().execute("SELECT id FROM stations ORDER BY created DESC")]


@app.get("/stations/{sid}")
def get_station(sid: str) -> dict:
    return _station(sid)


@app.delete("/stations/{sid}")
def delete_station(sid: str) -> dict:
    _station(sid)
    for r in con().execute("SELECT id FROM seeds WHERE station_id=?", (sid,)):
        for p in (config.LIBRARY_DIR / "seeds").glob(f"{r['id']}.*"):
            p.unlink(missing_ok=True)
    con().execute("DELETE FROM seeds WHERE station_id=?", (sid,))
    con().execute("DELETE FROM stations WHERE id=?", (sid,))
    con().commit()
    return {"ok": True}


# ---- seeds ------------------------------------------------------------------

@app.post("/stations/{sid}/seeds")
async def add_seed(sid: str, file: Optional[UploadFile] = File(default=None), url: Optional[str] = Form(default=None),
                   title: Optional[str] = Form(default=None)) -> dict:
    """Multipart: either `file` (any audio) or `url` (anything yt-dlp fetches). The audio is stored under
    library/seeds and analysed (SheetSage2 score + CLAP sound tags); the station profile is rebuilt."""
    st = _station(sid)
    seed_id = uuid.uuid4().hex[:12]
    if file is not None:
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty file")
        ext = ingest.ext_for(file.filename or "", file.content_type)
        meta = {"title": title or Path(file.filename or "upload").stem, "source": f"upload:{file.filename}"}
    elif url and ingest.is_url(url):
        try:
            data, meta = await ingest.fetch_link(url.strip())
        except abclib.MusicError as e:
            raise HTTPException(502, str(e))
        ext = "m4a"
        if title:
            meta["title"] = title
    else:
        raise HTTPException(400, "send a file or a URL")
    path = ingest.save_seed_audio(config.LIBRARY_DIR, seed_id, data, ext)
    seconds = ingest.probe_seconds(path) or meta.get("duration")
    try:
        analysis = await analyze.transcribe(data, name=meta["title"], language=(st.get("settings") or {}).get("language", "English"))
    except abclib.MusicError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(502, f"analysis failed: {e}")
    con().execute("INSERT INTO seeds (id, station_id, title, source, seconds, analysis, created) VALUES (?,?,?,?,?,?,?)",
                  (seed_id, sid, meta["title"][:200], meta.get("source"), seconds, json.dumps(analysis), time.time()))
    con().commit()
    _rebuild_profile(sid)
    return _station(sid)


@app.delete("/seeds/{seed_id}")
def delete_seed(seed_id: str) -> dict:
    r = con().execute("SELECT station_id FROM seeds WHERE id=?", (seed_id,)).fetchone()
    if not r:
        raise HTTPException(404, "seed not found")
    for p in (config.LIBRARY_DIR / "seeds").glob(f"{seed_id}.*"):
        p.unlink(missing_ok=True)
    con().execute("DELETE FROM seeds WHERE id=?", (seed_id,))
    con().commit()
    _rebuild_profile(r["station_id"])
    return _station(r["station_id"])


@app.get("/seeds/{seed_id}/audio")
def seed_audio(seed_id: str):
    hits = list((config.LIBRARY_DIR / "seeds").glob(f"{seed_id}.*"))
    if not hits:
        raise HTTPException(404, "no audio")
    return FileResponse(hits[0])


@app.get("/seeds/{seed_id}/analysis")
def seed_analysis(seed_id: str) -> dict:
    r = con().execute("SELECT analysis FROM seeds WHERE id=?", (seed_id,)).fetchone()
    if not r:
        raise HTTPException(404, "seed not found")
    return _row(r)["analysis"]


# ---- radio ------------------------------------------------------------------

async def _ensure_themes(sid: str) -> None:
    """First Play on a station: ask the LLM for its lyrical themes once (kept in settings)."""
    st = _row(con().execute("SELECT * FROM stations WHERE id=?", (sid,)).fetchone())
    settings = st.get("settings") or {}
    if settings.get("themes") or not st.get("profile"):
        return
    try:
        themes = await llmmod.station_themes(llmmod.LLM(), style=st["profile"]["style"], blurb=settings.get("blurb", ""))
    except Exception as e:
        log.warning("themes: %s", e)
        return
    settings["themes"] = themes
    con().execute("UPDATE stations SET settings=? WHERE id=?", (json.dumps(settings), sid))
    con().commit()


@app.post("/stations/{sid}/play")
async def radio_play(sid: str) -> dict:
    st = _station(sid)
    if not st.get("profile"):
        raise HTTPException(400, "seed the station first")
    await _ensure_themes(sid)
    r = radio_for(sid)
    r.play()
    return r.status()


@app.post("/stations/{sid}/stop")
def radio_stop(sid: str) -> dict:
    _station(sid)
    r = radio_for(sid)
    r.stop()
    return r.status()


@app.post("/stations/{sid}/next")
def radio_next(sid: str, song_id: Optional[str] = None) -> dict:
    """Next cued song; `?song_id=` plays a specific cued one now (the Up next list is clickable)."""
    _station(sid)
    r = radio_for(sid)
    song = r.next(song_id)
    return {"song": song, "status": r.status()}


@app.get("/stations/{sid}/radio")
def radio_status(sid: str) -> dict:
    _station(sid)
    return radio_for(sid).status()


class StationSettings(BaseModel):
    covers: Optional[float] = None      # 0 = never covers … 2 = cover-heavy
    blurb: Optional[str] = None
    themes: Optional[list[str]] = None


@app.patch("/stations/{sid}/settings")
def patch_settings(sid: str, req: StationSettings) -> dict:
    st = _row(con().execute("SELECT * FROM stations WHERE id=?", (sid,)).fetchone() or {}) or None
    if not st:
        raise HTTPException(404, "station not found")
    settings = st.get("settings") or {}
    for k, v in req.model_dump(exclude_none=True).items():
        settings[k] = v
    con().execute("UPDATE stations SET settings=? WHERE id=?", (json.dumps(settings), sid))
    con().commit()
    return _station(sid)


@app.get("/stations/{sid}/playlist")
def station_playlist(sid: str) -> dict:
    """The station's own playlist — every song the radio cued, in order; replayable from the station page."""
    _station(sid)
    return _playlist(store().auto_playlist(sid))


@app.get("/stations/{sid}/profile/effective")
def effective_profile(sid: str) -> dict:
    """The station profile as the agent sees it right now: seeds + steering from likes / less-like-this votes."""
    _station(sid)
    return store().station(sid)["profile"] or {}


# ---- songs ------------------------------------------------------------------

def _song(song_id: str) -> dict[str, Any]:
    r = con().execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
    if not r:
        raise HTTPException(404, "song not found")
    return store()._song(r)


@app.get("/stations/{sid}/songs")
def station_songs(sid: str, limit: int = 50) -> list[dict]:
    _station(sid)
    return [store()._song(r) for r in con().execute(
        "SELECT * FROM songs WHERE station_id=? AND status != 'rejected' ORDER BY created DESC LIMIT ?", (sid, limit))]


@app.get("/songs/{song_id}")
def get_song(song_id: str) -> dict:
    return _song(song_id)


@app.get("/songs/{song_id}/audio")
def song_audio(song_id: str):
    s = _song(song_id)
    p = Path(s["path"])
    if not p.exists():
        raise HTTPException(404, "audio missing")
    return FileResponse(p, media_type="audio/flac", filename=f"{s.get('title') or song_id}.flac")


class SongFlags(BaseModel):
    saved: Optional[bool] = None
    liked: Optional[bool] = None       # like = vote +1 (and liked=1); liked=false clears the vote
    vote: Optional[int] = None         # -1 "less like this" / 0 / +1
    title: Optional[str] = None


@app.patch("/songs/{song_id}")
def patch_song(song_id: str, req: SongFlags) -> dict:
    _song(song_id)
    if req.saved is not None:
        con().execute("UPDATE songs SET saved=? WHERE id=?", (1 if req.saved else 0, song_id))
    if req.liked is not None:
        con().execute("UPDATE songs SET liked=?, vote=? WHERE id=?", (1 if req.liked else 0, 1 if req.liked else 0, song_id))
    if req.vote is not None:
        v = max(-1, min(1, int(req.vote)))
        con().execute("UPDATE songs SET vote=?, liked=? WHERE id=?", (v, 1 if v > 0 else 0, song_id))
    if req.title is not None:
        con().execute("UPDATE songs SET title=? WHERE id=?", (" ".join(req.title.split())[:120], song_id))
    con().commit()
    return _song(song_id)


@app.delete("/songs/{song_id}")
def delete_song(song_id: str) -> dict:
    s = _song(song_id)
    for p in (Path(s["path"]), Path(s["path"]).with_suffix(".json")):
        p.unlink(missing_ok=True)
    con().execute("DELETE FROM playlist_items WHERE song_id=?", (song_id,))
    con().execute("DELETE FROM songs WHERE id=?", (song_id,))
    con().commit()
    return {"ok": True}


# ---- library ----------------------------------------------------------------

@app.get("/library")
def library_songs(saved: Optional[bool] = None, liked: Optional[bool] = None, limit: int = 500) -> list[dict]:
    """Across stations: saved and/or liked songs plus imports (never rejected renders)."""
    where = ["status != 'rejected'"]
    if saved:
        where.append("saved=1")
    if liked:
        where.append("vote > 0")
    if saved is None and liked is None:
        where.append("(saved=1 OR vote > 0 OR station_id IS NULL)")
    rows = con().execute(f"SELECT * FROM songs WHERE {' AND '.join(where)} ORDER BY created DESC LIMIT ?", (limit,))
    return [store()._song(r) for r in rows]


@app.post("/library/import")
async def library_import(file: UploadFile = File(...), title: Optional[str] = Form(default=None)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    tmp = config.LIBRARY_DIR / "songs" / f"import-{uuid.uuid4().hex[:8]}{Path(file.filename or '').suffix or '.bin'}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    seconds = ingest.probe_seconds(tmp)
    tmp.unlink(missing_ok=True)
    sid = library.import_song(con(), config.LIBRARY_DIR, data=data, filename=file.filename or "import.bin", title=title, seconds=seconds)
    return _song(sid)


@app.post("/library/prune")
def library_prune() -> dict:
    return {"pruned": library.prune(con(), config.LIBRARY_DIR)}


# ---- playlists --------------------------------------------------------------

class PlaylistCreate(BaseModel):
    name: str


class PlaylistPatch(BaseModel):
    name: Optional[str] = None
    order: Optional[list[str]] = None


class PlaylistAdd(BaseModel):
    song_id: str
    position: Optional[int] = None


def _playlist(pid: str) -> dict[str, Any]:
    r = con().execute("SELECT * FROM playlists WHERE id=?", (pid,)).fetchone()
    if not r:
        raise HTTPException(404, "playlist not found")
    ids = library.playlist_items(con(), pid)
    songs = {s["id"]: s for s in (store()._song(x) for x in con().execute(
        f"SELECT * FROM songs WHERE id IN ({','.join('?' * len(ids))})", ids))} if ids else {}
    items = [songs[i] for i in ids if i in songs]
    return {"id": r["id"], "name": r["name"], "created": r["created"], "items": items,
            "seconds": round(sum(float(s.get("seconds") or 0) for s in items), 1)}


@app.post("/playlists")
def playlists_create(req: PlaylistCreate) -> dict:
    return _playlist(library.create_playlist(con(), req.name))


@app.get("/playlists")
def playlists_list() -> list[dict]:
    return [_playlist(r["id"]) for r in con().execute("SELECT id FROM playlists ORDER BY created DESC")]


@app.get("/playlists/{pid}")
def playlists_get(pid: str) -> dict:
    return _playlist(pid)


@app.patch("/playlists/{pid}")
def playlists_patch(pid: str, req: PlaylistPatch) -> dict:
    _playlist(pid)
    if req.name is not None:
        con().execute("UPDATE playlists SET name=? WHERE id=?", (" ".join(req.name.split())[:80] or "Playlist", pid))
        con().commit()
    if req.order is not None:
        library.set_items(con(), pid, req.order)
    return _playlist(pid)


@app.post("/playlists/{pid}/items")
def playlists_add(pid: str, req: PlaylistAdd) -> dict:
    _playlist(pid)
    _song(req.song_id)
    library.add_item(con(), pid, req.song_id, req.position)
    return _playlist(pid)


@app.delete("/playlists/{pid}/items/{song_id}")
def playlists_remove(pid: str, song_id: str) -> dict:
    _playlist(pid)
    library.set_items(con(), pid, [s for s in library.playlist_items(con(), pid) if s != song_id])
    return _playlist(pid)


@app.delete("/playlists/{pid}")
def playlists_delete(pid: str) -> dict:
    _playlist(pid)
    con().execute("DELETE FROM playlist_items WHERE playlist_id=?", (pid,))
    con().execute("DELETE FROM playlists WHERE id=?", (pid,))
    con().commit()
    return {"ok": True}


@app.get("/playlists/{pid}/export.zip")
def playlists_export(pid: str):
    pl = _playlist(pid)
    data = library.export_zip(con(), pid, pl["name"], pl["items"])
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{pl["name"].replace(chr(34), "")}.zip"'})
