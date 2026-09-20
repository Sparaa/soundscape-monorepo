import asyncio, io
import httpx
from fastapi.testclient import TestClient

from app import analyze, ingest, main

ANALYSIS = {"abc": "X:1\nK:C\n% verse\nV: Vocal\nCDEF|", "seconds": 30.0, "key": "C major", "bpm": 100, "bars": 1, "sections": ["verse"],
            "tags": {"genre": [{"label": "folk", "p": 0.9}], "vocal": [{"label": "male vocals", "p": 0.8}]}, "style_guess": "English, folk", "warnings": []}


def test_station_lifecycle_with_upload_and_link(monkeypatch):
    calls = {}

    async def fake_transcribe(audio, *, name="", language="English", **kw):
        calls.setdefault("names", []).append(name)
        return dict(ANALYSIS, promoted_sections=[])

    async def fake_fetch(url, *, client=None):
        return b"FAKEAUDIO", {"title": "Linked song", "uploader": "someone", "duration": 30.0, "source": url}

    monkeypatch.setattr(analyze, "transcribe", fake_transcribe)
    monkeypatch.setattr(ingest, "fetch_link", fake_fetch)
    monkeypatch.setattr(ingest, "probe_seconds", lambda p: 30.0)
    c = TestClient(main.app)
    st = c.post("/stations", json={"name": "  Late night   drive "}).json()
    assert st["name"] == "Late night drive" and st["seeds"] == [] and st["profile"] is None
    sid = st["id"]
    r = c.post(f"/stations/{sid}/seeds", files={"file": ("mine.mp3", io.BytesIO(b"ID3..."), "audio/mpeg")})
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["seeds"][0]["title"] == "mine" and st["seeds"][0]["key"] == "C major" and st["seeds"][0]["has_score"]
    assert st["profile"]["seeds"] == 1 and st["profile"]["style"].startswith("English, folk")
    r = c.post(f"/stations/{sid}/seeds", data={"url": "https://youtu.be/abc"})
    assert r.status_code == 200 and len(r.json()["seeds"]) == 2 and r.json()["seeds"][1]["source"] == "https://youtu.be/abc"
    assert calls["names"] == ["mine", "Linked song"]
    seed_id = st["seeds"][0]["id"]
    assert c.get(f"/seeds/{seed_id}/audio").content == b"ID3..."
    assert c.get(f"/seeds/{seed_id}/analysis").json()["bpm"] == 100
    assert c.post(f"/stations/{sid}/seeds", data={"url": "not a url"}).status_code == 400
    r = c.delete(f"/seeds/{seed_id}")
    assert len(r.json()["seeds"]) == 1 and r.json()["profile"]["seeds"] == 1
    assert c.delete(f"/stations/{sid}").json() == {"ok": True, "deleted_songs": 0}
    assert c.get(f"/stations/{sid}").status_code == 404


def test_fetch_link_uses_clipgrab_audio_only():
    seen = {}
    def handler(req: httpx.Request):
        if req.url.path == "/info":
            return httpx.Response(200, json={"title": "Vid", "uploader": "U", "duration": 42})
        seen["body"] = req.read()
        return httpx.Response(200, content=b"M4A")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    data, meta = asyncio.run(ingest.fetch_link("https://example.com/v", client=client))
    import json
    assert data == b"M4A" and meta["title"] == "Vid" and json.loads(seen["body"]) == {"url": "https://example.com/v", "audio_only": True}
    assert ingest.ext_for("song.FLAC", None) == "flac" and ingest.ext_for("", "audio/mpeg") == "mp3" and ingest.is_url("ftp://x") is False


def test_transcribe_polls_the_job(monkeypatch):
    states = iter(["queued", "running", "done"])
    def handler(req: httpx.Request):
        if req.url.path == "/transcribe":
            return httpx.Response(200, json={"job_id": "j1", "state": "queued", "position": 0})
        return httpx.Response(200, json={"state": next(states), **ANALYSIS})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = asyncio.run(analyze.transcribe(b"x", name="n", client=client, poll_s=0))
    assert out["key"] == "C major" and out["promoted_sections"] == [] and "timing" in out


def test_radio_routes_with_a_fake_renderer(monkeypatch):
    import asyncio, json, time
    from app import radio
    from app.agent import Plan

    async def fake_render(station, seeds, plan, progress):
        await asyncio.sleep(0)
        sid = f"r{int(time.time()*1000)%100000}{len(station['name'])}"
        return {"id": sid, "title": "T", "style": "s", "lyrics": "[Verse]\nla", "abc": None, "plan": plan.to_dict(), "seconds": 100.0,
                "path": "/nonexistent.flac", "explain": plan.explain, "gate": {"ok": True, "reasons": []}}
    monkeypatch.setattr(main, "renderer", lambda: fake_render)
    async def no_themes(sid): return None
    monkeypatch.setattr(main, "_ensure_themes", no_themes)
    with TestClient(main.app) as c:
        st = c.post("/stations", json={"name": "R"}).json(); sid = st["id"]
        assert c.post(f"/stations/{sid}/play").status_code == 400          # no profile yet
        main.con().execute("UPDATE stations SET profile=? WHERE id=?", (json.dumps({"style": "English, pop, 110 BPM", "bpm": {"low": 100, "high": 120, "center": 110}, "keys": [], "tags": {}}), sid))
        main.con().commit()
        s = c.post(f"/stations/{sid}/play").json()
        assert s["state"] in ("warming", "playing")
        for _ in range(100):
            time.sleep(0.05)
            if len(c.get(f"/stations/{sid}/radio").json()["ready"]) >= 2:
                break
        nx = c.post(f"/stations/{sid}/next").json()
        assert nx["song"]["title"] == "T" and nx["status"]["now_playing"]["id"] == nx["song"]["id"]
        song_id = nx["song"]["id"]
        assert c.patch(f"/songs/{song_id}", json={"saved": True}).json()["saved"] is True
        assert c.get(f"/stations/{sid}/songs").json()[0]["id"] in {x["id"] for x in c.get(f"/stations/{sid}/songs").json()}
        assert c.get(f"/songs/{song_id}/audio").status_code == 404        # fake path
        assert c.patch(f"/stations/{sid}/settings", json={"covers": 0.5}).json()["settings"]["covers"] == 0.5
        pl = c.get(f"/stations/{sid}/playlist").json()                    # the station's own playlist holds what it cued
        assert pl["name"] == "R — radio" and song_id in [s["id"] for s in pl["items"]] and len(pl["items"]) >= 2
        assert c.get(f"/stations/{sid}/playlist").json()["id"] == pl["id"]      # stable id
        assert c.post(f"/stations/{sid}/stop").json()["state"] in ("stopping", "stopped")


def test_concurrent_reads_never_lose_a_station():
    """Regression 2026-09-17: one shared SQLite connection across worker threads returned "station not found" / 500
    for stations that exist once the UI's polling overlapped (the browser showed the bare 500s as NetworkError)."""
    import concurrent.futures as cf
    with TestClient(main.app) as c:
        sid = c.post("/stations", json={"name": "Busy"}).json()["id"]
        paths = [f"/stations/{sid}/radio", f"/stations/{sid}", "/stations", f"/stations/{sid}/songs", "/playlists", "/library"]
        with cf.ThreadPoolExecutor(24) as ex:
            codes = list(ex.map(lambda i: c.get(paths[i % len(paths)]).status_code, range(360)))
        assert set(codes) == {200}, {k: codes.count(k) for k in set(codes)}


def test_start_fresh_and_delete_station_remove_the_songs(tmp_path, monkeypatch):
    import json, time
    from app import config, radio
    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path)
    main.reset_db()
    rendered = []

    async def fake_render(station, seeds, plan, progress):
        await asyncio.sleep(0.02)
        sid = f"new{len(rendered)}"; rendered.append(sid)
        p = tmp_path / "songs" / f"{sid}.flac"; p.write_bytes(b"fLaC")
        return {"id": sid, "title": sid, "style": "s", "lyrics": "l", "abc": None, "plan": plan.to_dict(), "seconds": 120.0,
                "path": str(p), "explain": plan.explain, "gate": {"ok": True, "reasons": []}}

    async def no_themes(sid):
        return None
    monkeypatch.setattr(main, "renderer", lambda: fake_render)
    monkeypatch.setattr(main, "_ensure_themes", no_themes)
    with TestClient(main.app) as c:
        sid = c.post("/stations", json={"name": "Fresh"}).json()["id"]
        prof = {"style": "English, pop, 110 BPM", "bpm": {"low": 100, "high": 120, "center": 110}, "keys": [], "tags": {}, "sections": ["verse"], "phrases": {}}
        main.con().execute("UPDATE stations SET profile=? WHERE id=?", (json.dumps(prof), sid))
        main.con().execute("INSERT INTO seeds (id, station_id, title, source, seconds, analysis, created) VALUES ('sd1',?,'seed','upload',30,?,?)", (sid, json.dumps(ANALYSIS), time.time()))
        main.con().commit()
        (tmp_path / "seeds").mkdir(exist_ok=True); (tmp_path / "seeds" / "sd1.mp3").write_bytes(b"ID3")
        st = main.store()
        for i in range(2):
            p = tmp_path / "songs" / f"old{i}.flac"; p.write_bytes(b"fLaC"); p.with_suffix(".json").write_text("{}")
            st.add_song(sid, {"id": f"old{i}", "title": f"old {i}", "style": "s", "lyrics": "l", "abc": None, "plan": {"mode": "inspired"}, "seconds": 100.0,
                              "path": str(p), "explain": "", "gate": {"ok": True, "reasons": []}}, status="ready")
        st.add_song(sid, {"id": "rej", "title": "rejected", "style": "s", "lyrics": "l", "abc": None, "plan": {"mode": "inspired"}, "seconds": 10.0,
                          "path": str(tmp_path / "songs" / "rej.flac"), "explain": "", "gate": {"ok": False, "reasons": ["too short"]}}, status="rejected")
        pid = st.auto_playlist(sid)
        assert c.get(f"/stations/{sid}").json()["songs"] == 2 and len(c.get(f"/stations/{sid}/playlist").json()["items"]) == 2

        r = c.post(f"/stations/{sid}/fresh")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["deleted_songs"] == 3 and body["status"]["state"] == "warming" and body["status"]["ready"] == []
        assert not (tmp_path / "songs" / "old0.flac").exists() and not (tmp_path / "songs" / "old1.json").exists()
        assert c.get(f"/stations/{sid}/playlist").json() == c.get(f"/playlists/{pid}").json()      # same playlist row, now empty
        assert c.get(f"/playlists/{pid}").json()["items"] == []
        s = c.get(f"/stations/{sid}").json()
        assert s["songs"] == 0 and len(s["seeds"]) == 1 and s["profile"]["style"] == prof["style"] and s["settings"]["playlist_id"] == pid
        for _ in range(100):                            # the agent composes a new list from the same seeds right away
            time.sleep(0.02)
            if c.get(f"/stations/{sid}/radio").json()["ready"]:
                break
        ready = c.get(f"/stations/{sid}/radio").json()["ready"]
        assert ready and ready[0]["id"].startswith("new")
        assert c.get(f"/stations/{sid}/playlist").json()["items"][0]["id"] == ready[0]["id"]     # and the playlist refills

        r = c.delete(f"/stations/{sid}")
        assert r.status_code == 200 and r.json()["deleted_songs"] >= 1
        assert c.get(f"/stations/{sid}").status_code == 404 and c.get(f"/playlists/{pid}").status_code == 404
        assert main.con().execute("SELECT COUNT(*) FROM songs WHERE station_id=?", (sid,)).fetchone()[0] == 0
        assert not list((tmp_path / "songs").glob("new*.flac")) and not (tmp_path / "seeds" / "sd1.mp3").exists()
        assert sid not in main.state.get("radios", {})
    main.reset_db()
