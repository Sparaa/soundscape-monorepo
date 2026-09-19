import asyncio, json
from pathlib import Path

from app import agent, db, gate, radio


def test_gate_reasons(tmp_path):
    p = tmp_path / "x.flac"; p.write_bytes(b"")
    ok = gate.check(p, seconds=180, mean_db=-18, max_db=-1, profile_tags={"genre": [{"label": "pop", "weight": 0.6}]},
                    render_tags={"genre": [{"label": "pop", "p": 0.7}, {"label": "metal", "p": 0.1}]})
    assert ok["ok"] and ok["similarity"] > 0.9
    bad = gate.check(p, seconds=20, mean_db=-50, max_db=-30, profile_tags={"genre": [{"label": "pop", "weight": 1}]},
                     render_tags={"genre": [{"label": "metal", "p": 1}]})
    assert not bad["ok"] and len(bad["reasons"]) == 4
    assert gate.check(p, seconds=100, mean_db=-20, max_db=-2)["similarity"] is None   # no tags → similarity skipped


def test_radio_buffers_two_then_keeps_one_spare(tmp_path):
    con = db.connect(tmp_path)
    con.execute("INSERT INTO stations (id, name, created, profile, settings) VALUES ('s1','S',0,?,?)",
                (json.dumps({"style": "English, pop, 110 BPM", "bpm": {"low": 100, "high": 120, "center": 110}, "keys": [], "tags": {}}), json.dumps({})))
    con.commit()
    store = radio.Store(con, tmp_path)
    rendered = []

    async def fake_render(station, seeds, plan, progress):
        progress("rendering", 0.5)
        await asyncio.sleep(0.01)
        sid = f"song{len(rendered)}"
        rendered.append(plan.mode)
        return {"id": sid, "title": plan.theme, "style": "s", "lyrics": "l", "abc": None, "plan": plan.to_dict(), "seconds": 120.0,
                "path": str(tmp_path / f"{sid}.flac"), "explain": plan.explain, "gate": {"ok": len(rendered) != 2, "reasons": [] if len(rendered) != 2 else ["too quiet"]}}

    async def scenario():
        r = radio.Radio("s1", store=store, renderer=fake_render, loop_s=0.01)
        r.play()
        assert r.state == "warming"
        for _ in range(200):
            await asyncio.sleep(0.01)
            if len(store.ready("s1")) >= 2:
                break
        st = r.status()
        assert len(st["ready"]) == 2 and st["buffer_target"] == 2 and st["rendering"] is None
        assert any(e["kind"] == "rejected" for e in st["events"])      # the second render failed the gate and was replaced
        first = r.next()
        assert first["id"] == "song0" and r.state == "playing" and r.now_playing["id"] == "song0"
        for _ in range(200):
            await asyncio.sleep(0.01)
            if len(store.ready("s1")) >= 2:
                break
        assert len(store.ready("s1")) == 2                             # refilled after next()
        r.stop()
        assert r.state == "stopping"
        for _ in range(200):
            await asyncio.sleep(0.01)
            if r.state == "stopped":
                break
        assert r.state == "stopped" and len(store.ready("s1")) >= 1     # the spare survives Stop
        assert r.now_playing is None and store.recent("s1")[0]["status"] in ("played", "rejected")
        r.play()
        assert r.state == "playing"                                     # spare → instant play
        assert r.next()["id"] != "song0"
        r.stop()
        await asyncio.sleep(0.05)
    asyncio.run(scenario())
    assert rendered[0] == "inspired"
    # every cued (gate-passing) song was auto-saved into the station's own playlist; the rejected one was not
    pid = json.loads(con.execute("SELECT settings FROM stations WHERE id='s1'").fetchone()["settings"])["playlist_id"]
    assert con.execute("SELECT name FROM playlists WHERE id=?", (pid,)).fetchone()["name"] == "S — radio"
    listed = [r["song_id"] for r in con.execute("SELECT song_id FROM playlist_items WHERE playlist_id=? ORDER BY position", (pid,))]
    cued = [r["id"] for r in con.execute("SELECT id FROM songs WHERE station_id='s1' AND status != 'rejected' ORDER BY created")]
    assert listed == cued and len(listed) >= 3
    assert all(r["saved"] == 1 for r in con.execute("SELECT saved FROM songs WHERE status != 'rejected'"))
    assert con.execute("SELECT saved FROM songs WHERE status='rejected'").fetchone()["saved"] == 0


def test_next_can_pick_a_specific_cued_song(tmp_path):
    con = db.connect(tmp_path)
    con.execute("INSERT INTO stations (id, name, created, profile, settings) VALUES ('s1','S',0,'{}','{}')")
    con.commit()
    store = radio.Store(con, tmp_path)
    for i in range(3):
        store.add_song("s1", {"id": f"song{i}", "title": f"t{i}", "style": "s", "lyrics": "l", "abc": None, "plan": {}, "seconds": 100.0,
                             "path": str(tmp_path / f"{i}.flac"), "explain": "", "gate": {"ok": True, "reasons": []}}, status="ready")
    r = radio.Radio("s1", store=store, renderer=None, loop_s=0.01)
    assert r.next("song1")["id"] == "song1"                      # the clicked one, not the oldest
    assert [s["id"] for s in store.ready("s1")] == ["song0", "song2"]
    assert r.next("song1") is None                                # already played → nothing happens
    assert r.next("nope") is None
    assert r.next()["id"] == "song0"                              # plain next still pops the oldest
    assert r.now_playing["id"] == "song0"
