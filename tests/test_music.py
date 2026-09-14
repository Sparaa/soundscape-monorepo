"""vidmakr-music unit tests: pure helpers + the job lifecycle against a fake
pipeline. No GPU, no weights, no network — runs in the built image."""
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as m


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def test_normalize_lyrics_tags_and_blank_lines():
    raw = "Verse 1:\r\nNeon fades\n\n\n[chorus]\nLet the day\n\n"
    assert m.normalize_lyrics(raw) == "[Verse 1]\nNeon fades\n\n[Chorus]\nLet the day"


def test_normalize_lyrics_adds_verse_when_untagged():
    assert m.normalize_lyrics("just words\nmore words").startswith("[Verse]\njust words")


def test_normalize_lyrics_rejects_empty_and_huge():
    with pytest.raises(m.BadRequest):
        m.normalize_lyrics("   \n\n")
    with pytest.raises(m.BadRequest):
        m.normalize_lyrics("[Verse]\n" + "la " * m.MAX_LYRICS_CHARS)


def test_build_request_defaults_and_validation():
    req = m.build_request("  English,  pop ", "[Verse]\nhi")
    assert req["style"] == "English, pop"
    assert req["cot"] == "full" and req["id"] == "song" and "cfg_scale" not in req
    assert 0 <= req["seed"] < 2**31
    req = m.build_request("pop", "hi", cot="OFF", seed=7, cfg_scale=1.2, song_id="city_lights")
    assert req == {"style": "pop", "lyrics": "[Verse]\nhi", "cot": "off", "seed": 7,
                   "id": "city_lights", "cfg_scale": 1.2}
    for bad in (dict(cot="chords"), dict(seed=-1), dict(seed=True), dict(cfg_scale=25)):
        with pytest.raises(m.BadRequest):
            m.build_request("pop", "hi", **bad)


def test_build_request_cover_abc(monkeypatch):
    """Cover mode (2026-09-13): an external score rides as `abc`; YuE2 then
    tokenizes it instead of planning one, so cot=off is refused."""
    abc = "X:1\r\nM:4/4\r\nL:1/8\r\nK:G\r\n|: G2 A2 B2 d2 | e4 d4 :|\r\n"
    req = m.build_request("pop", "hi", cot="melody", seed=1, abc=abc)
    assert req["abc"] == "X:1\nM:4/4\nL:1/8\nK:G\n|: G2 A2 B2 d2 | e4 d4 :|\n"
    assert "abc" not in m.build_request("pop", "hi", cot="melody", seed=1, abc="   ")
    for bad_abc, cot in ((abc, "off"), ("X:1\nM:4/4\n|: G2 |", "melody"), ("K:G\n", "full"), ("K:G\n" + "|G|" * 40000, "full")):
        with pytest.raises(m.BadRequest):
            m.build_request("pop", "hi", cot=cot, seed=1, abc=bad_abc)


CITY_ABC = """X:1
T:
M:4/4
L:1/32
Q:1/4=83
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% intro
V: Vocal
z24z4"C"z4|"C"z32|"Em"z32|"Fmaj7"z32|
V: Ins
z24z6E2|E4E4B4E4E4B4E4B4|E4E4B4E4E4A4G4F4|E4E4B4E4E4B4E4B4|
% verse
V: Vocal
"C"z12B,4G4B8B4-|"C"B4G8F8E8G4-|"Am"G4z8B,4G4B2B6B4-|"F"B8G4F8E8C4-|
V: Ins
Z4|
"""


def test_abc_seconds_reads_meter_tempo_and_vocal_bars():
    # 8 Vocal bars at 4/4, 83 BPM → 8 × 4 × 60/83 = 23.1 s; the Ins voice (5 bars) is ignored
    assert m.abc_seconds(CITY_ABC) == 23.1
    assert m.abc_seconds(CITY_ABC.replace("Q:1/4=83", "Q:83")) == 23.1
    assert m.abc_seconds(CITY_ABC.replace("M:4/4", "M:3/4")) == 17.3
    assert m.abc_seconds(CITY_ABC.replace("Q:1/4=83", "Q:1/8=166")) == 23.1
    assert m.abc_seconds(None) is None and m.abc_seconds("") is None
    assert m.abc_seconds("X:1\nK:C\nCDEF|") is None          # no headers
    assert m.abc_seconds(CITY_ABC.replace("Q:1/4=83", "Q:fast")) is None


def test_expected_semantic_tokens_from_score_or_lyrics():
    # 23.1 s × 25 tok/s × 1.05 ≈ 606
    assert m.expected_semantic_tokens(CITY_ABC, "ignored") == 606
    # no score: per-line guess, headers excluded, floor at 4 lines
    assert m.expected_semantic_tokens(None, "[Verse]\na\nb\n\n[Chorus]\nc") == 4 * m.TOKENS_PER_LYRIC_LINE
    assert m.expected_semantic_tokens(None, "\n".join(["la"] * 200)) == m.SEMANTIC_MAX_TOKENS
    assert m.expected_semantic_tokens("X:1\nK:C\n", "") == 4 * m.TOKENS_PER_LYRIC_LINE


def test_stage_progress_is_monotonic_across_stages():
    order = ["loading", "plan", "semantic", "synthesize", "decode", "saving", "done"]
    last = -1.0
    for stage in order:
        for frac in (0.0, 0.5, 1.0):
            p = m.stage_progress(stage, frac)
            assert p >= last
            last = p
    assert m.stage_progress("semantic", 5.0) == m.stage_progress("semantic", 1.0)
    assert m.stage_progress("done", 1.0) == 1.0


def test_timed_fraction_never_reaches_one():
    assert m.timed_fraction(0, 20) == 0.0
    assert 0.4 < m.timed_fraction(10, 20) < 0.6
    assert m.timed_fraction(200, 20) <= 0.95
    assert m.timed_fraction(5, 0) == 0.0


def test_learn_seconds_ema():
    m._STAGE_SECONDS["x"] = 10.0
    assert m.learn_seconds("x", 20.0) == 15.0
    assert m.learn_seconds("x", 15.0) == 15.0


def test_ffmpeg_cmd_formats(tmp_path):
    src, dst = tmp_path / "a.flac", tmp_path / "a.mp3"
    cmd = m.ffmpeg_cmd(src, dst, "mp3")
    assert cmd[0] == "ffmpeg" and "libmp3lame" in cmd and cmd[-1] == str(dst)
    assert "aac" in m.ffmpeg_cmd(src, tmp_path / "a.m4a", "m4a")
    with pytest.raises(m.BadRequest):
        m.ffmpeg_cmd(src, dst, "ogg")


# --------------------------------------------------------------------------- #
# fake pipeline
# --------------------------------------------------------------------------- #

class _Plan:
    def __init__(self, request, abc):
        self.request, self.abc, self.truncated = request, abc, False


class _Semantic:
    def __init__(self, tokens):
        self.tokens, self.truncated = tokens, False


class FakePipe:
    """Mimics the four staged YuE2Pipeline calls, emitting token callbacks and
    honouring `cancelled`. `delay` slows each stage so cancel tests can land."""
    instances: list["FakePipe"] = []

    def __init__(self, delay=0.0, fail_at=None, seconds=2.0, abc="X:1\nK:C\nCDEF|"):
        self.delay, self.fail_at, self.seconds, self.abc = delay, fail_at, seconds, abc
        self.closed = False
        FakePipe.instances.append(self)

    def _tick(self, cancelled, n, on_token):
        for i in range(n):
            if cancelled and cancelled():
                raise InterruptedError("cancelled")
            if on_token:
                on_token("x", i)
            if self.delay:
                time.sleep(self.delay)

    def plan(self, request, cancelled=None, on_token=None, **_):
        if self.fail_at == "plan":
            raise RuntimeError("boom")
        self._tick(cancelled, 8, on_token)
        return _Plan(request, None if request.cot == "off" else self.abc)

    def generate_semantic(self, plan, cancelled=None, on_token=None, **_):
        if self.fail_at == "semantic":
            raise RuntimeError("boom")
        self._tick(cancelled, 20, on_token)
        return _Semantic(list(range(20)))

    def synthesize(self, semantic, cancelled=None):
        self._tick(cancelled, 3, None)
        return np.zeros((int(self.seconds * 48000 / 1920), 64), dtype=np.float32)

    def decode(self, latents):
        n = int(self.seconds * 48000)
        t = np.arange(n) / 48000
        tone = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return np.stack([tone, tone], axis=1)

    def close(self):
        self.closed = True


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "SCRATCH_DIR", str(tmp_path))
    FakePipe.instances.clear()
    runner = m.Runner(factory=lambda: FakePipe(delay=0.005))
    monkeypatch.setattr(m, "runner", runner)
    with TestClient(m.app) as c:
        c.runner = runner
        yield c
    runner.stop()


def _wait(client, job_id, states=("done", "failed", "cancelled"), timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        js = client.get(f"/jobs/{job_id}").json()
        if js["state"] in states:
            return js
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not settle: {js}")


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #

def test_healthz_reports_unloaded(client):
    js = client.get("/healthz").json()
    assert js["ok"] is True and js["loaded"] is False and js["queue"] == 0
    assert js["model"] == m.MODEL_ID and "gpu" in js


def test_generate_validates(client):
    r = client.post("/generate", json={"style": "", "lyrics": "hi"})
    assert r.status_code == 400 and "style" in r.json()["detail"]
    r = client.post("/generate", json={"style": "pop", "lyrics": "hi", "cot": "nope"})
    assert r.status_code == 400
    r = client.post("/generate", json={"style": "pop", "lyrics": "hi", "id": "../x"})
    assert r.status_code == 422


def test_full_lifecycle_flac_mp3_score_delete(client, tmp_path):
    r = client.post("/generate", json={"style": "English, pop", "lyrics": "Verse:\nla la", "seed": 5})
    assert r.status_code == 200
    js = r.json()
    job_id = js["job_id"]
    assert js["state"] == "queued" and js["position"] == 0 and js["seed"] == 5

    done = _wait(client, job_id)
    assert done["state"] == "done" and done["progress"] == 1.0 and done["stage"] == "done"
    assert done["audio_seconds"] == 2.0 and done["truncated"] == {"abc": False, "semantic": False}
    assert done["has_score"] is True and "lyrics" not in done["request"]
    assert done["timing"]["semantic_tokens"] == 20
    assert set(done["timing"]) >= {"plan_seconds", "semantic_seconds", "synthesize_seconds", "decode_seconds"}
    assert client.get("/healthz").json()["loaded"] is True

    # master
    r = client.get(f"/jobs/{job_id}/audio")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/flac"
    assert r.content[:4] == b"fLaC" and "attachment" in r.headers["content-disposition"]
    # delivery conversion via ffmpeg
    r = client.get(f"/jobs/{job_id}/audio", params={"format": "mp3"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and len(r.content) > 1000
    assert (Path(tmp_path) / job_id / "audio.mp3").is_file()
    assert client.get(f"/jobs/{job_id}/audio", params={"format": "ogg"}).status_code == 400
    # score
    r = client.get(f"/jobs/{job_id}/score")
    assert r.status_code == 200 and r.text.startswith("X:1")

    r = client.delete(f"/jobs/{job_id}")
    assert r.status_code == 200
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert not (Path(tmp_path) / job_id).exists()


def test_planned_seconds_and_semantic_target_from_score(client):
    client.runner.factory = lambda: FakePipe(abc=CITY_ABC)
    job_id = client.post("/generate", json={"style": "pop", "lyrics": "hi"}).json()["job_id"]
    done = _wait(client, job_id)
    assert done["planned_seconds"] == 23.1
    assert done["timing"]["semantic_target_tokens"] == 606
    assert client.get(f"/jobs/{job_id}/score").text == CITY_ABC


def test_semantic_progress_caps_below_stage_end(client):
    # a 1-line lyric with the toy score → target = 4 × 140 = 560 tokens; the fake emits 20,
    # so semantic progress must move but stay inside the semantic span
    job_id = client.post("/generate", json={"style": "pop", "lyrics": "hi", "cot": "off"}).json()["job_id"]
    done = _wait(client, job_id)
    assert done["timing"]["semantic_target_tokens"] == 560
    lo, hi = m.STAGE_SPAN["semantic"]
    assert lo < m.stage_progress("semantic", min(20 / 560, 0.98)) < hi
    assert m.stage_progress("semantic", min(5000 / 560, 0.98)) < hi


def test_cot_off_has_no_score(client):
    job_id = client.post("/generate", json={"style": "pop", "lyrics": "hi", "cot": "off"}).json()["job_id"]
    done = _wait(client, job_id)
    assert done["has_score"] is False
    assert client.get(f"/jobs/{job_id}/score").status_code == 404


def test_audio_before_done_is_409(client, monkeypatch):
    client.runner.factory = lambda: FakePipe(delay=0.05)
    job_id = client.post("/generate", json={"style": "pop", "lyrics": "hi"}).json()["job_id"]
    _wait(client, job_id, states=("running",))
    assert client.get(f"/jobs/{job_id}/audio").status_code == 409
    _wait(client, job_id)


def test_jobs_run_serially_and_progress_moves(client):
    client.runner.factory = lambda: FakePipe(delay=0.02)
    a = client.post("/generate", json={"style": "pop", "lyrics": "a"}).json()
    b = client.post("/generate", json={"style": "pop", "lyrics": "b"}).json()
    assert a["position"] == 0
    assert b["position"] == 1  # a is running or still queued: either way one job ahead
    _wait(client, a["job_id"], states=("running",))
    seen = set()
    for _ in range(40):
        js = client.get(f"/jobs/{a['job_id']}").json()
        seen.add((js["stage"], js["progress"]))
        if js["state"] == "done":
            break
        time.sleep(0.02)
    assert len({p for _, p in seen}) > 3
    assert client.get(f"/jobs/{b['job_id']}").json()["state"] in ("queued", "running", "done")
    _wait(client, b["job_id"])
    ja, jb = (client.get(f"/jobs/{x['job_id']}").json() for x in (a, b))
    assert ja["finished_at"] <= jb["started_at"]
    assert len(FakePipe.instances) == 1  # one load serves both


def test_cancel_running_job(client, tmp_path):
    client.runner.factory = lambda: FakePipe(delay=0.05)
    job_id = client.post("/generate", json={"style": "pop", "lyrics": "hi"}).json()["job_id"]
    _wait(client, job_id, states=("running",))
    assert client.delete(f"/jobs/{job_id}").status_code == 200
    js = _wait(client, job_id, states=("cancelled",))
    assert js["state"] == "cancelled"
    assert not (Path(tmp_path) / job_id).exists()


def test_cancel_queued_job(client):
    client.runner.factory = lambda: FakePipe(delay=0.05)
    a = client.post("/generate", json={"style": "pop", "lyrics": "a"}).json()["job_id"]
    b = client.post("/generate", json={"style": "pop", "lyrics": "b"}).json()["job_id"]
    assert client.delete(f"/jobs/{b}").status_code == 200
    # a queued job has nothing to keep: it is gone, not merely marked cancelled
    assert client.get(f"/jobs/{b}").status_code == 404
    done = _wait(client, a)
    assert done["state"] == "done"
    assert [j["job_id"] for j in client.get("/jobs").json()["jobs"]] == [a]


def test_failure_is_reported(client):
    client.runner.factory = lambda: FakePipe(fail_at="semantic")
    job_id = client.post("/generate", json={"style": "pop", "lyrics": "hi"}).json()["job_id"]
    js = _wait(client, job_id)
    assert js["state"] == "failed" and js["error"] == "RuntimeError: boom"
    assert client.get(f"/jobs/{job_id}/audio").status_code == 409


def test_queue_limit(client, monkeypatch):
    monkeypatch.setattr(m, "MAX_QUEUE", 2)
    client.runner.factory = lambda: FakePipe(delay=0.05)
    ids = [client.post("/generate", json={"style": "pop", "lyrics": str(i)}).json()["job_id"] for i in range(2)]
    _wait(client, ids[0], states=("running",))
    client.post("/generate", json={"style": "pop", "lyrics": "x"})  # 2 pending now (ids[1] + this)
    r = client.post("/generate", json={"style": "pop", "lyrics": "y"})
    assert r.status_code == 400 and "queue is full" in r.json()["detail"]
    for j in client.get("/jobs").json()["jobs"]:
        _wait(client, j["job_id"])


def test_idle_unload_and_retention(client, tmp_path):
    job_id = client.post("/generate", json={"style": "pop", "lyrics": "hi"}).json()["job_id"]
    _wait(client, job_id)
    runner = client.runner
    assert runner.pipe is not None
    # not idle yet
    runner.sweep_once(now=time.time())
    assert runner.pipe is not None and job_id in runner.jobs
    # long past idle + retention
    later = time.time() + max(m.IDLE_UNLOAD_MIN, m.JOB_RETAIN_MIN) * 60 + 1
    runner.sweep_once(now=later)
    assert runner.pipe is None and FakePipe.instances[-1].closed is True
    assert job_id not in runner.jobs and not (Path(tmp_path) / job_id).exists()
    # manual unload is a no-op when already unloaded
    assert client.post("/unload").json() == {"unloaded": False, "loaded": False}


def test_unload_refused_while_busy(client):
    client.runner.factory = lambda: FakePipe(delay=0.05)
    job_id = client.post("/generate", json={"style": "pop", "lyrics": "hi"}).json()["job_id"]
    _wait(client, job_id, states=("running",))
    assert client.post("/unload").json()["unloaded"] is False
    _wait(client, job_id)
    assert client.post("/unload").json() == {"unloaded": True, "loaded": False}


# --------------------------------------------------------------------------- #
# long songs (2026-09-13): chunking + stitching
# --------------------------------------------------------------------------- #

LONG_LYRICS = "[Verse]\na1\na2\na3\n\n[Chorus]\nc1\nc2\n\n[Verse 2]\nb1\nb2\nb3\n\n[Chorus]\nc1\nc2\n\n[Bridge]\nd1\nd2\n\n[Outro]\ne1"


def test_split_sections_and_plan_chunks():
    secs = m.split_sections(m.normalize_lyrics(LONG_LYRICS))
    assert [s.splitlines()[0] for s in secs] == ["[Verse]", "[Chorus]", "[Verse 2]", "[Chorus]", "[Bridge]", "[Outro]"]
    assert [m.section_lines(s) for s in secs] == [3, 2, 3, 2, 2, 1]
    # 6 lines per chunk: V(3)+C(2)=5 | V2(3)+C(2)=5 | B(2)+O(1)=3
    chunks = m.plan_chunks(secs, 6)
    assert [c.count("\n[") + 1 for c in chunks] == [2, 2, 2] and chunks[0].startswith("[Verse]") and chunks[2].startswith("[Bridge]")
    # overlap: the seam repeats the previous chunk's chorus when it fits
    chunks = m.plan_chunks(secs, 7, overlap_chorus=True)
    assert chunks[1].startswith("[Chorus]\nc1\nc2\n\n[Verse 2]")
    # one oversized section still becomes its own chunk
    assert len(m.plan_chunks(["[Verse]\n" + "\n".join(["x"] * 40)], 6)) == 1
    # untagged trailing block belongs to the previous section
    assert m.split_sections("[Verse]\na\n\nb\nc") == ["[Verse]\na\nb\nc"]


def test_abc_key_bpm_and_style_with_key():
    assert m.abc_key_bpm("X:1\nQ:1/4=92\nK:Dm\n|D|") == ("D minor", 92)
    assert m.abc_key_bpm("K:F#\n") == ("F# major", None)
    assert m.style_with_key("English, indie folk", "K:F\nQ:1/4=92\n") == "English, indie folk, in the key of F major, 92 BPM"
    assert m.style_with_key("English, pop, 120 BPM", "K:C\nQ:1/4=92\n") == "English, pop, 120 BPM, in the key of C major"
    assert m.style_with_key("pop, in the key of F major", "K:F\n") == "pop, in the key of F major"
    assert m.style_with_key("pop", None) == "pop"


def test_equal_power_crossfade_and_trims():
    a = np.ones((100, 2), dtype=np.float32)
    b = np.ones((100, 2), dtype=np.float32) * 3
    out = m.equal_power_crossfade(a, b, 20)
    assert out.shape == (180, 2)
    assert out[79, 0] == 1.0 and out[100, 0] == 3.0                    # untouched outside the seam
    t = 10 / 19                                                        # sample 10 of a 20-point ramp
    mid = out[80 + 10, 0]
    assert abs(mid - (np.cos(t * np.pi / 2) * 1 + np.sin(t * np.pi / 2) * 3)) < 1e-4
    gains = np.cos(np.linspace(0, 1, 20) * np.pi / 2) ** 2 + np.sin(np.linspace(0, 1, 20) * np.pi / 2) ** 2
    assert np.allclose(gains, 1.0)                                     # equal power across the seam
    assert m.equal_power_crossfade(a, b, 0).shape == (200, 2)
    assert m.equal_power_crossfade(a, b, 500).shape == (100, 2)        # overlap capped at the shorter chunk
    x = np.arange(48000 * 3, dtype=np.float32).reshape(-1, 1)
    assert len(m.trim_edges(x, 48000, 0.5, 1.0)) == 48000 * 3 - 72000
    assert len(m.trim_edges(x[:48000], 48000, 0.5, 0.5)) == 48000      # never below one second
    chunks = [np.ones((48000 * 2, 2), np.float32)] * 3
    stitched = m.stitch_chunks(chunks, 48000, 500, 0.25, 0.25)
    # inner trims: 0.25 s off chunk0's end, both ends of chunk1, chunk2's start = 1.0 s; two 0.5 s crossfades
    assert len(stitched) == 48000 * (6 - 1.0 - 1.0)


def test_normalize_long_and_build_request():
    assert m.normalize_long(None) is None and m.normalize_long(False) is None and m.normalize_long({"enabled": False}) is None
    assert m.normalize_long(True) == {"max_lines_per_chunk": m.LONG_MAX_LINES, "overlap_chorus": False,
                                      "crossfade_ms": 4000.0, "trim_start_s": 1.5, "trim_end_s": 2.5}
    assert m.normalize_long({"max_lines_per_chunk": 10, "overlap_chorus": True, "crossfade_ms": 2000})["max_lines_per_chunk"] == 10
    for bad in ({"max_lines_per_chunk": 2}, {"crossfade_ms": 99999}, {"trim_end_s": 99}, "yes"):
        with pytest.raises(m.BadRequest):
            m.normalize_long(bad)
    req = m.build_request("pop", "hi", seed=1, long=True)
    assert req["long"]["crossfade_ms"] == 4000.0
    with pytest.raises(m.BadRequest):   # cover + long is v2
        m.build_request("pop", "hi", cot="melody", seed=1, abc="K:C\n|C|", long=True)


def test_long_lifecycle_renders_chunks_and_stitches(client, tmp_path):
    r = client.post("/generate", json={"style": "English, indie folk", "lyrics": LONG_LYRICS, "seed": 5,
                                       "long": {"max_lines_per_chunk": 6, "crossfade_ms": 500, "trim_start_s": 0.25, "trim_end_s": 0.25}})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    done = _wait(client, job_id, timeout=20.0)
    assert done["state"] == "done" and done["long"] is True and done["cover"] is False
    assert [c["index"] for c in done["chunks"]] == [1, 2, 3] and [c["seed"] for c in done["chunks"]] == [5, 6, 7]
    assert [c["lines"] for c in done["chunks"]] == [5, 5, 3]
    # FakePipe returns 2.0 s per chunk; three chunks, inner trims 1.0 s total, two 0.5 s crossfades → 4.0 s
    assert done["audio_seconds"] == 4.0
    # chunk 2+ carry the first plan's key (FakePipe's ABC is K:C, no Q:)
    assert done["chunks"][0]["style"] == "English, indie folk"
    assert done["chunks"][1]["style"] == "English, indie folk, in the key of C major"
    score = client.get(f"/jobs/{job_id}/score").text
    assert score.startswith("% chunk 1\nX:1") and "% chunk 3" in score
    assert done["progress"] == 1.0 and done["timing"]["semantic_tokens"] == 60   # 3 × 20 fake tokens
    r = client.get(f"/jobs/{job_id}/audio")
    assert r.status_code == 200 and r.content[:4] == b"fLaC"


def test_long_progress_stays_within_chunk_spans(client):
    r = client.post("/generate", json={"style": "pop", "lyrics": LONG_LYRICS, "seed": 1, "long": {"max_lines_per_chunk": 6}})
    job_id = r.json()["job_id"]
    seen = []
    deadline = time.time() + 20
    while time.time() < deadline:
        js = client.get(f"/jobs/{job_id}").json()
        seen.append(js["progress"])
        if js["state"] in ("done", "failed"):
            break
        time.sleep(0.01)
    assert seen == sorted(seen), "progress must never go backwards across chunks"
    assert seen[-1] == 1.0


# --------------------------------------------------------------------------- #
# cover riffs (2026-09-13): ABC sections + hook splicing
# --------------------------------------------------------------------------- #

PLAN_44 = """X:1
T:
M:4/4
L:1/32
Q:1/4=92
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:F
% verse
V: Vocal
"F"z8c4c2f2f4e2d4c4A2|"Dm"A6G4z6z16|
V: Ins
Z2|
% chorus
V: Vocal
"F"PLAN-CHORUS-VOCAL|"Bb"z32|
V: Ins
PLAN-CHORUS-INS|Z|
% outro
V: Vocal
"F"z32|
V: Ins
Z|
"""

HOOK_24 = """X:1
T:
M:2/4
L:1/32
Q:1/4=92
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:F
% intro
V: Vocal
Z4|
V: Ins
Z|F4A4z2F2A2f2|Z2|
% chorus
V: Vocal
"F"g12a2g2-|"Bb"g6f6e4|d8c8|B8z8|Z2|
V: Ins
Z4|
"""


def test_parse_render_roundtrip_and_strip_chords():
    h, secs = m.parse_abc(PLAN_44)
    assert h[0] == "X:1" and h[-1] == "K:F" and any(x.startswith("V: Vocal clef=") for x in h)
    assert [label for label, _ in secs] == ["verse", "chorus", "outro"]
    assert m.render_abc(h, secs) == PLAN_44
    assert m.abc_header_value(h, "M") == "4/4" and m.abc_header_value(h, "Q") == "1/4=92"
    stripped = m.strip_chords(PLAN_44)
    assert '"F"' not in stripped.split("K:F")[1] and 'name="Vocal Melody"' in stripped   # voice names survive


def test_rebar_merges_2_4_bars_into_4_4():
    lines = m.rebar_lines(['g12a2g2-|g6f6e4|d8c8|B8z8|Z2|', "Z4|", "V: Ins"], factor=2, units_per_bar=16)
    assert lines == ["g12a2g2-g6f6e4|d8c8B8z8|Z|", "Z|Z|", "V: Ins"]
    # an odd bar count pads the last merged bar with a half-bar rest
    assert m.rebar_lines(["A16|B16|C16|"], 2, 16) == ["A16B16|C16z16|"]
    assert m.rebar_lines(["A16|B16|"], 1, 16) == ["A16|B16|"]


def test_splice_hook_replaces_chorus_and_rebars():
    abc, info = m.splice_hook(PLAN_44, HOOK_24)
    assert info["spliced"] is True and info["replaced"] == 1 and info["rebar"] == 2 and info["chords_stripped"] is False
    h, secs = m.parse_abc(abc)
    assert [label for label, _ in secs] == ["verse", "chorus", "outro"]
    chorus = dict(secs)["chorus"]
    assert "PLAN-CHORUS" not in "".join(chorus)
    assert chorus[1] == '"F"g12a2g2-"Bb"g6f6e4|d8c8B8z8|Z|'      # hook bars, 2/4 pairs merged into 4/4, chords kept
    assert dict(secs)["verse"] == dict(m.parse_abc(PLAN_44)[1])["verse"]   # verses untouched
    # a melody-only plan drops the hook's chords
    abc2, info2 = m.splice_hook(m.strip_chords(PLAN_44), HOOK_24)
    assert info2["chords_stripped"] is True and '"Bb"' not in abc2.split("K:F")[1]


def test_key_parsing_signatures_and_hook_shift():
    assert m.parse_key("F") == (5, False, False) and m.parse_key("D#m") == (3, True, False) and m.parse_key("Bb") == (10, False, True)
    assert m.key_signature(5, False, False) == {"C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "A": 0, "B": -1}      # F major: Bb
    assert m.key_signature(6, False, False)["F"] == 1 and m.key_signature(6, False, False)["E"] == 1        # F# major: 6 sharps
    assert m.key_signature(3, True, False) == m.key_signature(6, False, False)                               # D#m = relative of F#
    assert m.hook_shift("F", "F") == 0 and m.hook_shift("G", "F") == 2 and m.hook_shift("C", "F") == -5
    assert m.hook_shift("D#m", "F") == 1        # major hook over a minor plan → the plan's relative major (F#)
    assert m.hook_shift("F", "Dm") == 0         # minor hook over a major plan → its relative minor (Dm) — already there
    assert m.hook_shift("Am", "Dm") == -5


def test_transpose_lines_respells_under_the_new_signature():
    # In K:F a bare B is Bb; +1 semitone into K:F# it becomes B natural = bare B there. c → c# = bare c in F#.
    out = m.transpose_lines(['"F"c2B2A2|"Bb"F4z4|', "V: Vocal", "Z4|"], 1, "F", "F#")
    assert out == ['"F#"c2B2A2|"B"F4z4|', "V: Vocal", "Z4|"]
    # Down a fifth into C: F→C, bare B(b)→F, A→E, "Bb"→"F"; rests/durations untouched
    assert m.transpose_lines(['"F"c2B2A2|"Bb"F4z4|'], -5, "F", "C") == ['"C"G2F2E2|"F"C4z4|']
    # Explicit accidentals and octave marks survive: ^c (C#5) +2 → D#5 = bare d in K:E (D# in its signature);
    # C,3 +2 → D natural, which E major must write as =D,
    assert m.transpose_lines(["^c2C,2"], 2, "C", "E") == ["d2=D,2"]
    assert m._prefers_flats(m.parse_key("D#m")) is False and m._prefers_flats(m.parse_key("Gm")) is True and m._prefers_flats(m.parse_key("Bb")) is True
    assert m.transpose_lines(["c2"], 0, "F", "F") == ["c2"]


def test_splice_hook_transposes_a_hook_in_another_key():
    abc, info = m.splice_hook(PLAN_44.replace("K:F", "K:D#m"), HOOK_24)
    assert info["spliced"] is True and info["transposed"] == 1 and info["plan_key"] == "D#m" and info["hook_key"] == "F"
    chorus = dict(m.parse_abc(abc)[1])["chorus"]
    assert chorus[1].startswith('"F#"') and '"B"' in chorus[1]            # F→F#, Bb→B, hook bars re-barred 2/4→4/4


def test_rescale_durations_between_note_units():
    from fractions import Fraction
    # L:1/32 hook into an L:1/16 plan: halve every duration; odd ones become n/2, 1 becomes /2
    out = m.rescale_durations(['"F"g12a2g2-|d8c8|B8z8|Z2|', "A1B3c|", "V: Vocal"], Fraction(1, 2))
    assert out == ['"F"g6ag-|d4c4|B4z4|Z2|', "A/2B3/2c/2|", "V: Vocal"]
    assert m.rescale_durations(["A/2B3/2|"], Fraction(2)) == ["AB3|"]
    assert m.rescale_durations(["A8|"], Fraction(1)) == ["A8|"]


def test_splice_hook_rescales_units_and_transposes_together():
    plan = PLAN_44.replace("L:1/32", "L:1/16").replace("K:F", "K:D#m")
    abc, info = m.splice_hook(plan, HOOK_24)
    assert info["spliced"] is True and info["unit_rescaled"] == "1/2" and info["transposed"] == 1 and info["rebar"] == 2
    chorus = dict(m.parse_abc(abc)[1])["chorus"]
    # F→F# (Bb→B), 32nd durations halved, 2/4 pairs merged into 4/4 bars of 8 plan units
    assert chorus[1] == '"F#"g6ag-"B"g3f3e2|d4c4B4z4|Z|'


def test_splice_hook_refuses_mismatches():
    _, info = m.splice_hook(PLAN_44, HOOK_24.replace("K:F", "K:Xq"))
    assert info["spliced"] is False and "unreadable key" in info["reason"]
    _, info = m.splice_hook(PLAN_44.replace("M:4/4", "M:2/4"), HOOK_24.replace("M:2/4", "M:4/4"))
    assert "meter mismatch" in info["reason"]
    _, info = m.splice_hook(PLAN_44, HOOK_24.replace("% chorus", "% bridge"))
    assert "no chorus section" in info["reason"]
    _, info = m.splice_hook(PLAN_44.replace("% chorus", "% verse 2"), HOOK_24)
    assert "planned song has no chorus" in info["reason"]
    _, info = m.splice_hook(PLAN_44, HOOK_24.replace("L:1/32", "L:x"))
    assert "unreadable note unit" in info["reason"]


def test_build_request_hook_validation():
    req = m.build_request("pop", "[Chorus]\nhey", seed=1, hook_abc=HOOK_24, hook_sections=["Chorus", ""])
    assert req["hook"] == {"abc": HOOK_24, "sections": ["chorus"]} and "abc" not in req
    assert m.build_request("pop", "hi", seed=1, hook_abc=HOOK_24)["hook"]["sections"] == ["chorus"]
    for kw in (dict(cot="off"), dict(abc="K:F\n|F|", cot="melody"), dict(long=True)):
        with pytest.raises(m.BadRequest):
            m.build_request("pop", "hi", seed=1, hook_abc=HOOK_24, **kw)


def test_hook_lifecycle_splices_then_replans(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "SCRATCH_DIR", str(tmp_path))
    pipe = FakePipe(abc=PLAN_44)
    plans = []
    real_plan = pipe.plan

    def plan(request, cancelled=None, on_token=None, **_):
        plans.append(request.abc)
        if request.abc is not None:
            return _Plan(request, request.abc)   # a provided score is tokenized, not sampled
        return real_plan(request, cancelled=cancelled, on_token=on_token)

    pipe.plan = plan
    runner = m.Runner(factory=lambda: pipe)
    monkeypatch.setattr(m, "runner", runner)
    with TestClient(m.app) as c:
        r = c.post("/generate", json={"style": "English, pop", "lyrics": "[Verse]\nla\n\n[Chorus]\nhey", "seed": 5,
                                      "hook_abc": HOOK_24, "hook_sections": ["chorus"]})
        assert r.status_code == 200
        done = _wait(c, r.json()["job_id"])
        assert done["state"] == "done" and done["hook"]["spliced"] is True and done["hook"]["rebar"] == 2
        assert plans[0] is None and plans[1] is not None                     # fresh plan, then the spliced score
        score = c.get(f"/jobs/{done['job_id']}/score").text
        assert "g12a2g2-" in score and "PLAN-CHORUS" not in score and "% verse" in score
    runner.stop()
