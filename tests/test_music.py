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
