"""vidmakr-sheetsage unit tests: pure helpers + the job lifecycle against a
fake model. No GPU, no weights, no network — runs in the built image."""
import base64
import time

import pytest
from fastapi.testclient import TestClient

import app as m

ABC = """X:1
T:
M:4/4
L:1/8
Q:1/4=92
K:Dm
V:1
% verse
|: D2 F2 A2 d2 | c4 A4 | F2 E2 D4 :|
% chorus
|: A2 d2 f2 a2 | g4 f4 |
"""


def test_abc_key_bpm_and_summary():
    assert m.abc_key_bpm(ABC) == ("D minor", 92)
    assert m.abc_key_bpm("K:G\nQ:120\n|G|") == ("G major", 120)
    assert m.abc_key_bpm("K:F#min\n") == ("F# minor", None)
    assert m.abc_key_bpm("K:Bb Mixolydian\n") == ("Bb mixolydian", None)
    assert m.abc_key_bpm(None) == (None, None)
    s = m.abc_summary(ABC)
    assert s["key"] == "D minor" and s["bpm"] == 92 and s["voices"] == 1
    assert s["bars"] == 7 and s["sections"] == ["verse", "chorus"]


def test_decode_audio_b64_caps():
    data = m.decode_audio_b64(base64.b64encode(b"x" * 100).decode())
    assert data == b"x" * 100
    for bad in ("", "   ", base64.b64encode(b"tiny").decode(), "A" * int(m.MAX_AUDIO_MB * 1024 * 1024 * 4 / 3 + 10)):
        with pytest.raises(m.BadRequest):
            m.decode_audio_b64(bad)


def test_stage_progress_spans():
    assert m.stage_progress("encoding", 0.0) == 0.15 and m.stage_progress("encoding", 1.0) == 0.9
    assert m.stage_progress("done", 0.3) == 1.0


class FakeModel:
    """Stands in for SheetSage2Model: two windows of "encoding" progress, then a score."""
    def __init__(self, abc=ABC, fail_melody=False, midi=b"MThd"):
        self.abc, self.fail_melody, self.midi = abc, fail_melody, midi
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        for w in (1, 2):
            kw["progress"]({"stage": "encoding", "window": w, "windows": 2, "start": 0.0})
            time.sleep(0.01)
        if self.fail_melody:
            err = RuntimeError("melody-only ABC unavailable")
            err.result = {"abc": None, "abc_error": "no vocal melody found", "warnings": ["Window 1: quiet"], "midi": self.midi}
            raise err
        return {"abc": self.abc, "abc_error": None, "warnings": ["Window 2: overlap prefix trimmed"], "midi": self.midi}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "SCRATCH_DIR", str(tmp_path))
    monkeypatch.setattr(m, "probe_seconds", lambda path: 61.5)
    fake = FakeModel()
    m.runner = m.Runner(factory=lambda: fake)
    m.runner.start()
    yield TestClient(m.app), fake
    m.runner.stop()


def _wait(c, job_id, timeout=5.0):
    t = time.time()
    while time.time() - t < timeout:
        d = c.get(f"/jobs/{job_id}").json()
        if d["state"] in ("done", "failed", "cancelled"):
            return d
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_transcribe_lifecycle(client):
    c, fake = client
    b64 = base64.b64encode(b"fake-flac-bytes" * 20).decode()
    r = c.post("/transcribe", json={"audio_b64": b64, "name": "Harbor Lights.flac"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    d = _wait(c, job_id)
    assert d["state"] == "done" and d["progress"] == 1.0 and d["windows"] == 2
    assert d["abc"] == ABC and d["key"] == "D minor" and d["bpm"] == 92 and d["bars"] == 7
    assert d["sections"] == ["verse", "chorus"] and d["seconds"] == 61.5 and d["has_midi"] is True
    assert d["warnings"] == ["Window 2: overlap prefix trimmed"] and d["melody_only"] is True
    assert fake.calls[0]["melody_only"] is True and fake.calls[0]["output_dir"] is None
    assert fake.calls[0]["max_seconds"] == m.MAX_SECONDS   # default cap rides along
    assert c.get(f"/jobs/{job_id}/midi").content == b"MThd"
    assert c.get("/healthz").json()["loaded"] is True
    assert c.delete(f"/jobs/{job_id}").status_code == 200
    assert c.get(f"/jobs/{job_id}").status_code == 404
    # idle unload once nothing is running
    m.runner.last_activity -= m.IDLE_UNLOAD_MIN * 60 + 1
    m.runner.sweep_once()
    assert c.get("/healthz").json()["loaded"] is False


def test_melody_only_failure_is_a_failed_job_with_partials(client, monkeypatch):
    c, fake = client
    fake.fail_melody = True
    r = c.post("/transcribe", json={"audio_b64": base64.b64encode(b"x" * 200).decode()})
    d = _wait(c, r.json()["job_id"])
    assert d["state"] == "failed" and "melody-only ABC unavailable" in d["error"]
    assert d["abc"] is None and d["warnings"] == ["Window 1: quiet"] and d["has_midi"] is True


def test_transcribe_validation(client):
    c, _ = client
    assert c.post("/transcribe", json={"audio_b64": ""}).status_code == 400
    assert c.post("/transcribe", json={"audio_b64": base64.b64encode(b"x" * 200).decode(), "max_seconds": 0}).status_code == 400
    assert c.post("/transcribe", json={"audio_b64": base64.b64encode(b"x" * 200).decode(), "max_seconds": m.MAX_SECONDS + 1}).status_code == 400
    assert c.get("/jobs/nope").status_code == 404
