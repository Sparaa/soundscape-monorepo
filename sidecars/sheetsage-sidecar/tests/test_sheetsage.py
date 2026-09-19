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
    assert m.stage_progress("encoding", 0.0) == 0.15 and m.stage_progress("encoding", 1.0) == 0.85
    assert m.stage_progress("tagging", 1.0) == 0.99
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
    m.runner = m.Runner(factory=lambda: fake, tagger=False)   # no CLAP in the plain lifecycle tests
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


# --------------------------------------------------------------------------- #
# sound tagging (2026-09-13): CLAP zero-shot → YuE2 style line
# --------------------------------------------------------------------------- #

def _scores(**best):
    """Logit vectors where the named label leads its category by a wide margin."""
    out = {}
    for cat, (_templates, labels) in m.TAG_VOCAB.items():
        vals = [0.0] * len(labels)
        for i, lbl in enumerate(labels):
            if lbl in best.get(cat, ()):
                vals[i] = 6.0 - 0.5 * list(best[cat]).index(lbl)
        out[cat] = vals
    return out


def test_pick_tags_and_compose_style():
    tags = m.pick_tags(_scores(genre=["indie folk"], mood=["wistful", "hopeful"],
                               instruments=["fingerpicked acoustic guitar", "brushed drums", "upright bass"],
                               vocal=["female vocals"], voice=["breathy soft"], production=["an intimate acoustic recording"]))
    assert tags["genre"][0][0] == "indie folk" and tags["vocal"][0][0] == "female vocals" and len(tags["vocal"]) == 1
    assert [l for l, _ in tags["instruments"]] == ["fingerpicked acoustic guitar", "brushed drums", "upright bass"]
    style = m.compose_style(tags, language="English", bpm=92)
    assert style == ("English, indie folk, wistful, hopeful, fingerpicked acoustic guitar, brushed drums, upright bass, "
                     "breathy soft female vocals, an intimate acoustic recording, 92 BPM")
    # instrumental → no voice character; no bpm → no tempo
    inst = m.pick_tags(_scores(genre=["ambient"], vocal=["instrumental music without vocals"], voice=["whispered"]))
    inst["voice"] = []
    assert m.compose_style(inst, language="Mandarin") .startswith("Mandarin, ambient") and "instrumental, no vocals" in m.compose_style(inst)
    assert "BPM" not in m.compose_style(inst)
    # a close second genre becomes "… with … touches"
    close = m.pick_tags(_scores(genre=["synthwave", "synth-pop"]))
    close["genre"] = [("synthwave", 0.5), ("synth-pop", 0.2)]
    assert m.compose_style(close).startswith("English, synthwave with synth-pop touches")
    close["genre"] = [("synthwave", 0.5), ("synth-pop", 0.1)]
    assert m.compose_style(close).startswith("English, synthwave,") or m.compose_style(close) == "English, synthwave"
    assert all(isinstance(t, tuple) and t for t, _ in m.TAG_VOCAB.values())   # every category ensembles ≥ 1 template


def test_windows_cover_the_song_evenly():
    sr = m.TAG_SAMPLE_RATE
    assert m.Tagger.windows(5 * sr) == [(0, 5 * sr)]                         # shorter than one clip
    w = m.Tagger.windows(78 * sr)
    assert len(w) == 7 and w[0][0] == 0 and w[-1][1] == 78 * sr and all(b - a == 10 * sr for a, b in w)
    assert len(m.Tagger.windows(600 * sr)) == m.TAG_MAX_WINDOWS


class FakeTagger:
    model = None

    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def describe(self, audio, *, language="English", bpm=None):
        self.calls.append((len(audio), language, bpm))
        if self.fail:
            raise RuntimeError("clap exploded")
        self.model = object()
        return ({"tags": {"genre": [{"label": "indie folk", "p": 0.7}]}, "windows": 3},
                m.compose_style({"genre": [("indie folk", 0.7)], "vocal": [("female vocals", 0.8)]}, language=language, bpm=bpm))

    def unload(self):
        self.model = None


@pytest.fixture
def tagged_client(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "SCRATCH_DIR", str(tmp_path))
    monkeypatch.setattr(m, "probe_seconds", lambda path: 61.5)
    fake, tagger = FakeModel(), FakeTagger()
    m.runner = m.Runner(factory=lambda: fake, tagger=tagger)
    m.runner.start()
    yield TestClient(m.app), tagger
    m.runner.stop()


def test_transcription_carries_the_sound_description(tagged_client):
    c, tagger = tagged_client
    r = c.post("/transcribe", json={"audio_b64": base64.b64encode(b"x" * 300).decode(), "name": "Harbor", "language": "English"})
    d = _wait(c, r.json()["job_id"])
    assert d["state"] == "done" and d["style_guess"] == "English, indie folk, female vocals, 92 BPM"
    assert d["tags"]["genre"][0]["label"] == "indie folk" and d["timing"]["tag_windows"] == 3
    assert tagger.calls == [(300, "English", 92)]          # the score's tempo feeds the style line
    assert c.get("/healthz").json()["tagger"]["enabled"] is True
    # describe=false skips it
    r = c.post("/transcribe", json={"audio_b64": base64.b64encode(b"x" * 300).decode(), "describe": False})
    d = _wait(c, r.json()["job_id"])
    assert d["style_guess"] is None and len(tagger.calls) == 1


def test_tagging_failure_is_only_a_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "SCRATCH_DIR", str(tmp_path))
    monkeypatch.setattr(m, "probe_seconds", lambda path: 1.0)
    m.runner = m.Runner(factory=lambda: FakeModel(), tagger=FakeTagger(fail=True))
    m.runner.start()
    try:
        c = TestClient(m.app)
        d = _wait(c, c.post("/transcribe", json={"audio_b64": base64.b64encode(b"x" * 300).decode()}).json()["job_id"])
        assert d["state"] == "done" and d["abc"] == ABC and d["style_guess"] is None
        assert any("sound tagging failed" in w for w in d["warnings"])
    finally:
        m.runner.stop()
