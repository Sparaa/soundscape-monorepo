import pytest

import app as cg


def test_validate_url_accepts_public_https():
    assert cg.validate_url(" https://www.youtube.com/watch?v=abc ") == "https://www.youtube.com/watch?v=abc"


@pytest.mark.parametrize("bad", [
    "", "ftp://example.com/x", "file:///etc/passwd", "http://localhost/x",
    "http://127.0.0.1:3011/media", "http://10.0.0.5/", "http://172.19.0.1:18000/",
    "http://192.168.244.225:5002/", "http://[::1]/", "http://api/media", "http://foo.local/",
])
def test_validate_url_rejects_local_and_odd(bad):
    with pytest.raises(cg.BadUrl):
        cg.validate_url(bad)


def test_format_selector_prefers_avc1_then_falls_back():
    sel = cg.format_selector(720)
    parts = sel.split("/")
    assert parts[0].startswith("bv*[vcodec^=avc1][height<=720]+ba[acodec^=mp4a]")
    assert parts[-1] == "b"


def test_section_arg():
    assert cg.section_arg(None, None) is None
    assert cg.section_arg(12, 20) == "*12.000-20.000"
    assert cg.section_arg(None, 5) == "*0.000-5.000"
    assert cg.section_arg(7.5, None) == "*7.500-inf"


def test_check_range_limits():
    cg.check_range(0, 10, 100)
    cg.check_range(None, None, 30)
    with pytest.raises(ValueError):
        cg.check_range(20, 10, 100)
    with pytest.raises(ValueError):
        cg.check_range(150, None, 100)
    with pytest.raises(ValueError):
        cg.check_range(None, None, cg.MAX_CLIP_SEC + 1)
    cg.check_range(5, 5 + cg.MAX_CLIP_SEC, None)


def test_build_clip_cmd_whole_and_window():
    whole = cg.build_clip_cmd("https://x.example/v", "/scratch/a", None, None)
    assert whole[0] == "yt-dlp" and whole[-1] == "https://x.example/v"
    assert "--download-sections" not in whole
    assert "--merge-output-format" in whole and "mp4" in whole
    win = cg.build_clip_cmd("https://x.example/v", "/scratch/a", 3, 9)
    i = win.index("--download-sections")
    assert win[i + 1] == "*3.000-9.000"
    assert "--force-keyframes-at-cuts" in win


def test_parse_info_trims_and_unwraps_playlist():
    raw = {"title": "T", "duration": 42, "uploader": "U", "thumbnail": "http://t", "extractor_key": "Youtube",
           "webpage_url": "https://yt/x", "width": 1920, "height": 1080}
    got = cg.parse_info(raw)
    assert got == {"title": "T", "durationSec": 42, "uploader": "U", "thumbnail": "http://t",
                   "extractor": "Youtube", "webpageUrl": "https://yt/x", "width": 1920, "height": 1080,
                   "isLive": False, "thumbnailDataUrl": None}
    wrapped = cg.parse_info({"_type": "playlist", "entries": [raw]})
    assert wrapped["title"] == "T"


def test_find_output_prefers_mp4(tmp_path):
    (tmp_path / "clip.webm").write_bytes(b"x" * 10)
    assert cg.find_output(str(tmp_path)).name == "clip.webm"
    (tmp_path / "clip.mp4").write_bytes(b"x")
    assert cg.find_output(str(tmp_path)).name == "clip.mp4"
    assert cg.find_output(str(tmp_path / "nope")) is None if (tmp_path / "nope").mkdir() is None else True


def test_thumbnail_data_url_refuses_private_and_bad(monkeypatch):
    assert cg.thumbnail_data_url("http://127.0.0.1/x.jpg") is None
    assert cg.thumbnail_data_url("not a url") is None

    class Resp:
        headers = {"Content-Type": "image/jpeg"}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n): return b"\xff\xd8jpegbytes"
    monkeypatch.setattr(cg.urllib.request, "urlopen", lambda req, timeout: Resp())
    got = cg.thumbnail_data_url("https://i.ytimg.com/vi/x/hq.jpg")
    assert got.startswith("data:image/jpeg;base64,")

    class Html(Resp):
        headers = {"Content-Type": "text/html"}
    monkeypatch.setattr(cg.urllib.request, "urlopen", lambda req, timeout: Html())
    assert cg.thumbnail_data_url("https://i.ytimg.com/vi/x/hq.jpg") is None


def test_build_clip_cmd_audio_only():
    cmd = cg.build_clip_cmd("https://x.example/v", "/scratch/a", 3, 9, audio_only=True)
    assert "-x" in cmd and cmd[cmd.index("--audio-format") + 1] == cg.AUDIO_FORMAT
    assert cmd[cmd.index("-f") + 1] == "bestaudio/b"
    assert "--merge-output-format" not in cmd and "--remux-video" not in cmd
    assert cmd[cmd.index("--download-sections") + 1] == "*3.000-9.000"  # the window still applies
    assert cg.ClipRequest(url="https://x.example/v").audio_only is False


def test_find_output_prefers_m4a_for_audio(tmp_path):
    (tmp_path / "clip.m4a").write_bytes(b"a" * 10)
    (tmp_path / "clip.webm").write_bytes(b"b" * 100)
    assert cg.find_output(str(tmp_path), audio_only=True).name == "clip.m4a"
    assert cg.find_output(str(tmp_path)).name == "clip.webm"  # no mp4: largest media file
