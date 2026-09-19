"""The music agent: one plan per track (riff mode, seed, tempo, key, theme) and the YuE2 request that realises it."""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from . import abc as abclib

BASE_WEIGHTS = {"inspired": 50.0, "faithful": 20.0, "reinterpret": 15.0, "hook": 15.0}
COVER_MODES = ("faithful", "reinterpret", "hook")
COVER_COOLDOWN_S = 3600.0     # a faithful cover of the same seed at most once per hour of airtime
BPM_JITTER = 0.06
MIN_SECONDS, MAX_SECONDS = 120.0, 330.0
FALLBACK_THEMES = ["she keeps his voicemail but never calls back", "driving home after the diagnosis, radio off",
                   "closing the shop alone the night the team lost", "he finds her list of names for the dog",
                   "the friend who moved away texts at 2 a.m.", "moving out of the flat, one box is his",
                   "a promise kept a year too late", "walking home after the party, shoes in hand"]   # situations, not titles


@dataclass
class Plan:
    mode: str                      # inspired | faithful | reinterpret | hook
    seed_id: Optional[str]
    seed_title: Optional[str]
    theme: str
    bpm: int
    key: Optional[str]
    mood: list[str]
    duration_s: float
    explain: str
    cot: str = "full"
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _weights(settings: dict[str, Any], score_seeds: list[dict[str, Any]]) -> dict[str, float]:
    w = dict(BASE_WEIGHTS)
    covers = float(settings.get("covers", 1.0))        # 0 = never covers, 1 = default, 2 = cover-heavy
    for m in COVER_MODES:
        w[m] *= covers
    if not score_seeds:
        for m in COVER_MODES:
            w[m] = 0.0
    if not any(("chorus" in (s.lower()) for s in a.get("analysis", {}).get("sections") or []) for a in score_seeds):
        w["hook"] = 0.0
    return w


def choose_mode(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(weights.values())
    if total <= 0:
        return "inspired"
    x = rng.random() * total
    for m, wt in weights.items():
        x -= wt
        if x <= 0:
            return m
    return "inspired"


def plan_track(*, station: dict[str, Any], seeds: list[dict[str, Any]], history: list[dict[str, Any]],
               rng: Optional[random.Random] = None, now: Optional[float] = None) -> Plan:
    """station = {name, profile, settings}; seeds = [{id, title, analysis}]; history = recent plans (newest last)."""
    rng = rng or random.Random()
    now = now or time.time()
    profile = station.get("profile") or {}
    settings = station.get("settings") or {}
    score_seeds = [s for s in seeds if (s.get("analysis") or {}).get("abc")]
    weights = _weights(settings, score_seeds)
    # creativity rules: no back-to-back cover of the same seed; per-seed faithful cooldown
    last = history[-1] if history else None
    recent_faithful = {h["seed_id"] for h in history if h.get("mode") == "faithful" and now - h.get("created", 0) < COVER_COOLDOWN_S}
    mode = choose_mode(rng, weights)
    seed = None
    if mode in COVER_MODES:
        pool = score_seeds
        if mode == "hook":
            pool = [s for s in pool if any("chorus" in x.lower() for x in s["analysis"].get("sections") or [])]
        if mode == "faithful":
            pool = [s for s in pool if s["id"] not in recent_faithful]
        if last and last.get("mode") in COVER_MODES and len(pool) > 1:
            pool = [s for s in pool if s["id"] != last.get("seed_id")]
        if not pool:
            mode = "inspired"
        else:
            # least recently used seed
            used_at = {h.get("seed_id"): h.get("created", 0) for h in history}
            pool = sorted(pool, key=lambda s: used_at.get(s["id"], 0))
            seed = pool[0]
    bpm_info = profile.get("bpm") or {"low": 100, "high": 120, "center": 110}
    if seed and seed["analysis"].get("bpm"):
        bpm = int(seed["analysis"]["bpm"])
    else:
        c = float(bpm_info["center"])
        bpm = int(round(min(bpm_info["high"], max(bpm_info["low"], c * (1 + rng.uniform(-BPM_JITTER, BPM_JITTER))))))
    key = seed["analysis"].get("key") if seed else (rng.choice(profile["keys"]) if profile.get("keys") else None)
    themes = settings.get("themes") or FALLBACK_THEMES
    theme = themes[len(history) % len(themes)]
    moods = [t["label"] for t in (profile.get("tags") or {}).get("mood", [])]
    rng.shuffle(moods)
    seconds = seed["analysis"].get("seconds") if seed and mode == "faithful" else profile.get("seconds")
    duration = float(min(MAX_SECONDS, max(MIN_SECONDS, seconds or 200.0)))
    explain = {"inspired": "new song in the station's sound", "faithful": f"cover of “{seed['title']}” with new words" if seed else "",
               "reinterpret": f"the melody of “{seed['title']}” in the station's sound" if seed else "",
               "hook": f"the hook of “{seed['title']}” with new verses" if seed else ""}[mode]
    return Plan(mode=mode, seed_id=seed["id"] if seed else None, seed_title=seed["title"] if seed else None, theme=theme, bpm=bpm,
                key=key, mood=moods[:2], duration_s=duration, explain=explain, cot="melody" if mode == "reinterpret" else "full", created=now)


def styled(profile_style: str, plan: Plan) -> str:
    """The station's style line with this track's tempo (and mood swap when the plan chose one)."""
    parts = [p.strip() for p in profile_style.split(",") if p.strip()]
    parts = [p for p in parts if not p.lower().endswith("bpm")]
    if plan.mood:
        parts = [p for p in parts if p.lower() not in {m.lower() for m in plan.mood}]
        parts[2:2] = plan.mood
    parts.append(f"{plan.bpm} BPM")
    return ", ".join(parts)[: abclib.MAX_STYLE_CHARS]


def song_request(*, style: str, lyrics: str, plan: Plan, seed_analysis: Optional[dict[str, Any]], seed: Optional[int] = None) -> dict[str, Any]:
    """The YuE2 sidecar request for a plan (port of vidmakr's validate_song_request riff branches)."""
    style = " ".join(style.split())[: abclib.MAX_STYLE_CHARS]
    lyrics = abclib.tidy_lyrics(lyrics)
    if not style or not lyrics:
        raise abclib.MusicError("style and lyrics are required")
    req: dict[str, Any] = {"style": style, "lyrics": lyrics, "cot": plan.cot}
    if seed is not None:
        req["seed"] = int(seed)
    abc = abclib.validate_abc((seed_analysis or {}).get("abc")) if plan.mode != "inspired" else None
    if plan.mode != "inspired" and abc is None:
        raise abclib.MusicError(f"{plan.mode} needs the seed's score")
    if abc is not None:
        abc, promoted = abclib.promote_sparse_vocals(abc)
        facts = abclib.abc_facts(abc)
        if promoted:
            req["vocal_promoted"] = promoted
        if plan.mode == "faithful":
            req["cot"], req["abc"] = "full", abc
        elif plan.mode == "reinterpret":
            req["cot"], req["abc"] = "melody", abclib.strip_chords(abc)
        else:  # hook
            req["cot"] = "full"
            req["style"] = abclib.style_with_key_bpm(style, facts)[: abclib.MAX_STYLE_CHARS]
            req["hook_abc"] = abc
            req["hook_sections"] = ["chorus"]
    elif plan.key:
        req["style"] = abclib.style_with_key_bpm(style, {"key": plan.key, "bpm": None})[: abclib.MAX_STYLE_CHARS]
    req["cover_mode"] = plan.mode
    return req
