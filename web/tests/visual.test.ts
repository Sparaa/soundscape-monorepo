import { describe, expect, it } from "vitest";
import { sectionAt, sectionCues } from "@/lib/abc";
import { BeatClock, PunchDetector, adaptQuality, bandEnergies, frame, logSpectrum, meter, paletteFor } from "@/lib/visual";

const SCORE = ["X:1", "M:4/4", "L:1/16", "Q:1/4=120", "V: Vocal", "V: Ins", "K:C",
  "% intro", "V: Vocal", "Z|Z|", "V: Ins", "C4E4G4c4|C4E4G4c4|",
  "% verse", "V: Vocal", "C2D2E2F2G2A2B2c2|", "V: Ins", "Z|",
  "% chorus", "V: Vocal", "c4c4c4c4|", "V: Ins", "Z|"].join("\n");

describe("section cues", () => {
  it("finds section starts from the planned score and rescales to the actual length", () => {
    const cues = sectionCues(SCORE);                 // 120 BPM: one bar = 2 s
    expect(cues).toEqual([{ label: "intro", t: 0 }, { label: "verse", t: 4 }, { label: "chorus", t: 6 }]);
    expect(sectionCues(SCORE, 16).map((c) => c.t)).toEqual([0, 8, 12]);
    expect(sectionAt(cues, 5)?.label).toBe("verse");
    expect(sectionAt(cues, 100)?.label).toBe("chorus");
    expect(sectionCues("X:1\nK:C\n% verse\nV: Vocal\nz|")).toEqual([{ label: "verse", t: 0 }]);
  });
});

describe("bands + beat clock", () => {
  it("splits the FFT into bands", () => {
    const fft = new Uint8Array(1024).fill(0);
    for (let i = 0; i < 4; i++) fft[i] = 255;        // 44.1k/2048 = 21.5 Hz per bin → bins 1-3 are bass
    const b = bandEnergies(fft, 44100, 2048);
    expect(b.bass).toBeGreaterThan(0.4);
    expect(b.treble).toBe(0);
    expect(b.rms).toBeGreaterThan(0);
  });
  it("keeps the planned tempo and pulls the phase toward onsets", () => {
    const c = new BeatClock(120, 0);
    expect(c.period).toBe(0.5);
    expect(c.phase(0.25)).toBeCloseTo(0.5);
    expect(c.beatIndex(2.1)).toBe(4);
    expect(c.bar(2.1)).toBe(1);
    // a hit slightly late (t=0.55, phase 0.1) pulls t0 forward a little
    c.update(0.5, 0.1);
    expect(c.update(0.55, 0.9)).toBe(true);
    expect(c.hit).toBe(1);
    expect(c.phase(0.55)).toBeLessThan(0.1);
    expect(c.update(0.6, 0.95)).toBe(false);        // refractory: no double trigger within half a beat
  });
});

describe("palette + frame", () => {
  it("is deterministic from tags", () => {
    const p = paletteFor({ mood: [{ label: "dark" }], genre: [{ label: "EDM" }] });
    expect(p.hue).toBe(260); expect(p.sat).toBe(0.55); expect(p.name).toBe("dark");
    expect(paletteFor({ genre: [{ label: "EDM" }] }).hue).toBe(190);
    expect(paletteFor(null, 3).hue).toBe(141);
  });
  it("builds a feed frame with section progress", () => {
    const c = new BeatClock(120, 0);
    const f = frame(5, { bass: 0.5, mid: 0.2, treble: 0.1, rms: 0.3 }, c, sectionCues(SCORE), paletteFor(null), { id: "s", title: "T", mode: "inspired" }, 8);
    expect(f.v).toBe(1); expect(f.section).toEqual({ label: "verse", index: 1, progress: 0.5 }); expect(f.beat.index).toBe(10);
  });
});

describe("logSpectrum", () => {
  it("gives equal weight per octave and rides along in the frame", () => {
    const fft = new Uint8Array(1024).fill(0);
    for (let i = 1; i < 8; i++) fft[i] = 255;         // 21.5–172 Hz hot
    const s = logSpectrum(fft, 44100, 2048, 16);
    expect(s).toHaveLength(16);
    expect(s[0]).toBeGreaterThan(0.9);                // bass bins full
    expect(s[15]).toBe(0);                            // treble empty
    expect(s.slice(0, 5).every((v) => v > 0.5)).toBe(true);
    const c = new BeatClock(120, 0);
    const f = frame(1, { bass: 1, mid: 0, treble: 0, rms: 0.5 }, c, [], paletteFor(null), null, 10, s);
    expect(f.spectrum).toHaveLength(16);
    expect(frame(1, f.bands, c, [], f.palette, null, 10).spectrum).toBeUndefined();
  });
});

describe("meter", () => {
  it("draws GPU-Pulse style bars and clamps", () => {
    expect(meter(0.5, 10)).toBe("▰▰▰▰▰▱▱▱▱▱");
    expect(meter(2, 4)).toBe("▰▰▰▰");
    expect(meter(-1, 4)).toBe("▱▱▱▱");
    expect(meter(NaN, 3)).toBe("▱▱▱");
  });
});

describe("adaptQuality", () => {
  it("steps down on a genuinely slow second and back up on a fast one", () => {
    expect(adaptQuality(1, 30, 1000, true)).toBe(0.75);
    expect(adaptQuality(0.75, 30, 1000, true)).toBe(0.5);
    expect(adaptQuality(0.5, 30, 1000, true)).toBe(0.5);          // floor
    expect(adaptQuality(0.5, 60, 1000, true)).toBe(0.75);
    expect(adaptQuality(1, 60, 1000, true)).toBe(1);              // ceiling
    expect(adaptQuality(1, 50, 1000, true)).toBe(1);              // steady zone
  });
  it("ignores throttled samples: unfocused / hidden window, or a sample spanning a gap", () => {
    expect(adaptQuality(1, 5, 1000, false)).toBe(1);              // focus went to another window
    expect(adaptQuality(1, 2, 12000, true)).toBe(1);              // tab was frozen for 12 s
    expect(adaptQuality(0.5, 60, 1000, false)).toBe(0.5);         // nor climb while unfocused: nothing to see
  });
});

describe("punch (percussive transients)", () => {
  const quiet = new Array(64).fill(0.1);
  it("fires on a broadband transient above the bass band, then decays", () => {
    const d = new PunchDetector();
    for (let i = 0; i < 30; i++) d.update(quiet);                      // settle the running floor
    const snare = quiet.map((v, i) => (i >= 20 && i < 50 ? v + 0.35 : v));   // ~250 Hz–3 kHz burst: a snare, not a kick
    const hit = d.update(snare);
    expect(hit).toBeGreaterThan(0.8);
    const after = [d.update(snare), d.update(snare), d.update(snare)];     // sustained level = no new flux → decays
    expect(after[2]).toBeLessThan(hit * 0.6);
    expect(after[2]).toBeGreaterThan(0);
  });
  it("adapts to the track: a steady rumble is not punch", () => {
    const d = new PunchDetector();
    let last = 0;
    for (let i = 0; i < 60; i++) last = d.update(quiet.map((v, k) => v + 0.02 * Math.sin(i * 0.7 + k)));
    expect(last).toBeLessThan(0.35);
  });
  it("rides along in the frame", () => {
    const f = frame(1, { bass: 0, mid: 0, treble: 0, rms: 0 }, new BeatClock(120), [], paletteFor(null), null, 10, undefined, 0.7);
    expect(f.beat.punch).toBe(0.7);
  });
});
