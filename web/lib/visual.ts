/** Visualizer inputs, pure and testable: band energies from an FFT, a beat clock that stays locked to the planned BPM
 * and nudges its phase toward bass onsets, palettes from tags, and the VisualFeed frame — the contract a native
 * (Vulkan) front end can consume later. */
import type { SectionCue } from "./abc";

export interface Bands { bass: number; mid: number; treble: number; rms: number }

/** Byte FFT (0..255 per bin) → normalized band energies. Bass < 150 Hz, mid 150–2000 Hz, treble > 2000 Hz. */
export function bandEnergies(fft: Uint8Array | number[], sampleRate: number, fftSize: number): Bands {
  const bins = fft.length;
  const hzPerBin = sampleRate / fftSize;
  const sum = (lo: number, hi: number) => {
    const a = Math.max(0, Math.floor(lo / hzPerBin)), b = Math.min(bins - 1, Math.ceil(hi / hzPerBin));
    if (b < a) return 0;
    let s = 0;
    for (let i = a; i <= b; i++) s += fft[i];
    return s / ((b - a + 1) * 255);
  };
  let sq = 0;
  for (let i = 0; i < bins; i++) sq += (fft[i] / 255) ** 2;
  return { bass: sum(20, 150), mid: sum(150, 2000), treble: sum(2000, 12000), rms: Math.sqrt(sq / Math.max(1, bins)) };
}

/** Byte FFT → n log-spaced bins between lo and hi Hz (0..1 each; mean of the bins covered, at least one). This is what
 * bar/ray visualizers want: equal visual weight per octave instead of per Hz. */
export function logSpectrum(fft: Uint8Array | number[], sampleRate: number, fftSize: number, n = 64, lo = 30, hi = 9000): number[] {
  const hzPerBin = sampleRate / fftSize, out = new Array<number>(n);
  const ratio = Math.log(hi / lo) / n;
  for (let i = 0; i < n; i++) {
    const f0 = lo * Math.exp(i * ratio), f1 = lo * Math.exp((i + 1) * ratio);
    const a = Math.max(0, Math.floor(f0 / hzPerBin)), b = Math.max(a, Math.min(fft.length - 1, Math.floor(f1 / hzPerBin)));
    let sum = 0;
    for (let k = a; k <= b; k++) sum += fft[k];
    out[i] = sum / ((b - a + 1) * 255);
  }
  return out;
}

/** Beat clock: phase runs at the planned BPM; bass onsets (energy jumps) pull the phase toward the hit (PLL gain
 * `lock`), so beats stay in step even when the mix dips. */
export class BeatClock {
  bpm: number;
  private t0 = 0;
  private lastBass = 0;
  private lastHitT = -1;
  hit = 0;                 // 1 at a detected hit, decays per frame
  constructor(bpm: number, startT = 0, public lock = 0.15, public threshold = 0.12) { this.bpm = Math.max(40, bpm || 120); this.t0 = startT; }
  get period(): number { return 60 / this.bpm; }
  phase(t: number): number { const p = ((t - this.t0) / this.period) % 1; return p < 0 ? p + 1 : p; }
  beatIndex(t: number): number { return Math.floor((t - this.t0) / this.period); }
  bar(t: number): number { return Math.floor(this.beatIndex(t) / 4); }
  /** Feed one frame: t (s), bass 0..1. Returns true when an onset was detected. */
  update(t: number, bass: number): boolean {
    const flux = bass - this.lastBass;
    this.lastBass = bass;
    this.hit *= 0.85;
    if (flux > this.threshold && t - this.lastHitT > this.period * 0.45) {
      this.lastHitT = t;
      this.hit = 1;
      // pull t0 so that t lands on a beat: error in (-0.5, 0.5] periods
      let err = this.phase(t);
      if (err > 0.5) err -= 1;
      this.t0 += err * this.period * this.lock;
      return true;
    }
    return false;
  }
}

/** Percussive transient envelope ("punch"): positive spectral flux across the log spectrum from ~60 Hz to ~6 kHz
 * (kick body, snare, claps, hat attacks), measured against the track's own running flux floor so quiet and loud mixes
 * pulse alike; fast attack, ~100 ms decay. Unlike `BeatClock.hit` it has no refractory period and is not bass-only, so
 * the center of the scope can move with the whole drum kit. Bins index the 64-bin logSpectrum (30 Hz–9 kHz). */
export class PunchDetector {
  private prev: number[] = [];
  private avg = 0.02;
  punch = 0;
  constructor(public lo = 8, public hi = 60, public decay = 0.78, public gain = 2.5) {}
  update(spec: number[]): number {
    let flux = 0, n = 0;
    for (let i = Math.max(0, this.lo); i < Math.min(spec.length, this.hi); i++) {
      const d = spec[i] - (this.prev[i] ?? spec[i]);
      if (d > 0) flux += d;
      n++;
    }
    this.prev = spec.slice();
    flux = n ? flux / n : 0;
    this.avg += (flux - this.avg) * 0.05;
    const raw = Math.min(1, Math.max(0, (flux - this.avg) / Math.max(0.004, this.avg * this.gain)));
    this.punch = Math.max(raw, this.punch * this.decay);
    return this.punch;
  }
}

export interface Palette { hue: number; sat: number; light: number; accentHue: number; name: string }

const MOOD_HUES: Record<string, number> = { dark: 260, epic: 30, energetic: 10, playful: 320, happy: 50, sad: 210, calm: 170, dreamy: 280, aggressive: 0, romantic: 340, melancholic: 220, hopeful: 100 };
const GENRE_HUES: Record<string, number> = { edm: 190, "synth-pop": 300, "k-pop": 320, "hip hop": 40, rock: 15, metal: 0, jazz: 45, classical: 200, "orchestral film score": 30, folk: 90, country: 60, ambient: 180, techno: 200, house: 280, "lo-fi": 25, pop: 330 };

/** Deterministic palette from mood/genre tags: hue from the strongest mood (genre as fallback), accent = complement-ish. */
export function paletteFor(tags: { mood?: { label: string }[]; genre?: { label: string }[] } | null | undefined, seed = 0): Palette {
  const mood = tags?.mood?.[0]?.label?.toLowerCase();
  const genre = tags?.genre?.[0]?.label?.toLowerCase();
  let hue = mood && mood in MOOD_HUES ? MOOD_HUES[mood] : genre && genre in GENRE_HUES ? GENRE_HUES[genre] : (seed * 47) % 360;
  hue = (hue + 360) % 360;
  const dark = mood === "dark" || mood === "melancholic" || mood === "sad";
  return { hue, sat: dark ? 0.55 : 0.8, light: dark ? 0.45 : 0.6, accentHue: (hue + 150) % 360, name: mood ?? genre ?? "neutral" };
}

/** The visual feed contract (JSON-serialisable, one frame per animation tick). Native front ends consume this. */
export interface VisualFrame {
  v: 1;
  t: number;                       // playback position (s)
  bands: Bands;
  beat: { phase: number; index: number; bar: number; bpm: number; hit: number; punch: number };
  section: { label: string; index: number; progress: number } | null;
  palette: Palette;
  song: { id: string; title: string | null; mode: string | null } | null;
  spectrum?: number[];             // optional: 64 log-spaced bins 30 Hz–9 kHz, 0..1 (bar/ray scenes)
}

export function frame(t: number, bands: Bands, clock: BeatClock, cues: SectionCue[], palette: Palette,
                      song: VisualFrame["song"], durationS: number, spectrum?: number[], punch = 0): VisualFrame {
  let idx = -1;
  for (let i = 0; i < cues.length; i++) if (cues[i].t <= t) idx = i; else break;
  const section = idx >= 0 ? (() => {
    const start = cues[idx].t, end = idx + 1 < cues.length ? cues[idx + 1].t : Math.max(durationS, start + 1);
    return { label: cues[idx].label, index: idx, progress: Math.min(1, Math.max(0, (t - start) / Math.max(0.001, end - start))) };
  })() : null;
  return { v: 1, t, bands, beat: { phase: clock.phase(t), index: clock.beatIndex(t), bar: clock.bar(t), bpm: clock.bpm, hit: clock.hit, punch }, section, palette, song,
           ...(spectrum ? { spectrum } : {}) };
}

/** GPU-Pulse style text meter: meter(0.62, 10) → "▰▰▰▰▰▰▱▱▱▱". */
export function meter(v: number, n = 12): string {
  const k = Math.max(0, Math.min(n, Math.round((isFinite(v) ? v : 0) * n)));
  return "▰".repeat(k) + "▱".repeat(n - k);
}

/** Adaptive render quality (a multiplier on the device pixel ratio, 0.5..1 in 0.25 steps) from one second's frame rate.
 * A sample taken while the window was unfocused, hidden or occluded says nothing about the GPU — browsers throttle
 * animation frames there — so it is ignored; so is a sample that spans a gap (the tab was frozen). Without this the
 * visualizer stepped down to half resolution whenever the window lost focus and climbed back over two seconds. */
export function adaptQuality(quality: number, fps: number, sampleMs: number, attentive: boolean): number {
  if (!attentive || sampleMs > 1500) return quality;
  if (fps < 45 && quality > 0.5) return quality - 0.25;
  if (fps > 58 && quality < 1) return quality + 0.25;
  return quality;
}
