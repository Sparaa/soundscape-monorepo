# Visual feed contract (v1)

Every animation tick the web visualizer publishes one `VisualFrame` (JSON) on `BroadcastChannel("soundscape-visual-feed")`
and as `window.soundscapeFeed`. A native front end (the Vulkan idea) consumes the same frames — over a WebSocket bridge
later — so scenes port without re-deriving audio analysis.

```ts
interface VisualFrame {
  v: 1;
  t: number;                                   // playback position, seconds
  bands: { bass: number; mid: number; treble: number; rms: number };   // 0..1 (bass <150 Hz, mid 150–2k, treble >2k)
  beat: { phase: number; index: number; bar: number; bpm: number; hit: number; punch: number };
        // phase 0..1 within the beat (planned BPM, phase-locked to bass onsets), hit = 1 on an onset, decays per frame
        // punch 0..1: percussive transient envelope — positive spectral flux 60 Hz–6 kHz against the track's own
        // running floor, no refractory period, ~100 ms decay; the center of the scope pulses with it (kick + snare + hats)
  section: { label: string; index: number; progress: number } | null;  // from the song's planned score (ABC % labels)
  palette: { hue: number; sat: number; light: number; accentHue: number; name: string };  // from the station's mood/genre
  song: { id: string; title: string | null; mode: string | null } | null;
  spectrum?: number[];                         // 64 log-spaced bins, 30 Hz–9 kHz, 0..1 — what bar/ray scenes draw
}
```
Sources: `web/lib/visual.ts` (bands, `BeatClock`, `paletteFor`, `frame`), `web/lib/abc.ts` (`sectionCues`). Scenes in
`web/lib/scenes.ts` take a frame and a `dt`; that is the whole interface a scene needs.

Scenes: `pulse` (red-phosphor CRT: segmented meter rays in one red, glow pass, scanlines, vignette and a monospace
telemetry HUD — the GPU Pulse look), `radial` (default — ring bars, colored rays per bin with bass at the bottom, afterglow beams, starfield disc,
rotating emblem; drop a `web/public/logo.png` to replace the placeholder trefoil), `nebula`, `rings`.
