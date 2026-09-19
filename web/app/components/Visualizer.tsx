"use client";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import * as THREE from "three";
import type { Song } from "@/lib/api";
import { sectionCues, type SectionCue } from "@/lib/abc";
import { SCENES, type Scene } from "@/lib/scenes";
import { BeatClock, adaptQuality, bandEnergies, frame, logSpectrum, meter, paletteFor, type Palette, type VisualFrame } from "@/lib/visual";
import type { RadioPlayer } from "@/lib/player";

export const FEED_CHANNEL = "soundscape-visual-feed";

/** Full-panel three.js visualizer driven by the player's AnalyserNode. Publishes each VisualFrame on a
 * BroadcastChannel (`soundscape-visual-feed`) and as `window.soundscapeFeed` — the feed contract (docs/visual-feed.md). */
export default function Visualizer({ player, song, tags, sceneName, onScene, label, background }: {
  player: RadioPlayer | null; song: Song | null; tags: { mood?: { label: string }[]; genre?: { label: string }[] } | null;
  sceneName: string; onScene: (n: string) => void; label?: string; background?: boolean;
}) {
  const host = useRef<HTMLDivElement | null>(null);
  const [fps, setFps] = useState(0);
  const [rms, setRms] = useState(0);
  const [hud, setHud] = useState<VisualFrame | null>(null);
  const songRef = useRef<Song | null>(song); songRef.current = song;
  const tagsRef = useRef(tags); tagsRef.current = tags;

  useEffect(() => {
    const el = host.current;
    if (!el || !player) return;
    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    renderer.setClearColor(0x000000, 1);
    // The canvas fills its host by CSS; setSize(..., false) below never touches its style, so a lower pixel ratio only
    // lowers the backing resolution instead of shrinking the canvas itself into the top-left corner.
    renderer.domElement.style.cssText = "display:block;width:100%;height:100%";
    el.appendChild(renderer.domElement);
    const scene3 = new THREE.Scene();
    const cam = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
    cam.position.set(0, 0, 3.2);
    let sc: Scene = (SCENES[sceneName] ?? SCENES.radial)();
    scene3.add(sc.object);
    const analyser = player.analyser;
    const fft = new Uint8Array(analyser.frequencyBinCount);
    const channel = typeof BroadcastChannel !== "undefined" ? new BroadcastChannel(FEED_CHANNEL) : null;
    let clock = new BeatClock(120, 0), cues: SectionCue[] = [], palette: Palette = paletteFor(null), cueSong: string | null = null;
    let last = performance.now(), frames = 0, fpsT = last, raf = 0, quality = 1, hudT = 0;
    const resize = () => { const w = el.clientWidth, h = el.clientHeight; renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); };
    resize();
    const ro = new ResizeObserver(resize); ro.observe(el);
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const now = performance.now(); const dt = Math.min(0.1, (now - last) / 1000); last = now;
      const s = songRef.current;
      if (s && s.id !== cueSong) {                     // new song: cues, clock, palette
        cueSong = s.id;
        cues = s.abc ? sectionCues(s.abc, s.seconds) : [];
        const pos = player.position();
        clock = new BeatClock(s.plan?.bpm ?? 120, pos.t);
        palette = paletteFor(tagsRef.current, s.id.charCodeAt(0));
      }
      analyser.getByteFrequencyData(fft);
      const bands = bandEnergies(fft, player.ctx.sampleRate, analyser.fftSize);
      const pos = player.position();
      clock.update(pos.t, bands.bass);
      const spectrum = logSpectrum(fft, player.ctx.sampleRate, analyser.fftSize, 64);
      const f: VisualFrame = frame(pos.t, bands, clock, cues, palette, s ? { id: s.id, title: s.title, mode: s.plan?.mode ?? null } : null, pos.d, spectrum);
      sc.update(f, dt);
      renderer.render(scene3, cam);
      (window as unknown as { soundscapeFeed?: VisualFrame }).soundscapeFeed = f;
      channel?.postMessage(f);
      if (now - hudT > 100) { hudT = now; setHud(f); }               // HUD text at 10 Hz
      frames++;
      if (now - fpsT > 1000) {                         // adaptive quality: drop pixel ratio when below 45 fps
        const cur = frames * 1000 / (now - fpsT); setFps(Math.round(cur)); setRms(bands.rms);
        const attentive = document.visibilityState === "visible" && document.hasFocus();   // throttled frames are not a slow GPU
        const q = adaptQuality(quality, cur, now - fpsT, attentive);
        if (q !== quality) { quality = q; renderer.setPixelRatio(Math.min(2, window.devicePixelRatio) * quality); }
        frames = 0; fpsT = now;
      }
    };
    tick();
    return () => { cancelAnimationFrame(raf); ro.disconnect(); sc.dispose(); renderer.dispose(); channel?.close(); el.removeChild(renderer.domElement); };
  }, [player, sceneName]);

  const fullscreen = () => { const el = host.current?.parentElement; if (!el) return; if (document.fullscreenElement) void document.exitFullscreen(); else void el.requestFullscreen(); };
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const body = (
    <div className={background ? "fixed inset-y-0 left-0 right-0 sm:right-[420px] z-0 bg-black" : "relative w-full aspect-video bg-black rounded-xl overflow-hidden border border-zinc-800"}>
      <div ref={host} className="absolute inset-0" />
      <div className="absolute bottom-3 right-3 flex gap-1 text-[11px] text-zinc-500">
        {Object.keys(SCENES).map((n) => <button key={n} onClick={() => onScene(n)} className={`px-2 py-0.5 rounded border ${n === sceneName ? "border-zinc-300 text-zinc-100" : "border-zinc-800"}`}>{n}</button>)}
        <button onClick={fullscreen} className="px-2 py-0.5 rounded border border-zinc-800">⛶</button>
        <span className="px-1 font-mono" title="fps · analyser signal level (0.00 while a song plays = no audio reaching the analyser)">{fps} fps · {rms.toFixed(2)}</span>
      </div>
      {sceneName === "pulse" ? (
        <>
          <div className="crt-scanlines" /><div className="crt-vignette" />
          <div className="absolute top-4 left-5 font-mono text-[11px] leading-5 pointer-events-none" style={{ color: "#ff2a3c", textShadow: "0 0 6px rgba(255,42,60,.7)" }}>
            <div style={{ color: "#a0141f" }}>SOUNDSCAPE LINK PROTOCOL // SPECTRAL TELEMETRY</div>
            <div className="font-bold tracking-wider">[ {(label ?? "SOUNDSCAPE").replace(/^Soundscape · /i, "STATION :: ").toUpperCase()} ]</div>
          </div>
          <div className="absolute bottom-4 left-5 font-mono text-[11px] leading-5 pointer-events-none" style={{ color: "#ff2a3c", textShadow: "0 0 6px rgba(255,42,60,.7)" }}>
            <div className="font-bold">── NOW PLAYING ── {(song?.title ?? "STANDBY").toUpperCase()}</div>
            <div>BASS {meter(hud?.bands.bass ?? 0, 18)} {String(Math.round((hud?.bands.bass ?? 0) * 100)).padStart(3)}%</div>
            <div>MID  {meter((hud?.bands.mid ?? 0) * 1.6, 18)} {String(Math.round((hud?.bands.mid ?? 0) * 100)).padStart(3)}%</div>
            <div>TREB {meter((hud?.bands.treble ?? 0) * 3, 18)} {String(Math.round((hud?.bands.treble ?? 0) * 100)).padStart(3)}%</div>
            <div style={{ color: "#a0141f" }}>BPM {hud?.beat.bpm ?? "---"} · BAR {hud?.beat.bar ?? 0} · BEAT {meter(1 - (hud?.beat.phase ?? 0), 4)} · SECTION :: {(hud?.section?.label ?? "—").toUpperCase()}{song?.plan ? ` · MODE :: ${song.plan.mode.toUpperCase()}` : ""}</div>
          </div>
        </>
      ) : (
        <>
          {label && <div className="absolute top-4 left-5 text-zinc-500 text-xs tracking-[0.3em] uppercase pointer-events-none">{label}</div>}
          {song && <div className="absolute bottom-4 left-5 text-zinc-300 text-base pointer-events-none">{song.title}</div>}
        </>
      )}
      {!player && <div className="absolute inset-0 flex items-center justify-center text-zinc-600 text-sm">press Play</div>}
    </div>
  );
  // background mode renders straight into <body>: a fixed layer must not sit under a blurred/transformed ancestor
  if (background) return mounted ? createPortal(body, document.body) : null;
  return body;
}
