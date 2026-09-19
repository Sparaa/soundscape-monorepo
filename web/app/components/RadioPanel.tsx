"use client";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { patchSettings, patchSong, planLabel, playlistExportUrl, radioNext, radioPlay, radioStatus, radioStop, removeFromPlaylist, songAudioUrl, stationPlaylist, type Playlist, type RadioStatus, type Song, type Station } from "@/lib/api";
import { mmss } from "@/lib/profile";
import { RadioPlayer } from "@/lib/player";
import Visualizer from "@/app/components/Visualizer";

type Mode = "radio" | "playlist";
type Tab = "playlist" | "seeds" | "profile";

export default function RadioPanel({ station, onStation, seedsPane, profilePane }: {
  station: Station; onStation: (s: Station) => void; seedsPane?: React.ReactNode; profilePane?: React.ReactNode;
}) {
  const [status, setStatus] = useState<RadioStatus | null>(null);
  const [song, setSong] = useState<Song | null>(null);
  const [pos, setPos] = useState({ t: 0, d: 0 });
  const [err, setErr] = useState<string | null>(null);
  const [paused, setPausedState] = useState(false);
  const setPaused = (v: boolean) => { pausedRef.current = v; setPausedState(v); };
  const [playerObj, setPlayerObj] = useState<RadioPlayer | null>(null);
  const [scene, setScene] = useState<string>(() => { try { return localStorage.getItem("soundscape.scene") ?? "radial"; } catch { return "radial"; } });
  const player = useRef<RadioPlayer | null>(null);
  const sid = station.id;
  // what plays NEXT: fresh songs from the agent ("radio") or the station's saved songs in order ("playlist")
  const [mode, setModeState] = useState<Mode>("radio");
  const modeRef = useRef<Mode>("radio");
  const setMode = (m: Mode) => { modeRef.current = m; setModeState(m); };
  const [tab, setTab] = useState<Tab>("playlist");
  const [playlist, setPlaylist] = useState<Playlist | null>(null);
  const playlistRef = useRef<Playlist | null>(null); playlistRef.current = playlist;
  const plIndex = useRef(-1);

  const pausedRef = useRef(false);
  const refresh = useCallback(async () => {
    try {
      const st = await radioStatus(sid);
      setStatus(st);
      setErr(null);                                   // a transient failure (API restart, blip) clears itself
      // the API forgot we were playing (it restarted): tell it again so the buffer keeps filling (radio mode only —
      // saved songs need no rendering)
      if (st.state === "stopped" && modeRef.current === "radio" && player.current && player.current.current.song && !pausedRef.current) setStatus(await radioPlay(sid));
    } catch (e) { setErr(`connection to the API lost — retrying (${String(e).slice(0, 80)})`); }
  }, [sid]);
  useEffect(() => { void refresh(); const id = window.setInterval(refresh, 2000); return () => window.clearInterval(id); }, [refresh]);
  useEffect(() => { const id = window.setInterval(() => player.current && setPos(player.current.position()), 500); return () => window.clearInterval(id); }, []);
  useEffect(() => () => player.current?.stop(), []);
  useEffect(() => { document.documentElement.dataset.scene = scene; return () => { delete document.documentElement.dataset.scene; }; }, [scene]);

  const getPlayer = () => {
    if (!player.current) {
      const p = new RadioPlayer(songAudioUrl);
      p.onSongChange = setSong;
      p.onNextError = (e) => setErr(`next song: ${String(e).slice(0, 80)} — retrying`);   // cleared by the next good status poll
      p.onNeedNext = async () => {
        if (modeRef.current === "playlist") {                       // saved songs, in order, wrapping around
          const items = playlistRef.current?.items ?? [];
          if (!items.length) return null;
          plIndex.current = (plIndex.current + 1) % items.length;
          return items[plIndex.current];
        }
        const r = await radioNext(sid); setStatus(r.status); return r.song;
      };
      player.current = p;
      setPlayerObj(p);
    }
    return player.current;
  };

  const onPlay = async () => {
    setErr(null);
    try {
      const p = getPlayer();
      if (modeRef.current === "playlist") {                          // saved songs need no rendering
        if (p.paused && (await p.resume())) { setPaused(false); return; }
        await playSaved(Math.max(0, plIndex.current));
        return;
      }
      const st = await radioPlay(sid);
      setStatus(st);
      if (p.paused && (await p.resume())) { setPaused(false); return; }   // Stop paused it: pick the same song up again
      // start as soon as the first song is cued (the spare makes this instant after the first session)
      const first = (await radioNext(sid));
      setStatus(first.status);
      if (first.song) await p.start(first.song);
      else {
        const wait = window.setInterval(async () => {
          const r = await radioNext(sid); setStatus(r.status);
          if (r.song) { window.clearInterval(wait); await p.start(r.song); }
          if (r.status.state === "stopped") window.clearInterval(wait);
        }, 3000);
      }
    } catch (e) { setErr(String(e)); }
  };
  const onStop = async () => { player.current?.pause(); setPaused(true); setStatus(await radioStop(sid)); };
  /** Play a saved song now (crossfading out of whatever is on) and continue through the playlist from there. */
  const playSaved = async (i: number) => {
    const items = playlistRef.current?.items ?? [];
    if (!items[i]) return;
    setErr(null); setMode("playlist"); plIndex.current = i;
    const p = getPlayer();
    try {
      if (p.current.song) { await p.ctx.resume(); await p.playNow(async () => items[i], 0.4); }
      else await p.start(items[i]);
      setPaused(false);
    } catch (e) { setErr(String(e)); }
  };
  /** Play a cued song now (the Up next list is clickable): the API hands over that exact song and the buffer refills. */
  const playCued = async (s: Song) => {
    setErr(null); setMode("radio");
    const p = getPlayer();
    try {
      if (paused || (status && status.state !== "playing" && status.state !== "warming")) { setStatus(await radioPlay(sid)); setPaused(false); }
      const get = async () => { const r = await radioNext(sid, s.id); setStatus(r.status); return r.song; };
      if (p.current.song) await p.playNow(get, 0.5);
      else { const first = await get(); if (first) await p.start(first); }
    } catch (e) { setErr(`play now: ${String(e).slice(0, 80)}`); }
  };
  /** Back to fresh songs: the agent resumes buffering; the current song plays out, Skip jumps to a new one. */
  const toRadio = async () => { setMode("radio"); try { setStatus(await radioPlay(sid)); } catch (e) { setErr(String(e)); } };
  useEffect(() => { stationPlaylist(sid).then(setPlaylist).catch(() => undefined); }, [sid, song?.id, status?.ready.length]);
  const onSkip = async () => {
    try {
      if (paused) { await radioPlay(sid); setPaused(false); }
      await getPlayer().skip();
    } catch (e) { setErr(`skip: ${String(e).slice(0, 80)}`); }
  };
  const flag = async (s: Song, flags: { saved?: boolean; liked?: boolean; vote?: -1 | 0 | 1 }) => {
    const u = await patchSong(s.id, flags);
    if (song?.id === s.id) setSong(u);
    void refresh();
  };
  const covers = Number(station.settings?.covers ?? 1);
  const live = !paused && (mode === "playlist" ? !!song : (status?.state === "playing" || status?.state === "warming"));
  return (
    <section className="flex flex-col gap-3">
      <Visualizer player={playerObj} song={song} tags={station.profile?.tags ?? null} sceneName={scene} label={`Soundscape · ${station.name}`} background
                  onScene={(n) => { setScene(n); try { localStorage.setItem("soundscape.scene", n); } catch { /* per-viewer convenience only */ } }} />
      <div className="pane flex flex-col gap-3">
        <div className="flex items-center gap-3">
          {!live ? (
            <button onClick={onPlay} disabled={!station.profile} className="px-5 py-2 rounded-full bg-zinc-100 text-black font-medium disabled:opacity-40">▸ Play</button>
          ) : (
            <button onClick={onStop} className="px-5 py-2 rounded-full border border-zinc-600">■ Stop</button>
          )}
          <button onClick={onSkip} disabled={!song} className="px-3 py-2 rounded-full border border-zinc-800 disabled:opacity-40">» Skip</button>
          <div className="ml-auto flex rounded-full border border-zinc-700 overflow-hidden text-xs" title="What plays next: fresh songs from the agent, or this station's saved songs">
            <button onClick={toRadio} className={`px-3 py-1.5 ${mode === "radio" ? "bg-zinc-100 text-black" : "text-zinc-400"}`}>new</button>
            <button onClick={() => { setMode("playlist"); if (!song) void playSaved(0); }} disabled={!playlist?.items.length} className={`px-3 py-1.5 disabled:opacity-40 ${mode === "playlist" ? "bg-zinc-100 text-black" : "text-zinc-400"}`}>saved</button>
          </div>
        </div>
        <div className="text-xs text-zinc-500 font-mono">{mode === "radio" ? `${status?.state ?? "…"} · ${status?.ready.length ?? 0} cued` : `playlist · song ${plIndex.current + 1} of ${playlist?.items.length ?? 0}`}</div>
        <label className="text-xs text-zinc-400 flex items-center gap-3"><span className="whitespace-nowrap">covers</span>
          <input type="range" min={0} max={2} step={0.25} value={covers} onChange={async (e) => onStation(await patchSettings(sid, { covers: Number(e.target.value) }))} className="flex-1 range" />
          <span className="whitespace-nowrap">new</span>
        </label>
        {song ? (
          <div className="flex flex-col gap-1">
            <div className="text-2xl font-semibold">{song.title}</div>
            <div className="text-sm text-zinc-400">{planLabel(song.plan)} · {song.plan?.bpm} BPM{song.plan?.key ? ` · ${song.plan.key}` : ""}</div>
            <div className="h-1 bg-zinc-800 rounded"><div className="h-1 bg-zinc-200 rounded" style={{ width: `${pos.d ? Math.min(100, (100 * pos.t) / pos.d) : 0}%` }} /></div>
            <div className="text-xs text-zinc-500 font-mono">{mmss(pos.t)} / {mmss(pos.d)}</div>
            <div className="flex gap-2 text-xs">
              <button onClick={() => flag(song, { saved: !song.saved })} className={`px-2 py-1 rounded border ${song.saved ? "border-emerald-500 text-emerald-300" : "border-zinc-700"}`}>{song.saved ? "saved" : "save"}</button>
              <button onClick={() => flag(song, { vote: song.vote > 0 ? 0 : 1 })} className={`px-2 py-1 rounded border ${song.vote > 0 ? "border-pink-500 text-pink-300" : "border-zinc-700"}`} title="more like this — steers the station">{song.vote > 0 ? "↑ more like this ✓" : "↑ more like this"}</button>
              <button onClick={() => flag(song, { vote: song.vote < 0 ? 0 : -1 })} className={`px-2 py-1 rounded border ${song.vote < 0 ? "border-amber-500 text-amber-300" : "border-zinc-700"}`} title="less like this — steers the station away">{song.vote < 0 ? "↓ less like this ✓" : "↓ less"}</button>
              <Link href="/library" className="px-2 py-1 text-zinc-500 hover:text-zinc-200">library →</Link>
            </div>
            {song.lyrics && <pre className="text-xs text-zinc-400 whitespace-pre-wrap font-sans mt-2 max-h-48 overflow-auto">{song.lyrics}</pre>}
          </div>
        ) : (
          <div className="text-sm text-zinc-500">{live ? "Composing the first song… (about a minute)" : "Press Play — the agent composes from the station profile and keeps two songs cued."}</div>
        )}
        {status?.rendering && (
          <div className="text-xs text-zinc-400">
            <span className="animate-pulse">●</span> rendering: {planLabel(status.rendering.plan)} · {status.rendering.stage} {Math.round(status.rendering.progress * 100)}% · {status.rendering.seconds}s{status.rendering.attempt > 1 ? ` · attempt ${status.rendering.attempt}` : ""}
          </div>
        )}
        {status && status.ready.length > 0 && (
          <div className="text-xs text-zinc-400 flex flex-col gap-0.5">
            <div className="uppercase tracking-widest text-zinc-500">Up next <span className="normal-case tracking-normal text-zinc-600">· click a song to play it now</span></div>
            {status.ready.map((s) => (
              <button key={s.id} onClick={() => playCued(s)} className="group flex items-center gap-2 text-left rounded px-1 -mx-1 hover:bg-zinc-900 hover:text-zinc-100" title="Play this song now">
                <span className="text-zinc-600 group-hover:text-zinc-100">▸</span>
                <span className="truncate">{s.title} <span className="text-zinc-600">· {planLabel(s.plan)} · {mmss(s.seconds)}</span></span>
              </button>
            ))}
          </div>
        )}
        {status && status.recent.some((s) => s.status === "rejected") && (
          <div className="text-xs text-zinc-600 flex flex-col gap-0.5">
            {status.recent.filter((s) => s.status === "rejected").slice(0, 2).map((s) => <div key={s.id}>rejected: <span className="line-through">{s.title}</span> — {s.gate?.reasons.join(", ")}</div>)}
          </div>
        )}
        {err && <div className="text-xs text-red-400">{err}</div>}
      </div>
      <div className="pane flex flex-col gap-3">
        <div className="flex gap-1 text-xs">
          {(["playlist", "seeds", "profile"] as Tab[]).map((t) => (
            <button key={t} onClick={() => setTab(t)} className={`px-3 py-1.5 rounded-full border capitalize ${tab === t ? "border-zinc-300 text-zinc-100" : "border-zinc-800 text-zinc-500 hover:text-zinc-300"}`}>
              {t}{t === "playlist" && playlist ? ` · ${playlist.items.length}` : t === "seeds" ? ` · ${station.seeds.length}` : ""}
            </button>
          ))}
        </div>
        {tab === "playlist" && (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-3 text-xs text-zinc-500">
              <span>{playlist ? `${playlist.items.length} songs · ${mmss(playlist.seconds)}` : "…"}</span>
              <button onClick={() => playSaved(0)} disabled={!playlist?.items.length} className="px-2 py-0.5 rounded border border-zinc-700 text-zinc-300 disabled:opacity-40">▸ play all</button>
              {playlist && <Link href={`/playlists/${playlist.id}`} className="hover:text-zinc-200">reorder →</Link>}
              {playlist && <a href={playlistExportUrl(playlist.id)} className="hover:text-zinc-200">export .zip</a>}
            </div>
            {playlist && playlist.items.length === 0 && <div className="text-sm text-zinc-500">Every song the radio cues lands here, ready to play again.</div>}
            <div className="flex flex-col max-h-[22rem] overflow-y-auto -mx-1">
              {playlist?.items.map((s, i) => {
                const on = song?.id === s.id;
                return (
                  <div key={s.id} className={`group flex items-center gap-2 px-2 py-1.5 rounded-lg ${on ? "bg-zinc-800/80" : "hover:bg-zinc-900"}`}>
                    <button onClick={() => playSaved(i)} className={`w-6 text-center ${on ? "text-zinc-50" : "text-zinc-500 group-hover:text-zinc-100"}`} title="Play this song now">{on && !paused ? <span className="now-cursor">▮</span> : "▸"}</button>
                    <button onClick={() => playSaved(i)} className="flex-1 min-w-0 text-left">
                      <div className={`text-sm truncate ${on ? "text-zinc-50 font-medium" : "text-zinc-200"}`}>{s.title}</div>
                      <div className="text-[11px] text-zinc-500 truncate">{planLabel(s.plan) || s.explain}</div>
                    </button>
                    <span className="text-[11px] text-zinc-500 font-mono">{s.vote > 0 ? "↑ " : s.vote < 0 ? "↓ " : ""}{mmss(s.seconds)}</span>
                    <button onClick={async () => { if (playlist) setPlaylist(await removeFromPlaylist(playlist.id, s.id)); }} className="text-zinc-700 hover:text-red-400 text-xs opacity-0 group-hover:opacity-100" title="Remove from this playlist (the file stays in the library)">✕</button>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        {tab === "seeds" && seedsPane}
        {tab === "profile" && profilePane}
      </div>
    </section>
  );
}
