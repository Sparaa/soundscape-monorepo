"use client";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { createStation, deleteStation, deleteStationPrompt, getHealth, listStations, sidecarLine, type Health, type Station } from "@/lib/api";
import { profileHeadline } from "@/lib/profile";

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [stations, setStations] = useState<Station[]>([]);
  const [name, setName] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const refresh = useCallback(() => {
    getHealth().then(setHealth).catch((e) => setErr(String(e)));
    listStations().then(setStations).catch((e) => setErr(String(e)));
  }, []);
  useEffect(refresh, [refresh]);
  const onCreate = async () => {
    if (!name.trim()) return;
    try { const st = await createStation(name.trim()); setName(""); setStations((s) => [st, ...s]); } catch (e) { setErr(String(e)); }
  };
  return (
    <main className="min-h-screen max-w-3xl mx-auto flex flex-col gap-8 p-8">
      <header>
        <h1 className="text-5xl font-semibold tracking-tight">Soundscape</h1>
        <p className="text-zinc-400 mt-1">A radio that never runs out of songs.</p>
        <nav className="text-sm text-zinc-400 mt-2 flex gap-4"><Link href="/library" className="hover:text-zinc-100">Library</Link><Link href="/playlists" className="hover:text-zinc-100">Playlists</Link></nav>
      </header>
      <section className="flex gap-2">
        <input value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && onCreate()} placeholder="New station name…"
               className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 outline-none focus:border-zinc-500" />
        <button onClick={onCreate} className="px-4 py-2 rounded-lg bg-zinc-100 text-black font-medium">Create</button>
      </section>
      <section className="flex flex-col gap-2">
        {stations.length === 0 && <p className="text-zinc-500">No stations yet — create one, then seed it with a song.</p>}
        {stations.map((s) => (
          <div key={s.id} className="flex items-center gap-3 border border-zinc-800 rounded-xl px-4 py-3">
            <Link href={`/stations/${s.id}`} className="flex-1">
              <div className="font-medium">{s.name}</div>
              <div className="text-xs text-zinc-400">{s.profile ? profileHeadline(s.profile) : `${s.seeds.length} seed(s) — add a song to build the profile`}{s.songs ? ` · ${s.songs} song${s.songs === 1 ? "" : "s"}` : ""}</div>
            </Link>
            <button onClick={() => { if (confirm(deleteStationPrompt(s))) deleteStation(s.id).then(refresh).catch((e) => setErr(String(e))); }}
                    className="text-xs text-zinc-500 hover:text-red-400" title="Delete this station, its seeds and all its songs">delete</button>
          </div>
        ))}
      </section>
      <footer className="text-xs text-zinc-500 font-mono flex flex-col gap-0.5">
        {err && <span className="text-red-400">{err}</span>}
        {health && Object.entries(health.sidecars).map(([n, h]) => <span key={n}>{sidecarLine(n, h)}</span>)}
        {health && <span>llm · {health.llm.model}</span>}
      </footer>
    </main>
  );
}
