"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import SongRow from "@/app/components/SongRow";
import Visualizer from "@/app/components/Visualizer";
import { deleteSong, deleteSongPrompt, getPlaylist, moveItem, playlistExportUrl, removeFromPlaylist, reorderPlaylist, songAudioUrl, type Playlist, type Song } from "@/lib/api";
import { RadioPlayer } from "@/lib/player";
import { mmss } from "@/lib/profile";

export default function PlaylistPage() {
  const { id } = useParams<{ id: string }>();
  const [pl, setPl] = useState<Playlist | null>(null);
  const [song, setSong] = useState<Song | null>(null);
  const [player, setPlayer] = useState<RadioPlayer | null>(null);
  const [scene, setScene] = useState("radial");
  const idx = useRef(0);
  const plRef = useRef<Playlist | null>(null); plRef.current = pl;
  useEffect(() => { getPlaylist(id).then(setPl).catch(() => undefined); }, [id]);
  useEffect(() => () => player?.stop(), [player]);
  const playFrom = async (i: number) => {
    if (!pl || !pl.items.length) return;
    let p = player;
    if (!p) {
      p = new RadioPlayer(songAudioUrl);
      p.onSongChange = setSong;
      p.onNeedNext = async () => { const items = plRef.current?.items ?? []; idx.current += 1; return items[idx.current] ?? null; };
      setPlayer(p);
    }
    idx.current = i;
    await p.start(pl.items[i]);
  };
  const reorder = async (from: number, to: number) => { if (!pl) return; setPl(await reorderPlaylist(pl.id, moveItem(pl.items.map((s) => s.id), from, to))); };
  if (!pl) return <main className="p-8 text-zinc-400">Loading…</main>;
  return (
    <main className="min-h-screen max-w-3xl mx-auto flex flex-col gap-6 p-8">
      <header className="flex items-baseline gap-4 flex-wrap"><Link href="/playlists" className="text-zinc-500 hover:text-zinc-200">← playlists</Link>
        <h1 className="text-3xl font-semibold tracking-tight">{pl.name}</h1><span className="text-xs text-zinc-500">{pl.items.length} songs · {mmss(pl.seconds)}</span>
        <div className="ml-auto flex gap-2 text-xs">
          <button onClick={() => playFrom(0)} disabled={!pl.items.length} className="px-3 py-1 rounded-full bg-zinc-100 text-black font-medium disabled:opacity-40">▸ Play all</button>
          {player && <button onClick={() => { player.stop(); }} className="px-3 py-1 rounded-full border border-zinc-600">■ Stop</button>}
          <a href={playlistExportUrl(pl.id)} className="px-3 py-1 rounded-full border border-zinc-700">export .zip</a>
        </div>
      </header>
      {player && <Visualizer player={player} song={song} tags={null} sceneName={scene} onScene={setScene} />}
      {song && <div className="text-sm text-zinc-300">now playing: <span className="font-medium">{song.title}</span></div>}
      {pl.items.map((s, i) => (
        <SongRow key={s.id} song={s} onChange={(u) => setPl({ ...pl, items: pl.items.map((x) => (x.id === u.id ? u : x)) })}
                 extra={<span className="flex gap-1 text-xs">
                   <button onClick={() => playFrom(i)} className="px-2 py-1 rounded border border-zinc-700">▸</button>
                   <button onClick={() => reorder(i, i - 1)} disabled={i === 0} className="px-2 py-1 rounded border border-zinc-800 disabled:opacity-30">↑</button>
                   <button onClick={() => reorder(i, i + 1)} disabled={i === pl.items.length - 1} className="px-2 py-1 rounded border border-zinc-800 disabled:opacity-30">↓</button>
                   <button onClick={async () => setPl(await removeFromPlaylist(pl.id, s.id))} className="px-2 py-1 text-zinc-500 hover:text-zinc-200" title="Take it off this playlist (the file stays in the library)">remove</button>
                   <button onClick={async () => { if (confirm(deleteSongPrompt(s))) { await deleteSong(s.id); setPl(await getPlaylist(pl.id)); } }} className="px-2 py-1 text-zinc-500 hover:text-red-400" title="Delete the song and its audio from disk">delete</button>
                 </span>} />
      ))}
    </main>
  );
}
