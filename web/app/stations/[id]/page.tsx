"use client";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { addSeedFile, addSeedUrl, deleteSeed, deleteStation, deleteStationPrompt, getStation, seedAudioUrl, type Station } from "@/lib/api";
import { TAG_ORDER, profileHeadline, seedLine, tagChips } from "@/lib/profile";
import RadioPanel from "@/app/components/RadioPanel";

export default function StationPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [st, setSt] = useState<Station | null>(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => { getStation(id).then(setSt).catch((e) => setErr(String(e))); }, [id]);
  const run = async (label: string, fn: () => Promise<Station>) => {
    setErr(null); setBusy(label);
    try { setSt(await fn()); } catch (e) { setErr(String(e)); } finally { setBusy(null); }
  };
  const onDeleteStation = async () => {
    if (!st || !confirm(deleteStationPrompt(st))) return;
    try { await deleteStation(st.id); router.push("/"); } catch (e) { setErr(String(e)); }
  };
  if (!st) return <main className="p-8 text-zinc-400">{err ?? "Loading…"}</main>;
  const p = st.profile;
  return (
    <main className="min-h-screen">
      <aside className="fixed top-0 right-0 h-screen w-full sm:w-[420px] overflow-y-auto p-4 flex flex-col gap-4 z-10">
      <header className="pane flex items-baseline gap-3">
        <Link href="/" className="text-zinc-400 hover:text-zinc-100 text-sm">← stations</Link>
        <h1 className="text-xl font-semibold tracking-tight flex-1 min-w-0 truncate">{st.name}</h1>
        <button onClick={onDeleteStation} className="text-xs text-zinc-500 hover:text-red-400 whitespace-nowrap" title="Delete this station, its seeds and all its songs">delete station</button>
      </header>
      <RadioPanel station={st} onStation={setSt}
        seedsPane={(
      <div className="flex flex-col gap-2">
        <div className="flex gap-2">
          <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="Paste a YouTube / any link…" disabled={!!busy}
                 className="flex-1 min-w-0 bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 outline-none focus:border-zinc-500" />
          <button disabled={!!busy || !url.trim()} onClick={() => run("link", () => addSeedUrl(st.id, url.trim()).then((s) => { setUrl(""); return s; }))}
                  className="px-3 py-2 rounded-lg bg-zinc-100 text-black font-medium disabled:opacity-40 whitespace-nowrap">Add</button>
          <button disabled={!!busy} onClick={() => fileRef.current?.click()} className="px-3 py-2 rounded-lg border border-zinc-700 disabled:opacity-40 whitespace-nowrap">Upload</button>
          <input ref={fileRef} type="file" accept="audio/*,video/*" className="hidden"
                 onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) void run("file", () => addSeedFile(st.id, f)); }} />
        </div>
        {busy && <p className="text-sm text-zinc-400 animate-pulse">Fetching and listening ({busy}) — transcription + sound tags take ~10 s per song…</p>}
        {err && <p className="text-sm text-red-400">{err}</p>}
        {st.seeds.map((s) => (
          <div key={s.id} className="border border-zinc-800 rounded-xl px-4 py-3 flex flex-col gap-2">
            <div className="flex items-start gap-3">
              <div className="flex-1 min-w-0">
                <div className="font-medium truncate" title={s.title}>{s.title}</div>
                <div className="text-xs text-zinc-400">{seedLine(s)}</div>
              </div>
              <button onClick={() => run("delete", () => deleteSeed(s.id))} className="text-xs text-zinc-500 hover:text-red-400 whitespace-nowrap">remove</button>
            </div>
            <audio src={seedAudioUrl(s.id)} controls preload="none" className="h-8 w-full" />
            {s.style_guess && <div className="text-xs text-zinc-500">sounds like: {s.style_guess}</div>}
            {s.promoted_sections?.length ? <div className="text-xs text-amber-300">tune sits in the instrument line in: {s.promoted_sections.join(", ")} — covers will sing it</div> : null}
          </div>
        ))}
      </div>
        )}
        profilePane={(
      <div className="flex flex-col gap-2">
        {!p && <p className="text-zinc-500 text-sm">Add a seed to build the profile the agent composes from.</p>}
        {p && (
          <div className="border border-zinc-800 rounded-xl px-4 py-3 flex flex-col gap-3">
            <div className="font-medium">{profileHeadline(p)}</div>
            <div className="text-sm text-zinc-300">{p.style}</div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
              {TAG_ORDER.map((cat) => (
                <div key={cat}><div className="uppercase tracking-widest text-zinc-500 mb-1">{cat}</div>
                  <div className="flex flex-wrap gap-1">{tagChips(p, cat).map((c) => <span key={c} className="px-2 py-0.5 rounded-full bg-zinc-900 border border-zinc-800">{c}</span>)}</div></div>
              ))}
            </div>
            <div className="text-xs text-zinc-400">form: {p.sections.join(" → ")}</div>
            {Object.keys(p.phrases).length > 0 && <div className="text-xs text-zinc-500">lines per section: {Object.entries(p.phrases).map(([k, v]) => `${k} ${v.length}×(${v.join(",")})`).join(" · ")}</div>}
          </div>
        )}
      </div>
        )} />
      </aside>
    </main>
  );
}